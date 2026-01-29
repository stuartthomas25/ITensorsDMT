using LinearAlgebra
using ITensors.NDTensors
using ITensorMPS: set_leftlim!, set_rightlim!
import Distributed

struct DMT <: TruncationMethod end
struct NaiveTruncation <: TruncationMethod end
const iscu = NDTensors.iscu

# for generalizing `dmt` to dense matrices
ITensors.blockview(T::DenseTensor, ::Block) = T
ITensors.nzblocks(::DenseTensor) = [Block(1,1)]

Base.one(::Type{ITensor}) = ITensor(1.)

rank(S::AbstractVector{<:Number}, cutoff::Float64) = something( findfirst(x->abs(x)<cutoff, S), length(S) + 1) - 1

function _empty_Qstorage(T::BlockSparseTensor; tags="Link,q")
    use_gpu = NDTensors.iscu(T)
    cu_or_not(x) = use_gpu ? cu(x) : x

    u = first(inds(T))
    q = Index(u.space; tags, dir=-dir(u) )
    nzblocksQ = [(Block(i,i) for i in eachindex(space(q)))...]
    cu_or_not( BlockSparseTensor(eltype(T), undef, nzblocksQ, (u,q)) )
end

function _empty_Qstorage(T::DenseTensor, tags="Link,q")
    use_gpu = NDTensors.iscu(T)
    cu_or_not(x) = use_gpu ? cu(x) : x

    u = first(inds(T))
    q = Index(u.space; tags)
    cu_or_not( DenseTensor(eltype(T), undef, (u,q)) )
end


"""
`
full_Q(iT::ITensor, u::Index, μ::Index; tags="Link,q")
`

Calculate the full QR composition of a BlockSparse ITensor and return the non-thin orthogonal Q part

"""
function full_Q(iT::ITensor, u::Index, μ::Index)

    use_gpu = iscu(iT)
    cu_or_not_Matrix(x...) = use_gpu ? CuMatrix(x...) : Matrix(x...)

    T = tensor( permute(iT, u, μ; allow_alias=true) )
    QT = _empty_Qstorage(T)
    q = last(inds(QT))

    for b in nzblocks(QT)
        Tblocks = findall(b_->b_[1] == b[1], nzblocks(T))
        if !isempty(Tblocks)
            M = blockview(T, nzblocks(T)[only(Tblocks)])
            Qthin,_ = M|>matrix|>qr
            # @show typeof(Qthin)
            # Qblock = collect(Qthin[:,:])
            Qblock = Qthin * cu_or_not_Matrix(I,size(Qthin)...) # make dense

            blockview(QT, b).storage .= Dense(Qblock) # assign storage directly for GPU compatibility
            # blockview(QT, b) .= Qblock # assign storage directly for GPU compatibility
        else
            dim = space(u)[b[1]]|>last
            blockview(QT, b).storage .= Dense(cu_or_not_Matrix(I, dim, dim))
            # blockview(QT, b) .= cu_or_not_Matrix(I, dim, dim)
        end
    end
    itensor(QT), q
end

# for sparse
function relevant_dims(x::Index{<:Vector}, q::Index{<:Vector}, b::Block)
    dims = Dict(x.space)
    get(dims, qn(q, b[1]), 0)
end

function remove_smallest(v::T, cutoff)::T where {T}
    map(s->abs(s)>cutoff ? s : zero(s), v)
end

# for dense
relevant_dims(x::Index{Int64}, ::Index{Int64}, ::Block) = dim(x)

"""
Apply Density Matrix Truncation to the left-most two sites of a multi-site tensor.

Lsum and Rsum are the traces of the MPO to the left and right of ϕ. These are passed
since they can be

If ϕ has more than 2 site indices, truncate between the left two.
"""
function dmt(
    ϕ::ITensor,
    Lsum::ITensor,
    Rsum::ITensor;
    maxdim::Int64=typemax(Int64),
    cutoff::Float64=0.,
    remove_unconnected_component=true,
    ortho="left"
        )::Tuple{ITensor, ITensor, <:Spectrum}

    use_gpu = iscu(ϕ)
    _map = (ITensors.using_threaded_blocksparse() && !use_gpu) ? Distributed.pmap : map
    cu_or_not(x) = use_gpu ? cu(x) : x

    sites = sort([inds(ϕ, "Site")...], by=sitepos)
    ns = sitepos.(sites)
    leftlink = commonind(Lsum, ϕ)
    Lis = isnothing(leftlink) ? IndexSet(sites[1]) : IndexSet((leftlink, sites[1]))

    # take the first SVD
    U, S, Vt, _, u, v  = svd(ϕ, Lis...; cutoff)

    # Calculate the change of basis matrices QL and QR
    #if ϕ has more than two sites, trace over the extra ones for now
    xR = foldl(*, [Vt, Rsum, (cu_or_not(tracer(x)) for x in sites[3:end])... ])
    xL = Lsum * U

    QL, qL = full_Q(xL, u, sites[1])
    QR, qR = full_Q(xR, v, sites[2])
    M = QL * S * QR

    nzblocksM = nzblocks(M)
    # display(nzblocksM)

    @assert all( b[1]==b[2] for b in nzblocksM ) # ensure M is block diagonal
    unconnected_component = nothing

    uc_block = if remove_unconnected_component # block with the unconnected component
        nzblocksM[ findfirst(b->qL isa Index{Int} || iszero(flux((qL,qR), b)), nzblocksM) ]
    else
        Block()
    end

    svds = CUDA.@allowscalar _map(nzblocksM) do b
        bM = matrix(blockview(M.tensor, b))
        i = relevant_dims(sites[1], qL, b) # the rows that affect length 2 operators, equal to the onsite space dimension
        bMsub = bM[i+1:end, i+1:end]

        if b==uc_block && abs(bM[1,1]) > 1e-10
            isnothing(unconnected_component) || throw("multiple unconnected components defined")
            unconnected_component = (bM[i+1:end,1:1] * bM[1:1,i+1:end]) / bM[1,1]
            # @info "subtracting unconnected_component"
            bMsub .-= unconnected_component
        end

        if isempty(bMsub) # CUDA has issues with empty matrices
            return SVD(bMsub, eltype(bMsub)[], bMsub)
        end

        svd(bMsub; full=true)
    end


    total_relevant_dims = sum( relevant_dims(sites[1], qL, b) for b in nzblocksM )
    rank_offset = 2total_relevant_dims

    maxdim >= rank_offset || throw("Max dim must be greater than or equal to $rank_offset")

    # display( [(res.S for res in svds)...;] )
    # display(svds)

    CUDA.@allowscalar if !isempty(svds) # CUDA has an issue sorting empty arrays
        Ss = sort([(res.S for res in svds)...;]; rev=true, by=abs)
    else
        Ss = [(res.S for res in svds)...;]
    end

    # @show typeof(Ss)
    cut = min( rank(Ss, cutoff), maxdim-rank_offset)
    new_cutoff = CUDA.@allowscalar abs(get(Ss, cut, 0.))

    for (svd2, b) in zip(svds, nzblocksM)
        U2, S2, Vt2 = svd2.U, svd2.S, svd2.Vt
        # S2 = cu_or_not( [ x>=new_cutoff ? x : 0. for x∈S2 ] )
        # @show remove_smallest(S2, new_cutoff)
        # @show cu_or_not( remove_smallest(S2, new_cutoff) )
        S2 = cu_or_not( remove_smallest(S2, new_cutoff) )
        # @show S2
        new_bMsub = U2 * Diagonal(S2) * Vt2

        if b==uc_block && !isnothing(unconnected_component)
            new_bMsub .+= unconnected_component
        end

        m = length(S2)
        # blockview(M.tensor, b)[end-m+1:end, end-m+1:end] .= new_bMsub

        blockM = blockview(M.tensor, b)
        blockMinds = LinearIndices(blockM)[end-m+1:end, end-m+1:end]
        blockM.storage.data[blockMinds] .= new_bMsub

    end

    ϕ′ = U * dag(QL) * M * dag(QR) * Vt
    U3, S3, Vt3, spec, u3 = svd(ϕ′, Lis...; cutoff)
    newlink = sim(u3; tags="Link,l=$(ns[1])")

    if ortho=="left"
        L, R = U3, S3 * Vt3
    elseif ortho=="right"
        L, R = U3 * S3, Vt3
    else
        error("In `dmt`, ortho keyword $ortho not supported. Supported options are `left` or `right`.")
    end
    L = replaceind(L, u3, newlink)
    R = replaceind(R, u3, newlink)
    L, R, spec
end


function apply!(gates::Vector{ITensor}, ρ::MPS, ::DMT; kwargs...)
    for o=gates
        apply!(o, ρ, DMT(); kwargs...)
    end
end

function apply!(o::ITensor, ρ::MPS, ::DMT; kwargs...)
    use_gpu = NDTensors.iscu(o)
    cu_or_not(x) = use_gpu ? cu(x) : x

    ns = sort(findsites(ρ, o))
    isempty(ns) && throw("Gate and MPS do not share sites")
    N  = length(ns)
    x = ns[1]

    orthogonalize!(ρ, ns[1]+1)

    ϕ = foldl(*, [ρ[n] for n=ns])

    # check if this is prop to identity
    id_tensor = contract((cu_or_not(hastags(i, "Site") ? state("Id", i) : onehot(i=>1)) for i=inds(ϕ))...)
    if isapprox(ϕ ./ sum(ϕ), id_tensor; atol=1e-2)
        return
    end

    ϕ = product(o, ϕ)

    ρsums = map(ρ) do T
        x = only(inds(T; tags="Site", plev=0))
        T * cu_or_not(tracer(x))
    end

    Lsum = foldl(*, ρsums[1:ns[1]-1])
    Rsum = foldl(*, ρsums[ns[end]+1:end])

    ψ = Vector{ITensor}(undef, N)
    for n in 1:(N-1)
        L, R = dmt(ϕ, Lsum, Rsum; kwargs...)
        ψ[n] = L
        x = inds(L; tags="Site") |> only
        Lsum = Lsum * (L * cu_or_not(tracer(x)))
        ϕ = R
    end
    ψ[N] = ϕ

    newρ = MPS(ψ)

    # following ITensors/mps/abstractmps.jl
    set_leftlim!(newρ, N - 1)
    set_rightlim!(newρ, N + 1)
    orthogonalize!(newρ, ns[end] - ns[1] + 1)

    ρ[ns[1]:ns[end]] = newρ

end


function apply!(gates::Vector{ITensor}, ψ::ITensorMPS.AbstractMPS, ::NaiveTruncation; kwargs...)
    newψ = apply(gates, ψ; kwargs...)
    ψ.data = newψ.data
    ψ.rlim = newψ.rlim
    ψ.llim = newψ.llim
    nothing
end

"""
default to `NaiveTruncation`
"""
apply!(gates::Vector{ITensor}, ψ::ITensorMPS.AbstractMPS; kwargs...) =
    apply!(gates, ψ, NaiveTruncation(); kwargs...)

export apply!,
       DMT,
       NaiveTruncation
