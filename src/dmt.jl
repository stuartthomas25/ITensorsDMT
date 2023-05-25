using LinearAlgebra

struct DMT <: TruncationMethod end
struct NaiveTruncation <: TruncationMethod end



const BlockSparseTensor = NDTensors.BlockSparseTensor
const DenseTensor = NDTensors.DenseTensor

# for generalizing `dmt` to dense matrices
ITensors.blockview(T::DenseTensor, ::Block) = T
ITensors.nzblocks(T::DenseTensor) = [Block(1,1)]
const blockview = ITensors.blockview

const tensor = ITensors.tensor
Base.one(::Type{ITensor}) = ITensor(1.)

rank(S::Vector{<:Number}, cutoff::Float64) = something( findfirst(x->x<cutoff, S), length(S) + 1) - 1

function _empty_Qstorage(T::BlockSparseTensor; tags="Link,q")
    u = first(inds(T))
    q = Index(u.space; tags, dir=-dir(u) )
    nzblocksQ = [Block(b[1], b[1]) for b=nzblocks(T)]
    BlockSparseTensor(eltype(T), undef, nzblocksQ, (u,q))
end

function _empty_Qstorage(T::DenseTensor, tags="Link,q")
    u = first(inds(T))
    q = Index(u.space; tags)
    DenseTensor(eltype(T), undef, (u,q))
end


"""
`
full_Q(iT::ITensor, u::Index, μ::Index; tags="Link,q")
`

Calculate the full QR composition of a BlockSparse ITensor and return the non-thin orthogonal Q part

"""
function full_Q(iT::ITensor, u::Index, μ::Index)
    T = tensor( permute(iT, u, μ; allow_alias=true) )
    QT = _empty_Qstorage(T)
    q = last(inds(QT))

    for (b, bQ) in zip( nzblocks(T), nzblocks(QT) )
        M = blockview(T, b)
        Qthin,_ = M|>matrix|>qr
        Qblock = Qthin[:,:] # get full (non-thin) matrix

        blockview(QT, bQ) .= Qblock
    end
    itensor(QT),q
end

# for sparse
function relevant_dims(x::Index{<:Vector}, q::Index{<:Vector}, b::Block)
    dims = Dict(x.space)
    dims[qn(q, b[1])]
end
function flux(q::Index{<:Vector}, b::Block; kind::String="Nf")
    val(qn(q, b[1]), kind)
end

# for dense
relevant_dims(x::Index{Int64}, ::Index{Int64}, ::Block) = dim(x)
flux(q::Index{Int64}, b::Block; kind::String="Nf") = 0

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
    cutoff::Float64=0.
        )::Tuple{ITensor, ITensor}

    maxdim >= 8 || throw("Max dim must be greater than or equal to 7")

    sites = sort(inds(ϕ, "Site"), by=sitepos)
    ns = sitepos.(sites)
    leftlink = commonind(Lsum, ϕ)
    Lis = isnothing(leftlink) ? IndexSet(sites[1]) : IndexSet((leftlink, sites[1]))

    # take the first SVD
    U, S, Vt, _, u, v  = svd(ϕ, Lis...; cutoff=cutoff)

    # Calculate the change of basis matrices QL and QR
    #if ϕ has more than two sites, trace over the extra ones for now
    xR = foldl(*, [Vt, Rsum, (tracer(x) for x in sites[3:end])... ])
    xL = Lsum * U

    QL,qL = full_Q(xL, u, sites[1])
    QR,qR = full_Q(xR, v, sites[2])
    M = QL * S * conj(QR)

    nzblocksM = nzblocks(M)
    @assert all( b[1]==b[2] for b in nzblocksM )
    unconnected_component = nothing

    svds = map(nzblocksM) do b
        bM = matrix(blockview(M.tensor, b))
        i = relevant_dims(sites[1], qL, b)
        bMsub = bM[i+1:end, i+1:end]

        if flux(qL, b)|>iszero && abs(bM[1,1]) > 1e-10
            unconnected_component = (bM[i+1:end,1:1] * bM[1:1,i+1:end]) / bM[1,1]
            bMsub = bMsub - unconnected_component
        end

        svd(bMsub)
    end

    Ss = sort([(res.S for res in svds)...;]; rev=true)
    cut = min( rank(Ss, cutoff), maxdim-8)
    new_cutoff = get(Ss, cut+1, 0.)

    for (svd2, b) in zip(svds, nzblocksM)
        U2, S2, Vt2 = svd2.U, svd2.S, svd2.Vt
        S2 = [ x>=new_cutoff ? x : 0. for x∈S2 ]
        new_bMsub = U2 * Diagonal(S2) * Vt2

        if flux(qL,b)|>iszero && !isnothing(unconnected_component)
            new_bMsub .+= unconnected_component
        end

        m = length(S2)
        blockview(M.tensor, b)[end-m+1:end, end-m+1:end] .= new_bMsub
    end

    U3, S3, Vt3, _, u3 = svd(M, qL; cutoff)
    newlink = sim(u3; tags="Link,l=$(ns[1])")

    A = replaceind(U * dag(QL) * U3,  u3, newlink)
    B = replaceind(S3 * Vt3 * dag(conj(QR)) * Vt, u3, newlink)
    A, B
end


function apply!(gates::Vector{ITensor}, ρ::MPS, ::DMT; kwargs...)
    for o=gates
        apply!(o, ρ, DMT(); kwargs...)
    end
end

function apply!(o::ITensor, ρ::MPS, ::DMT; kwargs...)
    ns = sort(findsites(ρ, o))
    N  = length(ns)
    x = ns[1]

    orthogonalize!(ρ, ns[1])

    ϕ = foldl(*, [ρ[n] for n=ns])
    ϕ = product(o, ϕ)

    ρsums = map(ρ) do T
        x = only(inds(T; tags="Site", plev=0))
        T * tracer(x)
    end

    Lsum = foldl(*, ρsums[1:ns[1]-1])
    Rsum = foldl(*, ρsums[ns[end]+1:end])

    ψ = Vector{ITensor}(undef, N)
    for n in 1:(N-1)
        L, R = dmt(ϕ, Lsum, Rsum; kwargs...)
        ψ[n] = L
        Lsum = Lsum * (L * tracer(inds(L; tags="Site") |> only))
        ϕ = R
    end
    ψ[N] = ϕ

    newρ = MPS(ψ)

    # following ITensors/mps/abstractmps.jl:1889
    ITensors.setleftlim!(newρ, N - 1)
    ITensors.setrightlim!(newρ, N + 1)
    orthogonalize!(newρ, ns[end] - ns[1] + 1)

    ρ[ns[1]:ns[end]] = newρ

end


function apply!(gates::Vector{ITensor}, ψ::ITensors.AbstractMPS, ::NaiveTruncation; kwargs...)
    newψ = apply(gates, ψ; kwargs...)
    ψ.data = newψ.data
    ψ.rlim = newψ.rlim
    ψ.llim = newψ.llim
    nothing
end

"""
default to `NaiveTruncation`
"""
apply!(gates::Vector{ITensor}, ψ::ITensors.AbstractMPS; kwargs...) =
    apply!(gates, ψ, NaiveTruncation(); kwargs...)

# dense DMT

function qrDMT(x::ITensor)
    μ = inds(x; tags="Site")[1]
    α = inds(x; tags="Link")[1]
    A = Array(x, (α, μ))
    res = qr(A)
    m = ITensors.dim(α)
    res.Q*Matrix(I,m,m), Matrix(res.R) # QR decomp is thin by default
end

export apply!,
       DMT,
       NaiveTruncation
