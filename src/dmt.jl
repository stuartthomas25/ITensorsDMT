using LinearAlgebra

struct DMT <: TruncationMethod end
struct NaiveTruncation <: TruncationMethod end



BlockSparseTensor = NDTensors.BlockSparseTensor
blockview = ITensors.blockview
tensor = ITensors.tensor
Base.one(::Type{ITensor}) = ITensor(1.)


rank(S::Vector{<:Number}, cutoff::Float64) = something( findfirst(x->x<cutoff, S), length(S) + 1) - 1

"""
`
full_Q(iT::ITensor, u::Index, μ::Index; tags="Link,q")
`

Calculate the full QR composition of a BlockSparse ITensor and return the non-thin orthogonal Q part

"""
function full_Q(iT::ITensor, u::Index, μ::Index; tags="Link,q")
    q = Index(u.space; tags, dir=-dir(u) )
    T = tensor( permute(iT, u, μ; allow_alias=true) )
    nzblocksQ = [Block(b[1], b[1]) for b=nzblocks(T)]
    Qstorage = BlockSparseTensor(eltype(iT), undef, nzblocksQ, (u,q))

    for (b, bQ) in zip( nzblocks(T), nzblocksQ )
        M = blockview(T, b)
        Qthin,_ = M|>matrix|>qr
        Qblock = Qthin[:,:] # get full (non-thin) matrix

        blockview(Qstorage, bQ) .= Qblock
    end
    itensor(Qstorage),q
end

"""
Apply Density Matrix Truncation to the left-most two sites of a multi-site tensor.

Lsum and Rsum are the traces of the MPO to the left and right of ϕ.

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

    U, S, Vt, _, u, v  = svd(ϕ, Lis...; cutoff=cutoff)

    #if ϕ has more than two sites, trace over the extra ones for now
    xR = foldl(*, [Vt, Rsum, (tracer(x) for x in sites[3:end])... ])
    xL = Lsum * U

    QL,qL = full_Q(xL, u, sites[1]; tags="Link,qL")
    QR,qR = full_Q(xR, v, sites[2]; tags="Link,qR")
    M = dag(conj(QL)) * S * dag(QR)

    relevant_dims = Dict(sites[1].space)
    nzblocksM = nzblocks(M)
    @assert all( b[1]==b[2] for b in nzblocksM )
    unconnected_component = nothing

    svds = map(nzblocksM) do b
        qn_rule = qn(qL, b[1])
        flux = val(qn_rule, "Nf")
        bM = matrix(blockview(M.tensor, b))
        i = relevant_dims[qn_rule]
        bMsub = bM[i+1:end, i+1:end]

        if iszero(flux) && abs(bM[1,1]) > 1e-10
            unconnected_component = (bM[i+1:end,1:1] * bM[1:1,i+1:end]) / bM[1,1]
            bMsub = bMsub - unconnected_component
        end

        svd(bMsub)
    end

    Ss = sort([(res.S for res in svds)...;]; rev=true)
    cut = min( rank(Ss, cutoff), maxdim-9)
    new_cutoff = get(Ss, cut+1, 0.)

    for (svd2, b) in zip(svds, nzblocksM)
        U2, S2, Vt2 = svd2.U, svd2.S, svd2.Vt
        S2 = [ x>=new_cutoff ? x : 0. for x∈S2 ]
        new_bMsub = U2 * Diagonal(S2) * Vt2

        flux = val(qn(qL, b[1]), "Nf")
        if iszero(flux) && !isnothing(unconnected_component)
            new_bMsub .+= unconnected_component
        end

        m = length(S2)
        blockview(M.tensor, b)[end-m+1:end, end-m+1:end] .= new_bMsub
    end

    U3, S3, Vt3, _, u3 = svd(M, qL; cutoff)
    newlink = sim(u3; tags="Link,l=$(ns[1])")

    A = replaceind(U * conj(QL) * U3,  u3, newlink)
    B = replaceind(S3 * Vt3 * QR * Vt, u3, newlink)
    A, B
end


function apply!(gates::Vector{ITensor}, ρ::MPS, ::DMT; kwargs...)
    for o=gates
        apply!(o, ρ, DMT(); kwargs...)
    end
end

function apply!(o::ITensor, ρ::MPS, ::DMT; maxdim::Int64=64, cutoff::Float64=1e-16)
    ns = sort(findsites(ρ, o))
    N  = length(ns)
    x = ns[1]

    # oldl = linkinds(ρ)

    orthogonalize!(ρ, ns[1])
    # newl = linkinds(ρ)

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
        L, R = dmt(ϕ, Lsum, Rsum; maxdim, cutoff)
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

apply!(gates::Vector{ITensor}, ψ::ITensors.AbstractMPS; kwargs...) =
    apply!(gates, ψ, NaiveTruncation(); kwargs...)

function apply!(gates::Vector{ITensor}, ψ::MPO; kwargs...)
    newψ = apply(gates, ψ; kwargs...)
    ψ.data = newψ.data
    ψ.rlim = newψ.rlim
    ψ.llim = newψ.llim

end

# dense DMT

function qrDMT(x::ITensor)
    μ = inds(x; tags="Site")[1]
    α = inds(x; tags="Link")[1]
    A = Array(x, (α, μ))
    res = qr(A)
    m = ITensors.dim(α)
    res.Q*Matrix(I,m,m), Matrix(res.R) # QR decomp is thin by default
end

function dense_dmt(ϕ::ITensor, Lsum::ITensor, Rsum::ITensor; maxdim::Int64=Inf, cutoff::Float64=1e-12)::Tuple{ITensor, ITensor}
    maxdim >= 7 || throw("Max dim must be greater than or equal to 7")

    sites = sort(inds(ϕ, "Site"), by=sitepos)
    ns = sitepos.(sites)
    leftlink = commonind(Lsum, ϕ)

    Lis = isnothing(leftlink) ? IndexSet(sites[1]) : IndexSet((leftlink, sites[1]))

    U, S, Vt, spec, u, v  = svd(ϕ, Lis...; alg="divide_and_conquer", cutoff=cutoff)

    xL = Lsum * U
    QL, RL = qrDMT(xL)

    SA = Array(S, (u, v))

    xR = Vt * Rsum
    #if ϕ has more than two sites, trace over the extra ones
    for s in sites[3:end]
        xR = xR * onehot(dag(s)=>1)
    end
    QR, RR = qrDMT(xR)

    M = conj(QL') * SA * QR
    removeUnconnectedComponent = abs(RL[1,1] * M[1,1] * RR[1,1]) > 1e-10 # do not remove connected component if it is traceless

    unconnectedComponent = removeUnconnectedComponent ? (M[:,1:1] * M[1:1,:]) / M[1,1] : zeros(size(M))
    M̃ = M - unconnectedComponent
    Msub = M[5:end, 5:end]
    svd2 = svd(Msub)
    U2, S2, Vt2 = svd2.U, svd2.S, svd2.Vt

    #truncate
    cut = min( rank(S2, cutoff), maxdim-8)
    S2[cut+1:end] .= 0.
    M̃[5:end,5:end] = U2 * Diagonal(S2) * Vt2
    M = M̃ + connectedComponent

    try
        svd(M)
    catch
        @show M
        throw("SVD error")
    end
    svd3 = svd(M)

    U3, S3, Vt3 = svd3.U, svd3.S, svd3.Vt

    cut = max(rank(S3, cutoff), 1)

    if !(0 < cut <= maxdim)
        @show cut
        @show maxdim
        @show S3
    end

    @assert 0 < cut <= maxdim

    S3 = S3[1:cut]
    LA = conj(QL) * U3[:, 1:cut]
    RA = Diagonal(S3[1:cut]) * Vt3[1:cut, :] * QR'

    newlink = Index(cut, tags="Link,l=$(ns[1])")

    L,R = ITensor(LA, (u, newlink)), ITensor(RA, (newlink, v))

    U * L, R * Vt
end

export apply!,
       DMT,
       NaiveTruncation
