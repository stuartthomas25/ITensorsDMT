using LinearAlgebra

struct DMT <: TruncationMethod end
struct NaiveTruncation<: TruncationMethod end

Base.one(::Type{ITensor}) = ITensor(1.)

function qrDMT(x::ITensor)::Tuple{Matrix{Float64}, Matrix{Float64}}
    μ = inds(x; tags="Site")[1]
    α = inds(x; tags="Link")[1]
    A = Array{Float64}(x, (α, μ))
    res = qr(A)
    m = ITensors.dim(α)
    res.Q*Matrix(I,m,m), Matrix(res.R) # QR decomp is thin by default
end

rank(S::Vector{Float64}, cutoff::Float64) = something( findfirst(x->x<cutoff, S), length(S) + 1) - 1

function dmt(ϕ::ITensor, Lsum::ITensor, Rsum::ITensor; maxdim::Int64=Inf, cutoff::Float64=1e-12)::Tuple{ITensor, ITensor}
    maxdim >= 7 || throw("Max dim must be greater than or equal to 7")

    sites = sort(inds(ϕ, "Site"), by=sitepos)
    ns = sitepos.(sites)
    leftlink = commonind(Lsum, ϕ)

    #if not the leftmost
    Lis = isnothing(leftlink) ? IndexSet(sites[1]) : IndexSet((leftlink, sites[1]))

    USV = svd(ϕ, Lis...; alg="divide_and_conquer", cutoff=cutoff)

    U, S, Vt, spec, u, v = USV
    SA = Array(S, (u,v))

    xL = Lsum * U
    QL, RL = qrDMT(xL)

    xR = Vt * Rsum
    QR, RR = qrDMT(xR)

    M = conj(QL') * SA * QR
    removeConnectedComponent = abs(RL[1,1] * M[1,1] * RR[1,1]) > 1e-10 # do not remove connected component if it is traceless

    connectedComponent = removeConnectedComponent ? (M[:,1:1] * M[1:1,:]) / M[1,1] : zeros(size(M))
    M̃ = M - connectedComponent
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


tracetensor(T::ITensor)::ITensor = T * onehot(inds(T; tags="Site,Pauli", plev=0)[1] => 1)

function apply!(gates::Vector{ITensor}, ρ::MPS, ::DMT; maxdim::Int64=64, cutoff::Float64=1e-16)
    all(hastags.(ρ, "Pauli")) || throw("MPS must be in Pauli basis")

    for o=gates
        hastags(o, "Pauli") || throw("Gate must be in Pauli basis")

        ns = findsites(ρ, o)
        N  = length(ns)
        @assert N==2
        ns = sort(ns)
        x = ns[1]


        oldl = linkinds(ρ)

        orthogonalize!(ρ, ns[1])
        newl = linkinds(ρ)
        # for i=1:length(ρ)
        #     replaceinds!(ρ[i], newl, oldl) # keep old indices
        # end

        ϕ = reduce(*, [ρ[n] for n=ns])
        ϕ = product(o, ϕ)

        ρsums = tracetensor.(ρ)
        Lsum = reduce(*, ρsums[1:ns[1]-1])
        Rsum = reduce(*, ρsums[ns[2]+1:end])

        L, R = dmt(ϕ, Lsum, Rsum; maxdim, cutoff)

        newρ = MPS([L,R])
        ITensors.setleftlim!(newρ, N - 1)
        ITensors.setrightlim!(newρ, N + 1)
        orthogonalize!(newρ, ns[end] - ns[1] + 1)


        ρ[ns[1]:ns[end]] = newρ
    end

end

export apply!


function apply!(gates::Vector{ITensor}, ψ::ITensors.AbstractMPS, ::NaiveTruncation; kwargs...)
    newψ = apply(gates, ψ; kwargs...)
    ψ.data = newψ.data
    ψ.rlim = newψ.rlim
    ψ.llim = newψ.llim
end

apply!(gates::Vector{ITensor}, ψ::ITensors.AbstractMPS; kwargs...) = apply!(gates, ψ, NaiveTruncation(); kwargs...)
