Base.zero(::Type{ITensor}) = ITensor(0.)

function Ising(s::Vector{<:Index})::Vector{ITensor}
    map(1:length(s)-1) do j
        op("Sz",s[j]) * op("Sz",s[j+1])
    end
end



function Trivial(s::Vector{<:Index})::Vector{ITensor}
    map(1:length(s)-1) do j
        op("Id",s[j]) * op("Id",s[j+1])
    end
end

function TFIM(s::Vector{<:Index}; b::Union{Vector{<:Real},Real}=0.)::Vector{ITensor}
    hs = ITensor[]
    N = length(s)
    if (typeof(b) <: Number) b = fill(b, N) end
    for j=1:N-1
        s1,s2  = s[j], s[j+1]
        hj =                op("Sz",s1) * op("Sz",s2)    +
                      1/2 * op("Sz",s1) * op("Id",s2)    +
               (1+b[j])/2 * op("Sx",s1) * op("Id",s2)    +
                      1/2 * op("Id",s1) * op("Sz",s2)    +
               (1+b[j])/2 * op("Id",s1) * op("Sx",s2)

        if j==1
            hj +=     1/2 * op("Sz",s1) * op("Id",s2)    +
               (1+b[j])/2 * op("Sx",s1) * op("Id",s2)
        elseif j==N-1
            hj +=     1/2 * op("Id",s1) * op("Sz",s2)    +
               (1+b[j])/2 * op("Id",s1) * op("Sx",s2)
        end
        push!(hs, hj)
    end
    hs
end

function MFIM(s::Vector{<:Index}; hx::Float64=1.4, hz::Float64=0.9045)::Vector{ITensor}
    hs = ITensor[]
    N = length(s)
    for j=1:N-1
        s1,s2  = s[j], s[j+1]
        hj =            4 * op("Sz",s1) * op("Sz",s2)  +
                       hz * op("Sz",s1) * op("Id",s2)  + # factor of 2 from Pauli matrices cancels with 1/2 from Trotterization
                       hx * op("Sx",s1) * op("Id",s2)  +
                       hz * op("Id",s1) * op("Sz",s2)  +
                       hx * op("Id",s1) * op("Sx",s2)

        if j==1
            hj +=      hz * op("Sz",s1) * op("Id",s2)  +
                       hx * op("Sx",s1) * op("Id",s2)
        elseif j==N-1
            hj +=      hz * op("Id",s1) * op("Sz",s2)   +
                       hx * op("Id",s1) * op("Sx",s2)
        end
        push!(hs, hj)
    end
    hs
end

function XXX(s::Vector{Index{Int64}};J::Float64=1.0)
    hs = ITensor[]
    N = length(s)
    for j=1:N-1
        s1,s2  = s[j], s[j+1]
        hj =             J * op("Sx",s1) * op("Sx",s2)    +
                         J * op("Sy",s1) * op("Sy",s2)    +
                         J * op("Sz",s1) * op("Sz",s2)
        push!(hs, hj)
    end
    hs
end


function infTempState(s::Vector{<:Index})::MPO
    ρ₀ = MPO(s, ["Id" for x in s])
    for i=1:length(s)
        ρ₀[i] ./= sqrt(dim(s[1]))
    end
    ρ₀
end

spinval(ρ::MPS, opname::String) = spinval(convertToSpin(ρ), opname)
function spinval(ρ::MPO, opname::String)::Vector{Float64}
    Os = [op(opname, s) for s=siteinds(first, ρ; plev=0)]
    localop(ρ, Os)
end

energy_density(ρ::ITensors.AbstractMPS, ham::Vector{ITensor}) = localop(ρ, ham)

""" Energy current operator"""
current_operator(ham::Vector{ITensor}) = current_operator(ham, ham)

""" Calculate the current operator of a charge `O` under time evolution of `ham` """
function current_operator(ham::Vector{ITensor}, O::Vector{ITensor})::Vector{ITensor}
    js = [ITensor(0.)] # set first current to 0
    for On in O[1:end-1]
        relevantTerms = filter(h->!isempty(commoninds(h, On)), ham)
        ∂thTerms = map(t -> 1im * (product(t, On) - product(On, t)), relevantTerms)
        ∂th = sum(addIdentities(∂thTerms...))
        ∂th, prevj = addIdentities(∂th, js[end])
        j = prevj - ∂th

        push!(js, removeIdentities(j))
    end
    push!(js, ITensor(0.));
    clean.(js)
end

current_operator(ham::Vector{ITensor}, O::ITensor) = current_operator(ham, [O])


function entanglement_entropy(ψ₀::ITensors.AbstractMPS)::Vector{Float64}
    N = length(ψ₀)
    ψ = deepcopy(ψ₀)
    S = Float64[]
    for b=2:N
        orthogonalize!(ψ, b)
        push!(S, entanglement_entropy(ψ, b))
    end
    return S
end


function entanglement_entropy(ψ₀::MPS, b::Int)::Float64
    # ψ = orthogonalize(ψ₀, b)
    # truncate!(ψ; cutoff=1e-10)
    ψ = deepcopy(ψ₀)

    U,S,V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ,b)) )
    S /= norm(S)
    # ptot = 0.0
    SvN = 0.0
    for n=1:NDTensors.dim(S, 1)
        p = abs(S[n,n])^2
        # ptot += p
        SvN -= p * log2(p)
    end
    return SvN
end

function entanglement_entropy(ψ₀::MPO, b::Int)::Float64
    # ψ = orthogonalize(ψ₀, b)
    # truncate!(ψ; cutoff=1e-10)
    ψ = deepcopy(ψ₀)

    U,S,V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ,b), siteind(ψ,b)') )
    S /= norm(S)
    # ptot = 0.0
    SvN = 0.0
    for n=1:NDTensors.dim(S, 1)
        p = abs(S[n,n])^2
        # ptot += p
        SvN -= p * log2(p)
    end
    return SvN
end

function vonneumann(ρ::MPO)
    s = siteinds(first, ρ; plev=0)
    N = length(s)
    S = 0.
    for el in Iterators.product(fill([1,2], N)...)
        V = ITensor(1.)
        for j=1:N
            V *= ρ[j] * state(s[j],el[j]) * state(s[j]',el[j])
        end
        v = real( scalar(V) )
        S += -v * log(v)
    end
    S
end

function offdiagITensor(n::Integer, leftind::Index{Int}, rightind::Index{Int})
    # Dense
    @assert dim(leftind) == dim(rightind)
    m = dim(leftind)
    M = diagm(n=>fill(1., m-n))
    ITensor(M, leftind, rightind)
end

function offdiagITensor(n::Integer, leftind::Index{<:Vector}, rightind::Index{<:Vector})
    # Sparse
    @assert dim(leftind) == dim(rightind)
    m = dim(leftind)
    M = ITensor(leftind, dag(rightind))
    for i=1:m-n
        M[i,i+n] = 1.
    end
    M
end

"""
`
binom_rootn(n,k)
`

Calculate `binomial(n,k)^(1/n)` without an integer overflow.
"""
binom_rootn(n,k) = prod( (nn/dd)^(1/n) for (nn,dd) in zip(n-k+1:n, 1:k) )

function inftemp_state_with_filling(μ::Vector{<:Index}, ν::Rational)
    @assert sitetype(μ) === SiteType("FermionOperator")
    @assert isinteger(ν * length(μ))

    N = length(μ)
    n = numerator(ν * N)
    if μ isa Vector{<:Index{<:Vector}} # if there are conserved quantum numbers
        qns = [QN("Nf", i, -1) => 1 for i in 0:n]
        l = [Index(qns; tags="Link,l=$i") for i=0:N]
    else
        l = [Index(n+1; tags="Link,l=$i") for i=0:N]
    end
    As = map(1:N) do i
        emp = offdiagITensor(0, l[i], l[i+1]) * state(μ[i], "Emp")
        occ = offdiagITensor(1, l[i], l[i+1]) * state(μ[i], "Occ")
        (emp + occ) / binom_rootn(N,n)
    end
    As[1]   *= onehot(dag(l[1])=>1)
    As[end] *= onehot(l[end]=>n+1)
    truncate(MPS(As))
end



include("timedoubling.jl")
include("noise.jl")
include("models.jl")

export
    entanglement_entropy,
    vonneumann,
    current_operator,
    inftemp_state_with_filling
