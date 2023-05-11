Base.zero(::Type{ITensor}) = ITensor(0.)

function infTempState(s::Vector{<:Index})::MPO
    ρ₀ = MPO(s, ["Id" for x in s])
    for i=1:length(s)
        ρ₀[i] ./= dim(s[1])
    end
    ρ₀
end

spinval(ρ::MPS, opname::String) = spinval(convertToSpin(ρ), opname)
function spinval(ρ::MPO, opname::String)::Vector{Float64}
    Os = [op(opname, s) for s=extractsites(ρ)]
    localop(ρ, Os)
end

energy_density(ρ::ITensors.AbstractMPS, ham::Vector{ITensor}) = localop(ρ, ham)

currentOperator(ham::Vector{ITensor}) = currentOperator(ham, ham)

"""
`
currentOperator(ham::Vector{ITensor}, O::Vector{ITensor})::Vector{ITensor}
`

Calculate the current operator of a charge `O` under time evolution of `ham`

`
currentOperator(ham::Vector{ITensor})::Vector{ITensor}
`

Calculate the energy current
"""
function currentOperator(ham::Vector{ITensor}, O::Vector{ITensor})::Vector{ITensor}
    js = [ITensor(0.)] # set first current to 0
    for On in O[1:end-1]
        relevantTerms = filter(h->!isempty(commoninds(h, On)), ham)
        ∂thTerms = map(t -> 1im * (product(t, On) - product(On, t)), relevantTerms)
        ∂th = sum(addIdentities(∂thTerms))
        ∂th, prevj = addIdentities([∂th, js[end]])
        j = prevj - ∂th
        push!(js, removeIdentities(j))
    end
    push!(js, ITensor(0.));
    js
end

currentOperator(ham::Vector{ITensor}, O::ITensor) = currentOperator(ham, [O])


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
    s = extractsites(ρ)
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



include("timedoubling.jl")
include("noise.jl")
include("models.jl")

export
    entanglement_entropy,
    vonneumann,
    energy_density,
    infTempState,
    spinval,
    currentOperator
