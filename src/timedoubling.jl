"""
`
Θ[!](ρ::MPS)
`

Take the time-inverse, [in place]
"""
Θ!(ρ::MPS) = Θ!(sitetype(ρ), ρ)

function Θ(ρ::MPS)
    ρ′ = deepcopy(ρ)
    Θ!(ρ′)
    ρ′
end

function Θ!(::SiteType"FermionOperator", ρ::MPS)
    ρ[:] = conj.(ρ[:])
end

"""
In the Pauli basis, flip the σ₂ component (since it is imaginary). Then take the complex conjugate
"""
function Θ!(::SiteType"Pauli", ρ::MPS)
    A = diagm([1,1,-1,1])
    μ = siteinds(ρ)
    for i in eachindex(ρ)
        U = ITensor(A, μ[i], μ[i]')
        ρ[i] *= U
        ITensors.replaceprime!(ρ[i], 1, 0)
        ρ[i] .= conj(ρ[i])
    end
end

"""Take the inner product, but offset second MPSs but a certain number of sites"""
function offset_dot(M₁::MPS, M₂::MPS, Δx::Int)
    Δx<0 && return conj(offset_dot(M₂, M₁, -Δx))

    s = siteinds(M₁)
    M₁dag = mapprime(dag(M₁), 0, 1; tags="Link")
    O = one(ITensor)
    for j in 1:Δx
        O *= M₁dag[j] * tracer(dag(s[j]))
    end

    for j in eachindex(M₁)[Δx+1:end]
        shiftedM₂T = replaceind(M₂[j-Δx], s[j-Δx], s[j])
        O = (O * M₁dag[j]) * shiftedM₂T
    end

    for j in eachindex(M₂)[end-Δx+1:end]
        O *= M₂[j] * tracer(s[j])
    end

    only(O)
end
"""
`
Θexpect(ρ1::MPS, ρ2::MPS)
`

Take the inner product of ρ₁ and Θ(ρ₂) at different offsets.

"""

function Θexpect(ρ₁::MPS, ρ₂::MPS; realval=true)
    @assert length(ρ₁) == length(ρ₂)

    N = length(ρ₁)
    ρ₂dag = Θ(ρ₂)
    map((-N+1):(N-1)) do Δx
        res = offset_dot(ρ₁, ρ₂dag, Δx)
        realval ? real(res) : res
    end
end

export Θexpect
