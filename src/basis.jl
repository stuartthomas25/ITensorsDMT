"""
`
superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})
`

Turn two ITensors `A` and `B` into a superoperator A⊗B using the basis of `μ`
⟨σᵃ, A σᵇ B⟩ = Tr[σᵃ† A σᵇ B] = σᵃ*₃₀A₃₂σᵇ₂₁B₁₀
"""
function superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})::ITensor
    @assert ndims(A) == ndims(B)
    s = dag(inds(B; plev=0))
    A′ = replaceprime(A, 0=>2, 1=>3)
    AxB = A′*B
    for x=s
        y = μ[sitepos(x)]
        innerU = changeOfBasis(x', dag(y))
        outerU = swapprime(changeOfBasis(x'', dag(y')), 2=>0)
        AxB = (AxB * innerU) * dag(outerU)
    end
    AxB
end


superoperator(T::ITensor, μ::Vector{<:Index}) = superoperator(T, I, μ)
superoperator(Ts::Vector{ITensor}, μ::Vector{<:Index}) = [superoperator(T, μ) for T in Ts]


function superoperator(T::ITensor, M::UniformScaling, μ::Vector{<:Index})
    if isempty(inds(T))
        return T
    end
    M.λ * superoperator(T, op("Id", collect(inds(dag(T), plev=0))), μ)
end

function superoperator(M::UniformScaling, T::ITensor, μ::Vector{<:Index})
    if isempty(inds(T))
        return T
    end
    M.λ * superoperator(op("Id", collect(inds(dag(T), plev=0))), T, μ)
end

superoperator(M::UniformScaling, N::UniformScaling, μ::Vector{<:Index}) =
    M.λ * N.λ * op("Id", dag(μ))

changeOfBasis(x::Index, y::Index) =
    +( map(changeOfBasisTensors(sitetype(y), x) |> enumerate) do (i,T)
        T * onehot(y=>i)
    end... )

"""
`
MPO(ρ::MPS, s)
`

Separate the indices of an MPS to make an MPO
"""
function ITensors.MPO(ψ::MPS, s::Vector{<:Index})::MPO
    μ = siteinds(only, ψ)
    newdata = map(eachindex(ψ)) do i
        U = changeOfBasis(s[i], μ[i])
        dag(U) * ψ.data[i]
    end
    MPO(newdata, ψ.llim, ψ.rlim)
end

"""
`
MPS(ρ::MPO, s)
`

Combine the in and out indices of an DMPO to make an MPS
"""
function ITensors.MPS(ρ::MPO, μ::Vector{<:Index})::MPS
    s = siteinds(first, ρ; plev=0)
    newdata = map(eachindex(ρ)) do i
        # c = combiner(s[i], dag(s[i]'); dir=ITensors.Out)
        # cx = combinedind(c)
        U = changeOfBasis(s[i], μ[i])
        ρ.data[i] * U
    end
    MPS(newdata, ρ.llim, ρ.rlim)
end

# Fermion Operator Site Type
changeOfBasisTensors(::SiteType"FermionOperator", x::Index) =
    [
     1/√2 * op("Id", x),
     1/√2 * op("F",  x),
     1.   * op("c",  x),
     1.   * op("c†", x)
     ] # we use this order so that the (-1,0,0,1) block structure is preserved

ITensors.space(::SiteType"FermionOperator"; conserve_nf=false) = conserve_nf ? [
                                            QN("Nf", 0,-1)=>2,
                                            QN("Nf",-1,-1)=>1,
                                            QN("Nf", 1,-1)=>1
                                           ] : 4

ITensors.state(::StateName"Id", ::SiteType"FermionOperator") = [1.0, 0, 0, 0]

ITensors.state(::StateName"InfTemp", ::SiteType"FermionOperator") = [1., 0, 0, 0]
ITensors.state(::StateName"Emp", ::SiteType"FermionOperator") = [1/√2, 1/√2, 0, 0]
ITensors.state(::StateName"Occ", ::SiteType"FermionOperator") = [1/√2, -1/√2, 0, 0]

# Pauli Operator Site Type
changeOfBasisTensors(::SiteType"PauliOperator", x::Index) =
    [1/√2 * op("Id", x),
     1/√2 * op("X",  x),
     1/√2 * op("Y",  x),
     1/√2 * op("Z",  x)]

ITensors.space(::SiteType"PauliOperator") = 4

ITensors.state(::StateName"Id", ::SiteType"PauliOperator") = [1.0, 0, 0, 0]




export
    superoperator,
    Basis,
    PauliBasis,
    LadderBasis
