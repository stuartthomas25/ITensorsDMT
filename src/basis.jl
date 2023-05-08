"""
`
superoperator(A::ITensor, B::ITensor)
`

Turn two ITensors `A` and `B` into a superoperator A⊗B using the basis of `μ`
"""
function superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})::ITensor
    @assert ndims(A) == ndims(B)
    s  = inds(A; plev=0)
    A′ = replaceprime(A, 0=>2)
    B′ = replaceprime(B, 1=>3)
    AxB = A′*B′
    for x=s
        y = μ[sitepos(x)]
        innerU = swapprime( changeOfBasis(x'', y), 3=>2)
        outerU = changeOfBasis(x, y')
        AxB = (AxB * dag(innerU)) * outerU
    end
    AxB
end


# 'A'' '''B

# function old_superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})::ITensor
#     @assert ndims(A) == ndims(B)
#     s  = inds(A; plev=0)
#     B′ = replaceprime(B, 0=>2, 1=>3)
#     AxB = A*B′
#     for x in s
#         μi = μ[sitepos(x)]
#         innerC = combiner(x'', dag(x'); dir= dir(μi))
#         outerC = combiner(dag(x'''), x; dir=-dir(μi))
#         replaceind!(outerC, combinedind(outerC), μi)
#         replaceind!(innerC, combinedind(innerC), μi')
#         AxB = innerC * AxB * outerC
#     end
#     AxB
# end


# superoperator(::Type{StandardBasis}, A, B, μ) = superoperator(A, B, μ)

# function superoperator(::Type{basis}, A, B, μ) where {basis<:Basis}
#     AxB = superoperator(A, B, μ)
#     xs = inds(AxB; plev=0)
#     for x in xs
#         U = changeOfBasis(basis, dag(x))
#         Udag = swapprime(dag(U), 0=>1)
#         AxB = reduce(product, [U, AxB, Udag])
#     end
#     ITensors.dropzeros(AxB)
# end

superoperator(T::ITensor, M::UniformScaling, μ::Vector{<:Index}) =
    M.λ * superoperator(T, op("Id", collect(inds(dag(T), plev=0))), μ)

superoperator(M::UniformScaling, T::ITensor, μ::Vector{<:Index}) =
    M.λ * superoperator(op("Id", collect(inds(dag(T), plev=0))), T, μ)

superoperator(M::UniformScaling, N::UniformScaling, μ::Vector{<:Index}) =
    M.λ * N.λ * op("Id", dag(μ))

superoperator(A, B, x::Index) = superoperator(A,B,[x])

# function supertag(x::Index)::String
#     t = collect(tags(x))
#     st = TEBD.sitetype(x)

#     replace!(t, "$(st)"=>"Super$(string(st))")
#     join(t,",")
# end

# function gate2state(T::ITensor)::ITensor
#     s = inds(T; plev=0, tags="Site")
#     for x in s
#         T *= combiner(x, dag(x'); dir=ITensors.Out)
#     end
#     ITensors.dropzeros(T)
# end

# function gate2state(T::ITensor, μ::Vector{<:Index})::ITensor
#     s = inds(T; plev=0, tags="Site")
#     for x in s
#         y = μ[sitepos(x)]
#         c = combiner(x, dag(x'); dir=-dir(y))
#         replaceind!(c, combinedind(c), y)
#         T *= c
#     end
#     ITensors.dropzeros(T)
# end

# function state2gate(T::ITensor, s::Vector{<:Index})::ITensor
#     μ = inds(T; tags="Site")
#     for y in μ
#         x = s[sitepos(y)]
#         c = combiner(x, dag(x'); dir=-dir(y))
#         replaceind!(c, combinedind(c), y)
#         T *= c
#     end
#     ITensors.dropzeros(T)
# end

"""
`
mps2mpo(ρ::MPS)
`

Combine the in and out indices of an DMPO to make an MPS
"""
# ITensors.MPO(ψ::MPS, s::Vector{<:Index})::MPO = MPO(StandardBasis, ρ, s)

# ITensors.MPO(::Type{StandardBasis}, ψ::MPS, s::Vector{<:Index})::MPO =
#     MPO([state2gate(d, s) for d in ψ.data], ψ.llim, ψ.rlim)

# function ITensors.MPO(::Type{B}, ψ::MPS, s::Vector{<:Index})::MPO where {B<:Basis}
#     μ = siteinds(only, ψ)
#     newdata = map(eachindex(ψ)) do i
#         Udag = swapprime(dag(changeOfBasis(B, μ[i])), 0=>1)
#         T = product( Udag, ψ.data[i])
#         state2gate(T, s)
#     end
#     MPO(newdata,ψ.llim, ψ.rlim)
# end

function ITensors.MPO(ψ::MPS, s::Vector{<:Index})::MPO
    μ = siteinds(only, ψ)
    newdata = map(eachindex(ψ)) do i
        # c = combiner(s[i], dag(s[i]'); dir=ITensors.In)
        # cx = combinedind(c)
        U = changeOfBasis(s[i], μ[i])
        # @show inds(dag(U)) inds(ψ.data[i])
        dag(U) * ψ.data[i]
    end
    MPO(newdata, ψ.llim, ψ.rlim)
end

# ITensors.MPs(b::Basis, ψ::MPS, s::Vector{<:Index})::MPO =
#     MPO([product( changeOfBasis(b, s), state2gate(d, s)) for d in ψ.data], ψ.llim, ψ.rlim)

"""
`
mpo2mps(ρ::MPO)
`

Combine the in and out indices of an DMPO to make an MPS
"""
# ITensors.MPS(ρ::MPO, μ::Vector{<:Index})::MPS =
#     MPS(StandardBasis, ρ, μ)

# ITensors.MPS(::Type{StandardBasis}, ρ::MPO, μ::Vector{<:Index})::MPS =
#     MPS([gate2state(d, μ) for d in ρ.data], ρ.llim, ρ.rlim)

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
     1/√2 * op("Z",  x),
     1.   * op("c",  x),
     1.   * op("c†", x)
     ] # we use this order so that the (-1,0,0,1) block structure is preserved

ITensors.space(::SiteType"FermionOperator"; conserve_nf=false) = conserve_nf ? [
                                            QN("Nf", 0,-1)=>2,
                                            QN("Nf",-1,-1)=>1,
                                            QN("Nf", 1,-1)=>1
                                           ] : 4

ITensors.state(::StateName"Id", ::SiteType"FermionOperator") = [1.0, 0, 0, 0]

ITensors.state(::StateName"InfTemp", ::SiteType"FermionOperator") = [1/√2, 0, 0, 0]

function trace(::SiteType"FermionOperator", ρ::MPS)
    μ = siteinds(first, ρ; plev=0)
    prod = 1.
    for (x,T) in zip(μ,ρ)
        prod *= dim(x)^(1/4) * T * state(dag(x),1)
    end
    prod[1]
end

# Pauli Operator Site Type
changeOfBasisTensors(::SiteType"PauliOperator", x::Index) =
    [1/√2 * op("Id", x),
     1/√2 * op("X",  x),
     1/√2 * op("Y",  x),
     1/√2 * op("Z",  x)]

ITensors.space(::SiteType"PauliOperator") = 4

ITensors.state(::StateName"Id", ::SiteType"PauliOperator") = [1.0, 0, 0, 0]

function trace(::SiteType"PauliOperator", ρ::MPS)
    μ = siteinds(first, ρ; plev=0)
    prod = 1.
    for (x,T) in zip(μ,ρ)
        prod *= dim(x)^(1/4) * T * state(dag(x),1)
    end
    prod[1]
end



# changeOfBasis(::Type{B}, x::Index) where {B<:Basis} = +(
#     map(enumerate(changeOfBasisTensors(B))) do (i,T)
#         y = only(inds(T, plev=0))
#         c = combiner(y', dag(y); dir=-dir(x))
#         replaceind!(c, combinedind(c), x)
#         dag(T) * c * onehot(x'=>i)
#     end...)


changeOfBasis(x::Index, y::Index) =
    +( map(changeOfBasisTensors(sitetype(y), x) |> enumerate) do (i,T)
        T * onehot(y=>i)
    end... )


export
    superoperator,
    Basis,
    PauliBasis,
    LadderBasis
