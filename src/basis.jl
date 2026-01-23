"""
`
superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})
`

Turn two ITensors `A` and `B` into a superoperator A⊗B using the basis of `μ`
⟨σᵃ, A σᵇ B⟩ = Tr[σᵃ† A σᵇ B] = σᵃ*₃₀A₃₂σᵇ₂₁B₁₀
"""
function superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})::ITensor
    @assert issetequal(inds(A; tags="Site"), inds(B; tags="Site"))
    s = dag(inds(B; plev=0, tags="Site"))
    A′ = prime(A, 2; tags="Site")
    AxB = A′*B
    for x=s
        y = μ[sitepos(x)]
        innerU = changeOfBasis(x', dag(y))
        outerU = swapprime(changeOfBasis(x'', dag(y')), 2=>0; tags="Site")
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
    M.λ * superoperator(T, op("Id", collect(inds(dag(T), plev=0, tags="Site"))), μ)
end

function superoperator(M::UniformScaling, T::ITensor, μ::Vector{<:Index})
    if isempty(inds(T))
        return T
    end
    M.λ * superoperator(op("Id", collect(inds(dag(T), plev=0, tags="Site"))), T, μ)
end

function superoperator(A::MPO, M::UniformScaling, μ::Vector{<:Index})
    if isempty(A)
        return A
    end

    # multiplying by the identity preserves the orthogonality center
    MPO([superoperator(T,M,μ) for T in A], A.llim, A.rlim)
end
function superoperator(M::UniformScaling, A::MPO, μ::Vector{<:Index})
    if isempty(A)
        return A
    end

    # multiplying by the identity preserves the orthogonality center
    MPO([superoperator(M,T,μ) for T in A], A.llim, A.rlim)
end

function superoperator(A::MPO, B::MPO, μ::Vector{<:Index})
    if isempty(A)
        return A
    end
    @assert siteinds(A) == siteinds(B)

    # reset orthogonality center
    MPO([superoperator(Ta,Tb,μ) for (Ta,Tb) in zip(A,B)], 0, length(A))
end

superoperator(M::UniformScaling, N::UniformScaling, μ::Vector{<:Index}) =
    M.λ * N.λ * op("Id", dag(μ))

superoperator(::Type{MPO}, M::UniformScaling, N::UniformScaling, μ::Vector{<:Index}) =
    M.λ * N.λ * MPO(μ, "Id")

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
function ITensorMPS.MPO(ψ::MPS, s::Vector{<:Index})::MPO
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
function ITensorMPS.MPS(ρ::MPO, μ::Vector{<:Index})::MPS
    s = siteinds(first, ρ; plev=0)
    newdata = map(eachindex(ρ)) do i
        U = changeOfBasis(s[i], μ[i])
        ρ.data[i] * U
    end
    MPS(newdata, ρ.llim, ρ.rlim)
end

# FermionOperator Site Type
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

ITensors.state(::StateName"Id",      ::SiteType"FermionOperator") = [1,  0,  0,  0]
ITensors.state(::StateName"InfTemp", ::SiteType"FermionOperator") = [1,  0,  0,  0]
ITensors.state(::StateName"Emp",     ::SiteType"FermionOperator") = [1,  1,  0,  0]
ITensors.state(::StateName"Occ",     ::SiteType"FermionOperator") = [1, -1,  0,  0]


tracer(::SiteType"Fermion", x::Index) = delta(dag(x), x')
tracer(::SiteType"FermionOperator", x::Index) = state(dag(x), 1)


# ElectronOperator Site Type
changeOfBasisTensors(::SiteType"ElectronOperator", x::Index) = let pl=plev(x), x_=noprime(x)
    prime.([
     1/2  * op("Id",        x_),
     1/2  * op("F",         x_),
     1/2  * op("F↑",        x_),
     1/2  * op("F↓",        x_),
     1/√2 * op("a↑",        x_),
     1/√2 * op("a↑ * F",    x_),
     1/√2 * op("a↓",        x_),
     1/√2 * op("a↓ * F",    x_),
     1/√2 * op("a†↑",       x_),
     1/√2 * op("a†↑ * F",   x_),
     1/√2 * op("a†↓",       x_),
     1/√2 * op("a†↓ * F",   x_),
            op("a↓ * a↑",   x_),
            op("a†↓ * a†↑", x_),
            op("a†↓ * a↑",  x_),
            op("a↓ * a†↑",  x_)
     ], pl)
end

ITensors.space(::SiteType"ElectronOperator"; conserve_qns=false) = conserve_qns ? [
                                            QN(("Nf", 0,-1),("Sz", 0))=>4,

                                            QN(("Nf",-1,-1),("Sz",-1))=>2,
                                            QN(("Nf",-1,-1),("Sz", 1))=>2,
                                            QN(("Nf", 1,-1),("Sz", 1))=>2,
                                            QN(("Nf", 1,-1),("Sz",-1))=>2,

                                            QN(("Nf",-2,-1),("Sz", 0))=>1,
                                            QN(("Nf", 2,-1),("Sz", 0))=>1,
                                            QN(("Nf", 0,-1),("Sz",-2))=>1,
                                            QN(("Nf", 0,-1),("Sz", 2))=>1
                                           ] : 16

ITensors.state(::StateName"Id",      ::SiteType"ElectronOperator") = [j==1 ? 1 : 0 for j=1:16]
ITensors.state(::StateName"InfTemp", ::SiteType"ElectronOperator") = [j==1 ? 1 : 0 for j=1:16]

ITensors.state(::StateName"Emp", ::SiteType"ElectronOperator")  = [0.5, 0.5, 0.5, 0.5, zeros(12)...]
ITensors.state(::StateName"Up", ::SiteType"ElectronOperator")   = [0.5,-0.5,-0.5, 0.5, zeros(12)...]
ITensors.state(::StateName"Dn", ::SiteType"ElectronOperator")   = [0.5,-0.5, 0.5,-0.5, zeros(12)...]
ITensors.state(::StateName"UpDn", ::SiteType"ElectronOperator") = [0.5, 0.5,-0.5,-0.5, zeros(12)...]
ITensors.state(::StateName"0", st::SiteType"ElectronOperator")  = state(StateName("Emp"), st)
ITensors.state(::StateName"↑", st::SiteType"ElectronOperator")  = state(StateName("Up"), st)
ITensors.state(::StateName"↓", st::SiteType"ElectronOperator")  = state(StateName("Dn"), st)
ITensors.state(::StateName"↑↓", st::SiteType"ElectronOperator") = state(StateName("UpDn"), st)

tracer(::SiteType"Electron", x::Index) = delta(dag(x), x')
tracer(::SiteType"ElectronOperator", x::Index) = state(dag(x), 1)

# Pauli Operator Site Type
changeOfBasisTensors(::SiteType"PauliOperator", x::Index) =
    [1/√2 * op("Id", x),
     1/√2 * op("X",  x),
     1/√2 * op("Y",  x),
     1/√2 * op("Z",  x)]

ITensors.space(::SiteType"PauliOperator") = 4

ITensors.state(::StateName"Id", ::SiteType"PauliOperator")      = [1.0, 0, 0, 0]
ITensors.state(::StateName"X", ::SiteType"PauliOperator")       =  [0, 1.0, 0, 0]
ITensors.state(::StateName"Y", ::SiteType"PauliOperator")       =  [0, 0, 1.0, 0]
ITensors.state(::StateName"Z", ::SiteType"PauliOperator")       =  [0, 0, 0, 1.0]
ITensors.state(::StateName"↑", st::SiteType"PauliOperator")     = state(StateName("Up"), st)
ITensors.state(::StateName"↓", st::SiteType"PauliOperator")     = state(StateName("Dn"), st)
ITensors.state(::StateName"Up", ::SiteType"PauliOperator")      = [ 0.5, 0.5, 0., 0.]
ITensors.state(::StateName"Dn", ::SiteType"PauliOperator")      = [ 0.5,-0.5, 0., 0.]
ITensors.state(::StateName"InfTemp", st::SiteType"PauliOperator") = state(StateName("Id"), st)

tracer(::SiteType"S=1/2",   x::Index) = delta(dag(x), x')
tracer(::SiteType"PauliOperator", x::Index) = state(dag(x), 1)




# QUDIT
function changeOfBasisTensors(::SiteType"QuditOperator", x::Index)
    function basis(n) # create an orthogonal basis starting with 1/√n [1,1,1,..]
        ℬ = Vector[fill(1, n) / √(n)]
        for i in 1:n-1
            push!(ℬ, [ fill(1, i) ; [-i] ; fill(0, n - i - 1) ] / √(i+i^2))
        end
        ℬ
    end

    superbasis = vcat(map(sort(-dim(x)+1:dim(x)-1, by=abs)) do n
        [diagm(n=>v) for v∈basis(dim(x)-abs(n))]
    end...)

    [op(V, x) for V∈superbasis]
end

function ITensors.space(::SiteType"QuditOperator"; conserve_qns=false, dim=3, qnname_number)
    if conserve_qns
        map(sort(-dim+1:dim-1, by=abs)) do i
            QN(qnname_number, i) => dim - abs(i)
        end
    else
        dims^2
    end
end

ITensors.state(::StateName"InfTemp", ::SiteType"QuditOperator", x::Index) = [i==1 ? 1 : 0 for i in 1:dim(x)]


tracer(::SiteType"QuditOperator", x::Index) = state(dag(x), 1)

export
    superoperator,
    Basis,
    PauliBasis,
    LadderBasis
