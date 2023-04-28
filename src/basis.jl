abstract type Basis end

struct PauliBasis <: Basis end
struct LadderBasis <: Basis end

ITensors.space(::SiteType"SuperFermion") = [QN("Nf",-1,-1)=>1,
                                            QN("Nf", 0,-1)=>2,
                                            QN("Nf", 1,-1)=>1]

_id_like(T::ITensor)::ITensor =  *( (op("Id", dag(i)) for i in inds(T, plev=0))...)

"""
`
superoperator(A::ITensor, B::ITensor)
`

Turn two ITensors A and B into a superoperator A⊗B
"""
function superoperator(A::ITensor, B::ITensor, μ::Vector{<:Index})::ITensor
    @assert ndims(A) == ndims(B)
    s  = inds(A; plev=0)
    A′ = replaceprime(A, 0=>2, 1=>3)
    AxB = A′*B
    for x in s
        μi = μ[sitepos(x)]
        innerC = combiner(x'', dag(x'); dir= dir(μi))
        outerC = combiner(dag(x'''), x; dir=-dir(μi))
        replaceind!(outerC, combinedind(outerC), μi)
        replaceind!(innerC, combinedind(innerC), μi')
        AxB = innerC * AxB * outerC
    end
    AxB
end
superoperator(T::ITensor, M::UniformScaling, μ::Vector{<:Index}) = (M.λ) * superoperator(T, _id_like(T), μ)
superoperator(M::UniformScaling, T::ITensor, μ::Vector{<:Index}) = (M.λ) * superoperator(_id_like(T), T, μ)

function supertag(x::Index)::String
    t = collect(tags(x))
    st = TEBD.sitetype(x)

    replace!(t, "$(st)"=>"Super$(string(st))")
    join(t,",")
end


function gate2state(T::ITensor, μ::Vector{<:Index})::ITensor
    s = inds(T; plev=0, tags="Site")
    for x in s
        y = μ[sitepos(x)]
        c = combiner(x, dag(x'); dir=dir(y))
        replaceind!(c, combinedind(c), y)
        T *= c
    end
    T
end

function state2gate(T::ITensor, s::Vector{<:Index})::ITensor
    μ = inds(T; tags="Site")
    for y in μ
        x = s[sitepos(y)]
        c = combiner(x, dag(x'); dir=-dir(y))
        replaceind!(c, combinedind(c), y)
        T *= c
    end
    T
end

"""
`
mps2mpo(ρ::MPS)
`

Combine the in and out indices of an DMPO to make an MPS
"""
mps2mpo(ψ::MPS, s::Vector{<:Index})::MPO = MPO([state2gate(d, s) for d in ψ.data], ψ.llim, ψ.rlim)

"""
`
mpo2mps(ρ::MPO)
`

Combine the in and out indices of an DMPO to make an MPS
"""
mpo2mps(ρ::MPO, μ::Vector{<:Index})::MPS = MPS([gate2state(d, μ) for d in ρ.data], ρ.llim, ρ.rlim)


function _changeOfBasisTensors(::LadderBasis)
    x = siteind("Fermion", conserve_nf=true)
    [1/sqrt(2) * op("Id", x),
     1/sqrt(2) * op("Z",  x),
     1.        * op("c",  x),
     1.        * op("c†", x)]
end

function _changeOfBasisTensors(::PauliBasis)
    x = siteind("S=1/2")
    [1/sqrt(2) * op("Id", x),
     1/sqrt(2) * op("X",  x),
     1/sqrt(2) * op("Y",  x),
     1/sqrt(2) * op("Z",  x)]
end

changeOfBasis(basis::Basis, x::Index) = +(
    map(enumerate(_changeOfBasisTensors(basis))) do (i,T)
        y = only(inds(T, plev=0))
        c = combiner(y', dag(y); dir=-dir(x))
        replaceind!(c, combinedind(c), x)
        dag(T) * c * onehot(x'=>i)
    end...)
