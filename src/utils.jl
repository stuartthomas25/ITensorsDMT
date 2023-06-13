sitepos(s::Index)::Int64 = parse(Int64, match(r"[ln]=(\d+)", join(tags(s)))[1])

sitetype(ρ::ITensors.AbstractMPS) = sitetype(siteinds(first, ρ))

_sitetypefilter(t::String) = t!="Site" && t[1:2]!="n="
sitetype(s::Vector{<:Index}) = SiteType(
    only(filter(_sitetypefilter∘string,
                collect(ITensors.commontags(s)))))

sitetype(s::Index) = sitetype([s])
Base.string(::SiteType{T}) where {T} = T

clean(T::ITensor) = T.tensor isa NDTensors.DenseTensor ? T : ITensors.dropzeros(T)

function logpurity(ρ::MPO)::Float64
    s = siteinds(first, ρ; plev=0)
    sum::Float64 = 0.
    prod::ITensor = ITensor(1. + 0im)
    for (x,T) in zip(s,ρ)
        prod *= T
        prod *= swapprime(T, 0=>1, "Link")
        f = norm(prod)
        sum += log(f)
        prod /= f
    end
    sum + log(real(prod[1]))
end


"""
`
tracer(::SiteType, x::Index)
`

a tensor that takes a trace over an index
"""
tracer(x::Index) = tracer(sitetype(x), x)
tracer(::SiteType"FermionOperator", x::Index) = state(dag(x), 1)
tracer(::SiteType"PauliOperator", x::Index) = state(dag(x), 1)
tracer(::SiteType"Fermion", x::Index) = delta(dag(x), x')
tracer(::SiteType"S=1/2",   x::Index) = delta(dag(x), x')
tracer(st::SiteType, ::Index) = throw("Unsupported site type \"$(string(st))\"")

function trace(ψ::ITensors.AbstractMPS)
    μ = siteinds(first, ψ; plev=0)
    prod = 1.
    for (x,T) in zip(μ,ψ)
        prod *= T * tracer(x)
    end
    prod[1]
end

function logtrace(ψ::ITensors.AbstractMPS)::Float64
    μ = siteinds(first, ψ; plev=0)
    sum = 0.
    prod = 1.
    for (x,T) in zip(s,ψ)
        prod *= T * tracer(x)
        f = norm(prod)
        sum += log(f)
        prod /= f
    end
    sum + log(real(prod[1]))
end


"""Take a vector of Pauli operators and add identities such that their support is the same"""
function addIdentities(ts...)
    length(ts)==0 && return ts
    is = unioninds(ts..., plev=0)
    map(ts) do t
        newix = uniqueinds(is, inds(t))
        if !isempty(newix)
            t * reduce(*, [dag(op("Id", i)) for i in newix])
        else
            t
        end
    end
end


∝(a, b) = b[1,1] * a == b * a[1,1]
∝(a) = b->∝(a,b)

"""remove any superfluous identities in a Pauli string"""
function removeIdentities(T)
    s = sort(inds(T; plev=0), by=sitepos)
    for i in reverse(eachindex(s)) # work from back so as not to mess up indexing
        others = vcat(s[1:i-1]..., s[i+1:end]...)
        common_others = commoninds(T, others)
        C = combiner(common_others..., dag.(common_others')...)
        U = C * T
        A = Array(U, s[i], dag(s[i])', combinedind(C))
        if all(mapslices(∝(I), A; dims=[1,2]))
            T *= onehot(s[i]'=>1) * onehot(dag(s[i])=>1)
        end
    end
    T
end

function probe(Os::Vector{ITensor}, μ::Vector{<:Index}; kwargs...)
    wrappers = [probe(O, μ; kwargs...) for O in Os]
    return ρ -> [w(ρ) for w in wrappers]
end

"""
`
probe(O::ITensor, μ::Vector{<:Index}; realval=true)
`
Create a function that measures an MPS using the observable `O`. `μ` are the site indices of the MPS.
"""
function probe(O::ITensor, μ::Vector{<:Index}; realval=true)
    if inds(O) |> isempty
        return ρ -> O[1]
    end

    SO = superoperator(O,I,μ)
    tSO = SO * foldl(*, (dag(tracer(x)') for x in inds(SO; plev=0)))

    function wrapper(ρ::MPS)
        μ = siteinds(first, ρ; plev=0)
        traced_ρ = ITensor(one(ComplexF64))
        for (x,T) in zip(μ,ρ)
            traced_ρ *= T
            if commoninds(tSO, T) |> isempty
                traced_ρ *= tracer(x)
            end
        end

        res = traced_ρ * tSO
        @assert isempty(inds(res))
        return realval ? real(only(res)) : only(res)
    end
end

"""
`
dagger(T::ITensor)
`
Take the full Hermitian conjugate of an ITensor.

This function takes the complex conjugate, flips arrows and flips prime levels between 1 and 0.
"""
function dagger(T::ITensor)
    return swapprime(dag(T),0,1,tags="Site")
end

function dagger(ρ::MPO)
    MPO([dagger(T) for T in ρ.data])
end

decomplexify(nt::NamedTuple) = map(pairs(nt) |> collect) do (k,v)
    if v isa Complex
        k_re = Symbol(string(k)*"_re")
        k_im = Symbol(string(k)*"_im")
        [k_re=>real(v), k_im=>imag(v)]
    else
        [k=>v]
    end
end |> Iterators.flatten |> collect |> NamedTuple

export
    trace,
    localop,
    logtrace,
    probe,
    dagger,
    decomplexify
