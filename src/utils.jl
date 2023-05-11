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
tracer(::SiteType"FermionOperator", x::Index) = dim(x)^(1/4) * state(dag(x), 1)
tracer(::SiteType"PauliOperator", x::Index) = dim(x)^(1/4) * state(dag(x), 1)
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
    s = siteinds(first, ψ; plev=0)
    st = sitetype(ψ)
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
function addIdentities(ts)
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
        C = combiner(others..., dag.(others')...)
        U = C * T
        A = Array(U, s[i], dag(s[i])', combinedind(C))
        if all(mapslices(∝(I), A; dims=[1,2]))
            T *= onehot(s[i]'=>1) * onehot(dag(s[i])=>1)
        end
    end
    T
end

function localop(ρ::MPS, Os::Vector{ITensor})::Vector{Float64}
    map(Os) do O
        # little hack to include 0-dim tensors
        if isempty(inds(O))
            return O[1]
        end
        real( trace( product(O, ρ) )  )
    end
end

export
    normalizedm!,
    convertToPauli,
    convertToSpin,
    sitepos,
    trace,
    dagger,
    spin,
    mpo2superoperator,
    mps2superoperator,
    localop
