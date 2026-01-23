sitepos(s::Index)::Int64 = parse(Int64, match(r"[ln]=(\d+)", join(tags(s)))[1])

sitetype(ρ::ITensorMPS.AbstractMPS) = sitetype(siteinds(first, ρ))

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
tracer(st::SiteType, ::Index) = throw("Unsupported site type \"$(string(st))\"")

function trace(ψ::ITensorMPS.AbstractMPS)
    μ = siteinds(first, ψ; plev=0)
    prod = 1.
    for (x,T) in zip(μ,ψ)
        prod *= T * tracer(x)
    end
    prod[1]
end

function logtrace(ψ::ITensorMPS.AbstractMPS)::Float64
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
    s = sort([inds(T; plev=0)...], by=sitepos)
    for i in reverse(eachindex(s)) # work from back so as not to mess up indexing
        others = vcat(s[1:i-1]..., s[i+1:end]...)
        common_others = commoninds(T, others)
        C = combiner(common_others..., dag.(common_others')...)
        isnothing(combinedind(C)) && continue
        U = C * T
        A = Array(U, s[i], dag(s[i])', combinedind(C))
        if all(mapslices(∝(I), A; dims=[1,2]))
            T *= onehot(s[i]'=>1) * onehot(dag(s[i])=>1)
        end
    end
    T
end

# function probe(Os::Vector{ITensor}, μ::Vector{<:Index}; kwargs...)
#     wrappers = [probe(O, μ; kwargs...) for O in Os]
#     return ρ -> [(w(ρ) for w in wrappers)...]
# end


""" Efficiently compute expectation values for a density matrix MPS """
function dm_expect(ρ::MPS, As::Vector{ITensor})
    s = siteinds(ρ)
    N = length(ρ)

    support(O::ITensor)::Vector{Int} = indexin(commoninds(s, O), s)
    leftmost_point  = first∘support
    rightmost_point = last∘support

    # initialize right tail cache
    cache = Vector{ITensor}(undef, N-1)
    t = one(ITensor)
    for j in reverse(2:N)
        t *= ρ[j] * tracer(s[j])
        cache[j-1] = t
    end

    C = Vector{ComplexF64}(undef, length(As))
    iAs = sort(collect(enumerate(As)), by=first∘support∘last)

    T = one(ITensor)
    for (j,A) in iAs
        A = foldl(*, tracer(x) for x∈commoninds(A,s'); init=A)
        # remove sites we never need again
        for old_site in commoninds( s[1:leftmost_point(A)-1], T)
            T *= tracer(old_site)
        end

        T_rightpoint = something(indexin(inds(T), linkinds(ρ))..., 0)
        for y in T_rightpoint+1:rightmost_point(A)
            T *= ρ[y]
        end
        T2 = product(A, T)

        # trace over commoninds(T3, s)
        T2 = foldl(*, tracer(x) for x∈commoninds(T2,s); init=T2)

        # multiply cache (identified from link ind)
        if !isempty(inds(T2))
            k = only(indexin(inds(T2), linkinds(ρ)))
            T2 *= cache[k]
        end

        C[j] = only(T2)
    end

    C
end


""" Compute the correlator for a density matrix MPS """
function dm_correlator(ρ::MPS, As::Vector{ITensor}, Bs::Vector{ITensor})
    s = siteinds(only, ρ)
    N = length(ρ)
    support(O::ITensor)::Vector{Int} = indexin(commoninds(s, O), s)
    leftmost_point  = first∘support
    rightmost_point = last∘support

    # initialize right tail cache
    cache = Vector{ITensor}(undef, N-1)
    t = one(ITensor)
    for j in reverse(2:N)
        t *= ρ[j] * tracer(s[j])
        cache[j-1] = t
    end


    C = Matrix{ComplexF64}(undef, length(As), length(Bs))
    iAs = As |> enumerate |> collect
    iBs = Bs |> enumerate |> collect
    iAs = sort(iAs, by=leftmost_point∘last)
    iBs = sort(iBs, by=leftmost_point∘last)

    T = one(ITensor)

    function _correlator_upper_tri!(C, iAs, iBs, T; diag=true)
        s = siteinds(only, ρ)
        N = length(ρ)
        past_point = diag ? Base.:>= : Base.:>
        x = something(indexin(inds(T), linkinds(ρ))..., 0) + 1

        for (i,A) in filter(==(x)∘leftmost_point∘last, iAs)
            T2 = T
            for y in x:last(support(A))
                T2 *= ρ[y]
            end
            T2 = product(A, T2)

            for (j,B) in filter(past_point(x)∘leftmost_point∘last, iBs)
                B′ = foldl(*, tracer(x) for x∈commoninds(B,s'); init=B)
                for old_site in commoninds( s[1:leftmost_point(B)-1], T2) # we never need these sites again
                    T2 *= tracer(old_site)
                end


                T2_rightpoint = something(indexin(inds(T2), linkinds(ρ))..., N)
                for y in T2_rightpoint+1:rightmost_point(B)
                    # if y∉support(A) && y∉support(B)
                    #     T2 *= (ρ[y] * tracer(s[y]))
                    # else
                        T2 *= ρ[y]
                    # end
                end
                # B′ = traced over s[Bns]'
                T3 = product(B′, T2)

                # trace over commoninds(T3, s)
                T3 = foldl(*, tracer(x) for x∈commoninds(T3,s); init=T3)

                # multiply cache (identified from link ind)
                if !isempty(inds(T3))
                    k = only(indexin(inds(T3), linkinds(ρ)))
                    T3 *= cache[k]
                end

                C[i,j] = only(T3)
            end
        end
    end

    for x in 1:N
        _correlator_upper_tri!(C, iAs, iBs, T; diag=true)
        _correlator_upper_tri!(Transpose(C), iBs, iAs, T; diag=false)

        x==N && continue

        T *= ρ[x] * tracer(s[x])
    end
    C
end

_handle_noind_tensor(T::ITensor, μ::Vector{<:Index}) = inds(T) |> isempty ? only(T) * op("Id", first(μ)) : T

"""
`
probe(O::Vector{ITensor}, μ::Vector{<:Index}; realval=true)
`
Create a function that measures an MPS using the observable
`O`. `μ` are the site indices of the MPS.
"""
function probe(Os::Vector{ITensor}, μ::Vector{<:Index}; realval=true)
    # precompute half-traced superoperators
    tSOs = map(Os) do O
        SO = superoperator(O,I,μ)
        tSO = SO * foldl(*, tracer(x) for x in inds(SO; plev=1); init=one(ITensor) )
        _handle_noind_tensor(tSO, μ)
    end
    _real_or_id = realval ? real : identity

    function wrapper(ρ::MPS)
        dm_expect(ρ, tSOs) .|> _real_or_id
    end
end

probe(A::ITensor, μ::Vector{<:Index}; kwargs...) = probe([A], μ; kwargs...)

"""
`
probe(As::Vector{ITensor}, Bs::Vector{ITensor}, μ::Vector{<:Index}; realval=true)
`
Create a function that measures correlation functions of an
MPS using the observables `As` and `Bs`. `μ` are the site
indices of the MPS.
"""
function probe(As::Vector{ITensor}, Bs::Vector{ITensor}, μ::Vector{<:Index}; realval=true)
    SAs = [_handle_noind_tensor(superoperator(A,I,μ),μ) for A in As]
    SBs = [_handle_noind_tensor(superoperator(B,I,μ),μ) for B in Bs]
    _real_or_id = realval ? real : identity

    function wrapper(ρ::MPS)
        dm_correlator(ρ, SAs, SBs) .|> _real_or_id
    end
end




# """
# `
# probe(A::ITensor, Bμ::Vector{<:Index}; realval=true)
# `
# Create a function that measures an MPS using the observable `O`. `μ` are the site indices of the MPS.
# """
# function probe(O::ITensor, μ::Vector{<:Index}; realval=true)
#     if inds(O) |> isempty
#         return _ -> O[1]
#     end

#     SO = superoperator(O,I,μ)

#     # the measurement pipeline is
#     #    ρ -> apply SO -> trace
#     # For efficiency, we compose the last two
#     #    tSO ≡ (trace ∘ apply SO)
#     tSO = SO * foldl(*, (dag(tracer(x)') for x in inds(SO; plev=0)))

#     function wrapper(ρ::MPS)
#         μ = siteinds(first, ρ; plev=0)
#         traced_ρ = ITensor(one(ComplexF64))
#         for (x,T) in zip(μ,ρ)
#             traced_ρ *= T
#             if commoninds(tSO, T) |> isempty
#                 traced_ρ *= tracer(x)
#             end
#         end

#         res = traced_ρ * tSO
#         @assert isempty(inds(res))
#         return realval ? real(only(res)) : only(res)
#     end
# end

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

decomplexify(z::Complex) = (real(z), imag(z))
decomplexify(z::Array{<:Complex}) = (real.(z), imag.(z))
decomplexify(nt::NamedTuple) = map(pairs(nt) |> collect) do (k,v)
    if hasmethod(decomplexify, (typeof(v),))
        k_re = Symbol(string(k)*"_re")
        k_im = Symbol(string(k)*"_im")
        v_re, v_im = decomplexify(v)
        [k_re=>v_re, k_im=>v_im]
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
