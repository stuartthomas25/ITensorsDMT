sitepos(s::Site)::Int64 = parse(Int64, match(r"[ln]=(\d+)", join(tags(s)))[1])


function logpurity(ρ::MPO)::Float64
    s = extractsites(ρ)
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


function logtrace(ρ::MPO)::Float64
    s = extractsites(ρ)
    sum = 0
    prod = 1
    for (x,T) in zip(s,ρ)
        prod *= T * delta(x,x')
        f = norm(prod)
        sum += log(f)
        prod /= f
    end
    sum + log(real(prod[1]))
end


function trace(ρ::MPS)
    s = extractsites(ρ)
    prod = 1.
    for (x,T) in zip(s,ρ)
        prod *= T * state(x,1)
    end
    prod[1]
end

function trace(ρ::MPO)
    s = extractsites(ρ)
    prod = 1.
    for (x,T) in zip(s,ρ)
        prod *= 1/sqrt(2) * T * delta(x,x')
    end
    prod[1]
end

function linkdims(ρ::ITensors.AbstractMPS)::Vector{Int64}
    dims = Int64[]
    for i=1:length(ρ)-1
        ix = inds(ρ.data[i])
        ldim = ix[ findfirst(x->hastags(x,"Link"), ix) ]
        χ = ITensors.dim(ldim)
        push!(dims, χ)
    end
    dims
end

function dagger(T::ITensor; braket=false)
    if braket
        T = braket2prime(T)
        T = dagger(T)
        prime2braket!(T)
        return T
    else
        return swapprime(conj.(T),0,1,tags="Site")
    end
end

function dagger(ρ::MPO)
    MPO([dagger(T) for T in ρ.data])
end

function apply!(gates::Vector{ITensor}, ψ::MPO; kwargs...)
    newψ = apply(gates, ψ; kwargs...)
    ψ.data = newψ.data
    ψ.rlim = newψ.rlim
    ψ.llim = newψ.llim

end

function extractsites(ψ::ITensors.AbstractMPS)::Vector{Index{Int64}}
    sites = Index{Int64}[]
    for d in ψ.data
        ix = inds(d)
        s = ix[ findfirst(x->(hastags(x,"Site") & hasplev(x,0)), ix)]
        push!(sites, s)
    end
    sites
end


function prime2braket(T::ITensor, method=:mpo)::ITensor
    T = copy(T)
    prime2braket!(T, method)
    return T
end

function braket2prime(T::ITensor, method=:mpo)::ITensor
    T = copy(T)
    braket2prime!(T, method)
    return T
end

oinds(sbra, sket) = Dict(:mpo => [sket; sbra], :left=> [sket; sket'], :right=> [sbra'; sbra])

function prime2braket!(T::ITensor, method=:mpo)
    ix = inds(T)
    s = ix[ findall(x->(hastags(x,"Site") & hasplev(x, 0)), ix) ]
    s = [x for x in s]
    sket = [addtags(i,"Ket") for i in s]
    sbra = [addtags(i,"Bra") for i in s]

    return replaceinds!(T, [s; s'], oinds(sbra, sket)[method])
end

function braket2prime!(T::ITensor, method=:mpo)
    ix = inds(T)
    sket = [ x for x in ix[ findall(x->(hastags(x,"Ket")), ix) ]]
    sbra = [ x for x in ix[ findall(x->(hastags(x,"Bra")), ix) ]]
    s = removetags(sket,"Ket")

    return replaceinds!(T, oinds(sbra, sket)[method], [s; s'])
end

function prime2braket!(ψ::T) where {T<:ITensors.AbstractMPS}
    for d=ψ.data
        prime2braket!(d, :mpo)
    end
end

function braket2prime!(ψ::T) where {T<:ITensors.AbstractMPS}
    for d=ψ.data
        braket2prime!(d, :mpo)
    end
end

function mpo2superoperator(O::ITensor, method=:right)::ITensor
    ix = inds(O; plev=0)
    if isempty(ix)
        return O
    end
    newO = prime2braket(O, method)
    newO * reduce(*, [prime2braket(op("Id", i), (method==:right ? :left : :right)) for i=ix])
end


ITensors.op(::OpName"X1", ::SiteType"Electron") = [
    0.0 1.0 0.0 0.0
    1.0 0.0 0.0 0.0
    0.0 0.0 0.0 1.0
    0.0 0.0 1.0 0.0
  ]

ITensors.op(::OpName"X2", ::SiteType"Electron") = [
    0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
    1.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0
  ]

ITensors.op(::OpName"Y1", ::SiteType"Electron") = [
    0.0   -1.0im 0.0    0.0
    1.0im  0.0   0.0    0.0
    0.0    0.0   0.0   -1.0im
    0.0    0.0   1.0im  0.0
  ]

ITensors.op(::OpName"Y2", ::SiteType"Electron") = [
    0.0   0.0   -1.0im  0.0
    0.0   0.0    0.0   -1.0im
    1.0im 0.0    0.0    0.0
    0.0   1.0im  0.0    0.0

  ]

ITensors.op(::OpName"Z1", ::SiteType"Electron") = [
    1.0  0.0 0.0  0.0
    0.0 -1.0 0.0  0.0
    0.0  0.0 1.0  0.0
    0.0  0.0 0.0 -1.0
  ]

ITensors.op(::OpName"Z2", ::SiteType"Electron") = [
    1.0 0.0  0.0  0.0
    0.0 1.0  0.0  0.0
    0.0 0.0 -1.0  0.0
    0.0 0.0  0.0 -1.0
  ]



"""Generate array for S=1/2"""

i = Index(2; tags="S=1/2")
const σarrayS = Array{ComplexF64,3}(undef, (2,2,4))
for (μ, S) in zip(1:4, ["Id","X","Y","Z"])
    σarrayS[1:2, 1:2, μ] = Array(op(S,i), i', i) / sqrt(2)
end

"""Generate array for Electron"""

i = Index(4; tags="Electron")
const σarrayE = Array{ComplexF64,3}(undef, (4,4,16))
for (μ, (S1,S2)) in enumerate(Iterators.product(["Id","X1","Y1","Z1"],["Id","X2","Y2","Z2"]))
    σarrayE[1:4, 1:4, μ] = Array( mapprime( op(S1,i)'*op(S2,i), 2=>1), i', i) / sqrt(4)
end


const paulidimdict = Dict(
    "S=1/2" => 4,
    "Electron" => 16
)

const paulidimdictrev = Dict( v => k for (k,v) in paulidimdict)

const σhatDict = Dict(
    "S=1/2" => σarrayS,
    "Electron" => σarrayE
)

sitetype(ρ::ITensors.AbstractMPS) = sitetype(siteinds(first, ρ))
sitetype(s::ITensor) = sitetype(Site[inds(s)...])
function sitetype(s::Union{Site, Vector{Site}})
    for st in keys(σhatDict)
        if hastags(s, st)
            return String(st)
        end
    end
end

function convertToPauli(ρ::MPO; makereal=true)::MPS
    tag = sitetype(ρ)
    paulidim = paulidimdict[tag]
    μ = siteinds(paulidim, length(ρ))
    μ = addtags(μ, "Pauli")
    convertToPauli(ρ, μ; makereal)
end

function convertToPauli(ρ::MPO, μ::Vector{Site}; makereal=true)::MPS
    s = siteinds(ρ)

    MPS( map(1:length(ρ)) do i
        (sbra, sket) = s[i]
        σhat = σhatDict[ sitetype(ρ) ]
        U = ITensor(σhat, sbra, sket, μ[i])

        T = ρ[i] * U
        makereal && @assert all(isapprox.(Array(imag.(T), inds(T)...),0; atol=1e-12))
        makereal ? real(T) : T
    end)
end



convertToSpin(ρ::MPS)::MPO = convertToSpin(ρ, siteinds( paulidimdictrev[ dim( siteinds(ρ)[1] ) ], length(ρ) ) )

function convertToSpin(ρ::MPS, s::Vector{Site})::MPO
    # ρ = truncate(ρ; cutoff=1e-16) # this is necessary for convertToPauli to accurately make a Real tensor network
    μ = siteinds(ρ)

    σhat = σhatDict[ paulidimdictrev[ dim( siteinds(ρ)[1] ) ] ]
    MPS( map(1:length(ρ)) do i
        x = s[i]
        U = ITensor(σhat, x', x, μ[i])
        T = ρ[i] * U
    end)
end


function convertToPauli(Ts::Vector{ITensor}, pauliSites::Vector{Site})::Vector{ITensor}
    map(Ts) do T

        if isempty(inds(T)) return T end

        hastags(T, "Pauli") && throw("Tensor is already in Pauli basis!")
        hastags(T, "Bra") || throw("Tensor must be in superoperator form. Use `mpo2superoperator` to convert`")

        xs = unique([sitepos(x) for x in inds(T)])

        σhat = σhatDict[ sitetype(T) ]

        Us = map(xs) do x
            sites = filter(hastags("n=$x"), pauliSites)
            length(sites)==1 || throw("pauliSites must have only one index per site!")
            μ = sites[1]

            dummyInd = Index(dim(μ), "Dummy")
            jket = get( inds(T; tags="Site,Ket,n=$x", plev=0), 1, dummyInd)
            jbra = get( inds(T; tags="Site,Bra,n=$x", plev=0), 1, dummyInd)

            @assert !(dummyInd in [jket, jbra]) # dummyInd is deprecated, but just in case
            ITensor(σhat, jket, jbra, μ)
            #deltaTensor = (dummyInd in [jket, jbra]) ? delta(dummyInd, dummyInd') : ITensor(1)

            # U = ITensor(σhat, jket, jbra, μ)
            # conj.(U) * U' * deltaTensor
        end
        U = reduce(*, Us)
        real(U' * T * conj.(U))
    end
end
convertToPauli(T::ITensor, pauliSites::Vector{Site})::ITensor = convertToPauli([T], pauliSites)[1]

"""Take a vector of Pauli operators and add identities such that their support is the same"""
function addIdentities(ts)
    length(ts)==0 && return ts
    is = unioninds(ts..., plev=0)
    map(ts) do t
        newix = uniqueinds(is, inds(t))
        if length(newix)>0
            t * reduce(*, [op("Id", i) for i in newix])
        else
            t
        end
    end
end

"""remove any superfluous identities in a Pauli string"""
function removeIdentities(T)
    s = inds(T; plev=0)
    for i in reverse(eachindex(s)) # work from back so as not to mess up indexing
        others = vcat(s[1:i-1], s[i+1:end])
        C = combiner(others..., (others')...)
        U = C * T
        A = Array(U, s[i], s[i]', combinedind(C))
        if all(A[1,2,:] .== A[2,1,:] .== 0) && all(A[1,1,:] .== A[2,2,:])
            T *= onehot(s[i]=>1) * onehot(s[i]'=>1)
        end
    end
    T
end

function _localop(ρ::ITensors.AbstractMPS, Os::Vector{ITensor})::Vector{Float64}
    map(Os) do O
        if isempty(inds(O)) return O[1] end # little hack to include 0-dim tensors
        real( trace( product(O, ρ) )  )
    end
end

""" Find the expectation of a local operator """
localop(ρ::MPO, Os::Vector{ITensor})::Vector{Float64} = ₗ_localop(ρ, Os)

function localop(ρ::MPS, Os::Vector{ITensor})::Vector{Float64}
    if sitetype(Os[1]) != "Pauli"
        μ = siteinds(first, ρ; plev=0)
        soOs = mpo2superoperator.( Os )
        Os = convertToPauli( soOs, μ )
    end

    _localop(ρ, Os)
end
#localop(ρ::MPS, Os::Vector{ITensor}) = localop(convertToSpin(ρ, unioninds(Os...; plev=0)), Os)
#
export
    extractsites,
    normalizedm!,
    convertToPauli,
    convertToSpin,
    sitepos,
    trace,
    dagger,
    spin,
    mpo2superoperator,
    localop