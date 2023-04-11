
struct TrotterSweep
    τ::Float64
    parity::Symbol
    direction::Symbol
end

const TrotterAlgorithm = Vector{TrotterSweep}

const TrotterAlgorithmITensor = [TrotterSweep(0.5, :all, :right), TrotterSweep(0.5, :all, :left)]
const TrotterAlgorithmOrder1 = [TrotterSweep(1., :odd, :right), TrotterSweep(1., :even, :left)]
const TrotterAlgorithmOrder2 = [TrotterSweep(0.5, :even, :right), TrotterSweep(1., :odd, :left), TrotterSweep(0.5, :even, :right)]
const TrotterAlgorithmOrder4 = begin
    τ2 = 1/(4-4^(1/3))
    τ2b = 1/(4-4^(1/3))/2
    τ3 = (1-4/(4-4^(1/3)))
    τ4 = (1-3/(4-4^(1/3)))/2
    TS = TrotterSweep
    [
        TS(τ2b, :even, :right),
        TS(τ2, :odd, :left),
        TS(τ2, :even, :right),
        TS(τ2, :odd, :left),
        TS(τ4, :even, :right),
        TS(τ3, :odd, :left),
        TS(τ4, :even, :right),
        TS(τ2, :odd, :left),
        TS(τ2, :even, :right),
        TS(τ2, :odd, :left),
        TS(τ2b, :even, :right)
    ]
end

function generate_liouvillian(
        hj::ITensor,
        τ::Number,
        noise::Noise
        )::ITensor

    # Identity tensors are used liberally since ITensor can only
    # add Tensors with the same indices.
    l = sort(inds(hj, plev=0), by=sitepos)
    IL = Tuple( prime2braket(op("Id",s), :left)  for s in l)
    IR = Tuple( prime2braket(op("Id",s), :right) for s in l)

    # TODO: replace this mess with mpo2superoperator function
    hL = prime2braket(hj, :left) * IR[1] * IR[2]
    hR = IL[1] * IL[2] * prime2braket(hj, :right)
    liouvillian = 1im * (conj(τ) * hL - τ * hR)
    # @show liouvillian

    for i=(1, 2) # left and right sites of link
        s = l[i]
        Ls = dissipator(noise, s)

        for L=Ls
            LL = prime2braket(L, :left)
            LR = prime2braket(L, :right)
            term1 = LL * dagger(LR)
            term2 = - 1/2 * product(dagger(LL), LL) * IR[i]
            term3 = - 1/2 * IL[i] * product(LR, dagger(LR)) # note that right operators are flipped when applied

            I_index = i==1 ? 2 : 1 # which index not to change
            # factor of 1/2 since each gate applies the dissipator to left and right sites
            liouvillian += 1/2 * real(τ) * (term1 + term2 + term3) * IL[I_index] * IR[I_index]
        end
    end

    return liouvillian

end



function generate_gates(
        algo::TrotterAlgorithm,
        ham::Vector{ITensor},
        τ::Union{Number,Vector{<:Number}},
        noise::Noise
        # Ls::Dict{Site, Vector{ITensor}} = Dict{Site, Vector{ITensor}}()
        )::Vector{ITensor}

    gates = ITensor[]
    Ngates = length(ham)

    if (typeof(τ) <: Number) τ = fill(τ, Ngates) end

    ranges = Dict(:all => 1:Ngates, :odd => 1:2:Ngates, :even=>2:2:Ngates)
    for sweep in algo
        ran = ranges[sweep.parity]
        for l in (sweep.direction==:left ? reverse(ran) : ran)
            S = generate_liouvillian(ham[l], sweep.τ * τ[l], noise)
            push!(gates, exp(S))
        end
    end

    return gates
end

"normalize a density matrix such that the trace is 1"
function normalizedm!(ρ::MPO)
    for n in 1:length(ρ)
        ρ[n] /= tr(ρ[n])[1]
    end
end

"apply the superoperators gates and normalize"
function dosuperstep!(ψ::MPO, gates::Vector{ITensor}; method::TruncationMethod, kwargs...)
    s = extractsites(ψ)
    prime2braket!(ψ)
    apply!(gates, ψ, method; kwargs...)
    braket2prime!(ψ)
end

"Pauli basis"
function dosuperstep!(ψ::MPS, gates::Vector{ITensor}; method::TruncationMethod, kwargs...)
    s = extractsites(ψ)
    apply!(gates, ψ, method; kwargs...)
end

" Use a TEPD algorithm to calculate the time evolution"
function time_evolve!(ψ::ITensors.AbstractMPS, ham::Vector{ITensor},
                measure::Function = ((ψ,t) -> nothing);
                χlim::Int = 256,         # maximum bond dimension
                cutoff::Float64 = 1e-12,   # singular value cutoff
                τ::Float64 = 0.1,          # time per Trotterization step
                t0::Float64 = 0.,          # Initial time
                ttotal::Float64 = 10.0,    # total time
                pm::IO = stdout,
                algo::TrotterAlgorithm = TrotterAlgorithmOrder4,
                noise::Noise = NoNoise(),
                method::TruncationMethod=NaiveTruncation()
            )

    if length(ψ)%2==1 @warn "For best results, use system sizes in powers of 2" end
    s = extractsites(ψ)

    if hastags(s, "Pauli")
        gates = convertToPauli(generate_gates(algo, ham, τ, noise), s)
    else
        gates = generate_gates(algo, ham, τ, noise)
    end

    t = 0.
    N = ceil(Int, ttotal/τ)
    iterator = ProgressBar(1:N, output_stream=pm)

    for step=iterator
        # manually iterate up to t0 so that the ProgressBar is accurate
        if t<t0
            t += τ
            continue
        end
        dosuperstep!(ψ, gates; method, cutoff, maxdim=χlim)
        t += τ
        measure(ψ, t)
        flush(pm)
    end
end


function temp_evolve(ψ₀::MPO, args...; kwargs...)
    ψ = deepcopy(ψ₀)
    temp_evolve!(ψ,args...;kwargs...)
    return ψ
end

function temp_evolve!(ψ::MPO, ham::Vector{ITensor}, T::Vector{Float64};
                τ::Float64 = 0.5,          # time per Trotterization step
                χlim::Int = 256,         # maximum bond dimension
                cutoff::Float64 = 1e-15,   # singular value cutoff
                algo::TrotterAlgorithm = TrotterAlgorithmOrder4
                     )

    s = extractsites(ψ)

    β = 1 ./ T

    nsweeps = ceil(Int, (mean(β)/2) / τ)
    τ = 1im * (β./2) ./ nsweeps

    gates = generate_gates(algo,ham,s,τ)

    for step=1:nsweeps
        dosuperstep!(ψ, gates; method=NaiveTruncation(), cutoff, maxdim=χlim)
    end
end


export
    time_evolve!,
    generate_gates,
    dosuperstep!,
    temp_evolve,
    temp_evolve!
