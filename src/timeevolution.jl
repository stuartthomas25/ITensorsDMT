
function gate(
        h::ITensor,
        δt::Number,
        μ::Vector{<:Index},
        noise::Noise=NoNoise()
        )::ITensor

    hL = superoperator(h, I, μ)
    hR = superoperator(I, h, μ)

    liouvillian = 1im * (conj(δt) * hL - δt * hR)

    l = sort(inds(h, plev=0), by=sitepos)
    idT = superoperator(I, I, collect(l))
    for i=eachindex(l)
        s = l[i]
        Ls = dissipator(noise, s)

        for L=Ls
            LL = superoperator(L, I, μ)
            LR = superoperator(I, L, μ)
            term1 = LL * dagger(LR) * idT
            term2 = - 1/2 * product(dagger(LL), LL) * idT
            term3 = - 1/2 * idT * product(LR, dagger(LR)) # note that right operators are flipped when applied

            # factor of 1/length(l) due to Trotterization
            liouvillian += 1/length(l) * real(δt) * (term1 + term2 + term3)
        end
    end

    return exp(ITensors.dropzeros(liouvillian))

end

make_sweep(g, algo::TrotterAlgorithm, N::Int, τ::Number) =
    make_sweep(g, algo, N, fill(τ, N))

function make_sweep(
        g,
        algo::TrotterAlgorithm,
        N::Int,
        τ::Vector{<:Number}
        )

    gates = Tuple{Int, Number}[]

    for sweep in algo
        ran = (1+sweep.offset):abs(sweep.step):N
        ran = sweep.step < 0 ? reverse(ran) : ran
        for l in ran
            push!(gates, (l, sweep.τ * τ[l]))
        end
    end

    return map(g, gates)
end

function generate_gates(
        algo::TrotterAlgorithm,
        ham::Vector{ITensor},
        τ::Union{Number,Vector{<:Number}},
        noise::Noise,
        μ::Vector{<:Index}
        )::Vector{ITensor}

    gates = ITensor[]
    Ngates = length(ham)

    if (typeof(τ) <: Number) τ = fill(τ, Ngates) end

    for sweep in algo
        ran = (1+sweep.offset):abs(sweep.step):Ngates
        ran = sweep.step < 0 ? reverse(ran) : ran
        for l in ran
            S = generate_liouvillian(ham[l], sweep.τ * τ[l], noise, μ)
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

# "apply the superoperators gates and normalize"
# function dosuperstep!(ψ::MPO, gates::Vector{ITensor}; method::TruncationMethod, kwargs...)
#     s = siteinds(first, ψ; plev=0)
#     prime2braket!(ψ)
#     apply!(gates, ψ, method; kwargs...)
#     braket2prime!(ψ)
# end

"Pauli basis"
function dosuperstep!(ψ::MPS, gates::Vector{ITensor}; method::TruncationMethod, kwargs...)
    s = siteinds(first, ψ; plev=0)
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
    s = siteinds(first, ψ; plev=0)

    if ψ isa MPO
        stype = string(sitetype(s))
        newstype = "Super$stype"
        μ = siteinds(newstype, length(s))
        ψ = MPS(ψ, μ)
        gates = generate_gates(algo, ham, τ, noise, μ)
    else
        gates = generate_gates(algo, ham, τ, noise, s)
    end

    t = 0.
    N = ceil(Int, ttotal/τ)
    iterator = ProgressBar(1:N, output_stream=pm)

    for step=iterator
        if t<t0 # manually iterate up to t0 so that the ProgressBar is accurate
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

    s = siteinds(first, ψ; plev=0)

    β = 1 ./ T

    nsweeps = ceil(Int, (mean(β)/2) / τ)
    τ = 1im * (β./2) ./ nsweeps

    gates = generate_gates(algo,ham,τ,NoNoise())

    for step=1:nsweeps
        dosuperstep!(ψ, gates; method=NaiveTruncation(), cutoff, maxdim=χlim)
    end
end


export
    time_evolve!,
    generate_gates,
    dosuperstep!,
    temp_evolve,
    temp_evolve!,
    gate,
    make_sweep
