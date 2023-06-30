
function gate(
        h::ITensor,
        δt::Number,
        μ::Vector{<:Index},
        noise::Noise=NoNoise()
        )::ITensor

    hL = superoperator(h, I, μ)
    hR = superoperator(I, h, μ)

    liouvillian = 1im * (conj(δt) * hL - δt * hR)

    # add Lindbladians for open system
    l = sort(inds(h, plev=0), by=sitepos)
    idT = superoperator(I, I, commoninds(hL, μ))
    for s=l
        Ls = dissipator(noise, s)

        for L=Ls
            LL = superoperator(L, I, μ)
            LR = superoperator(I, L, μ)
            term1 =        product(LL, dagger(LR))
            term2 = -1/2 * product(dagger(LL), LL)
            term3 = -1/2 * product(LR, dagger(LR)) # note that right operators are flipped when applied

            # factor of 1/length(l) due to Trotterization
            liouvillian += 1/length(l) * real(δt) * product(term1 + term2 + term3, idT)
        end
    end

    L = clean(liouvillian)
    if L.tensor.storage isa NDTensors.EmptyStorage # this can happen if the Liouvillian is the identity
        return exp(liouvillian)
    end

    return exp(L)

end

make_sweep(g, algo::TrotterAlgorithm, N::Int, τ::Number) =
    make_sweep(g, algo, N, fill(τ, N))

function make_sweep(
        g::Function,
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

export
    gate,
    make_sweep
