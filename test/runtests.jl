using Test
using ITensors
import Random
using TEBD
using LinearAlgebra
using Distributions: Uniform



@testset "Superoperator evolution" begin
    # test that superoperator evolution is equivalent to normal evolution
    N = 10
    s = siteinds("S=1/2", N)
    steps = 4
    τ = 0.5
    W = 0.3

    # mpo0 = MPO(s, "Sy");
    mpo0 = MPO(s, [n==N÷2 ? "Z" : "Id" for n=1:N])
    mpo1 = deepcopy(mpo0)
    mpo2 = deepcopy(mpo0)

    b = rand(Uniform(-W,W), N)
    ham = TFIM(s; b)
    time_evolve!(mpo1, ham; τ=τ, ttotal=τ*steps, algo=TEBD.TrotterAlgorithmITensor)

    gates = ITensor[]
    for j=1:N-1
        Gj = exp(-1.0im * τ/2 * ham[j])
        push!(gates,Gj)
    end
    append!(gates, reverse(gates))


    for i=1:steps
        apply!(gates, mpo2; apply_dag=true)
    end

    normalize!(mpo1)
    normalize!(mpo2)


    @test dot(mpo1,mpo2) ≈ 1
end

@testset "Identity" begin
    # test that identity is H eigenstate
    N = 10
    s = siteinds("S=1/2", N)
    steps = 3
    W = 0.4
    τ = 0.5
    b = rand(Uniform(-W,W), N)
    ham = TFIM(s; b)

    mpo0 = MPO(s, "Id")
    normalize!(mpo0)
    mpo = deepcopy(mpo0)
    time_evolve!(mpo, ham; τ=τ, ttotal=τ, algo=TEBD.TrotterAlgorithmOrder4)
    # gates = TEBD.generate_gates(TEBD.TrotterAlgorithmOrder4, ham, τ, 0.)
    # TEBD.dosuperstep!(mpo, gates)

    @test dot(mpo,mpo0) ≈ 1
end

@testset "MPO evolution vs MPS evolution" begin
    N = 10
    s = siteinds("S=1/2", N)
    steps = 4
    τ = 0.5
    W = 0.3

    mpo0 = MPO(s, [n==N÷2 ? "Z" : "Id" for n=1:N])
    b = rand(Uniform(-W,W), N)
    ham = TFIM(s; b)

    ψ0 = productMPS(s, n -> "Up")

    ψ   = deepcopy(ψ0)
    mpo = deepcopy(mpo0)
    mpotest = deepcopy(mpo0)

    gates = ITensor[]

    gates = ITensor[]
    for j=1:N-1
        Gj = exp(1.0im * τ/2 * ham[j]) # we evolve backwards in time here because `time_evolve!` is designed to work with density matrices, not observables
        push!(gates,Gj)
    end
    append!(gates, reverse(gates))

    for i=1:steps
        apply!(gates, ψ)
    end


    time_evolve!( mpo, ham; τ=τ, ttotal=τ*steps, algo=TEBD.TrotterAlgorithmITensor)

    normalize!(mpo)
    normalize!(mpo0)

    @test isapprox(inner(ψ0', (mpo * ψ0)), inner(ψ', (mpo0 * ψ)), atol=1e-4)
end

@testset "Hermiticity" begin
    # test Hermiticity
    N = 10
    s = siteinds("S=1/2", N)
    steps = 3
    W = 0.4
    b = rand(Uniform(-W,W), N)
    τ = 0.5
    ham = TFIM(s; b)

    mpo = MPO(s, "Sy")

    time_evolve!( mpo, ham; τ=τ, ttotal=τ*steps, algo=TEBD.TrotterAlgorithmOrder4)

    ψ = randomMPS(ComplexF64, s)
    @test isapprox(inner(ψ',mpo,ψ), inner(ψ',dagger(mpo),ψ), atol=1e-10)
end

@testset "Lindbladian" begin
    N = 10
    s = siteinds("S=1/2", N)
    τ = 0.2
    W = 1.0
    γ = 0.4
    steps = 1;

    b = rand(Uniform(-W,W), N)

    ψ = randomMPS(s)
    ρ₀ = outer(ψ', ψ)
    ρ  = deepcopy(ρ₀)
    ρc = deepcopy(ρ₀)
    ρu = deepcopy(ρ₀)


    ham = TFIM(s; b)
    clean_ham = TFIM(s)

    time_evolve!( ρ, ham; τ=τ, noise=DepolarizingNoise(γ), ttotal=τ*steps, algo=TEBD.TrotterAlgorithmOrder1)
    time_evolve!( ρc, clean_ham; τ=τ, noise=DepolarizingNoise(γ), ttotal=τ*steps, algo=TEBD.TrotterAlgorithmOrder1)
    time_evolve!( ρu, clean_ham; τ=τ, noise=NoNoise(), ttotal=τ*steps, algo=TEBD.TrotterAlgorithmOrder1)

    @test trace(ρ) ≈ trace(ρ₀) # trace-preserving
    @test !(dot(ρ,ρu) ≈ 0) # not equivalent to unitary evolution
    @test (dot(ρ, dagger(ρ)) / dot(ρ, ρ)) ≈ 1 # hermitian

end

@testset "Trotter algorithms" begin
    # test that superoperator evolution is equivalent to normal evolution
    N = 64
    s = siteinds("S=1/2", N)
    steps = 3
    τ = 0.1
    W = 0.3
    b = rand(Uniform(-W,W), N)
    ham = TFIM(s; b)

    algos = [TEBD.TrotterAlgorithmITensor,
             TEBD.TrotterAlgorithmOrder1,
             TEBD.TrotterAlgorithmOrder2,
             TEBD.TrotterAlgorithmOrder4]

    mpo0 = MPO(s, "Sy");


    mpos = map(algos) do algo
        mpo = deepcopy(mpo0)
        time_evolve!(mpo, ham; τ=τ, ttotal=τ*steps, algo=algo)
        normalize(mpo)
    end

    for mpo in mpos
        @test isapprox( dot(mpos[end],mpo) , 1, atol=1e-2)
    end
end


@testset "Pauli basis" begin
    N = 4
    s = siteinds("S=1/2",N)

    ham = TFIM(s)
    ρ₀ = MPO(s, ["Id" for x in s])
    # normalizedm!(ρ₀)
    h₀ = ham[N÷2]
    ρ = product(h₀, ρ₀); # initial energy state
    truncate!(ρ; cutoff=1e-16) # this is necessary for convertToPauli to accurately make a Real tensor network

    TestAlgo = [TEBD.TrotterSweep(1., :odd, :right)]

    options = (noise=NoNoise(),
            ttotal=2.5,
            τ=0.5,
            χlim=64,
            cutoff=1e-16,
            algo=TEBD.TrotterAlgorithmOrder1)

    ρ1 = deepcopy(ρ)
    ρ1 = convertToPauli(ρ1)
    time_evolve!(ρ1, ham; options...)

    ρ2 = deepcopy(ρ)
    time_evolve!(ρ2, ham; options...)
    ρ2 = convertToPauli(ρ2, siteinds(ρ1); makereal=false)

    @test isapprox(reduce(*, ρ1), reduce(*, ρ2), atol=1e-12)

    ρP = convertToPauli(ρ, makereal=false)
    ρS = convertToSpin(ρP, s)

    @show isapprox(reduce(*, ρ), reduce(*, ρS), atol=1e-10)

end



@testset "DMT" begin
    N = 128
    s = siteinds("S=1/2",N)

    ham = TFIM(s)
    ρ₀ = infTempState(s)
    x = s[N÷2]
    χlim = 16
    truncχlim = 8

    # verify that tests fail for Naive Truncation but pass for DMT
    for (method, expresult) in zip([TEBD.DMT(), TEBD.NaiveTruncation()], [true, false])

        # try normal and traceless cases
        for (h₀, testtrace) in zip([op("Id",x) + op("Z",x), ham[N÷2]], [true, false])

            ρ = product(h₀, ρ₀); # initial energy state, high cutoff since we know that all element are on the order of unity at this point
            truncate!(ρ; cutoff=1e-16)
            ρ = convertToPauli(ρ)


            time_evolve!(ρ, ham;
                            ttotal=20.,
                            τ=1.0,
                            χlim=χlim,
                            method=method,
                            algo=TEBD.TrotterAlgorithmOrder1)

            @test linkdim(ρ,N÷2) <= χlim

            if testtrace
                @test isapprox(trace(ρ), 1; rtol = 1e-8) == expresult
            end

            Eold = energy_density(ρ, ham)[N÷2]

            # just truncate, don't time-evolve
            idGate = op("Id", s[N÷2]) * op("Id", s[N÷2+1])
            idGate = TEBD.prime2braket(idGate, :left) * TEBD.prime2braket(idGate, :right)
            idGate = convertToPauli(idGate, siteinds(ρ))
            apply!([idGate], ρ, method; maxdim=truncχlim)

            Enew = energy_density(ρ, ham)[N÷2]

            @test linkdim(ρ,N÷2) <= truncχlim

            @test isapprox(Enew, Eold; rtol=1e-10) == expresult
        end
    end
end

@testset "DMT Superoperator evolution" begin
    # test that superoperator evolution is equivalent to normal evolution
    N = 10
    s = siteinds("S=1/2", N)
    steps = 4
    τ = 0.5
    W = 0.3

    # mpo0 = MPO(s, "Sy");
    mpo0 = MPO(s, [n==N÷2 ? "Z" : "Id" for n=1:N])
    mpo1 = convertToPauli(deepcopy(mpo0))
    mpo2 = deepcopy(mpo0)

    b = rand(Uniform(-W,W), N)
    ham = TFIM(s; b)
    time_evolve!(mpo1, ham; τ=τ, ttotal=τ*steps, algo=TEBD.TrotterAlgorithmITensor, method=TEBD.DMT())

    gates = ITensor[]
    for j=1:N-1
        Gj = exp(-1.0im * τ/2 * ham[j])
        push!(gates,Gj)
    end
    append!(gates, reverse(gates))


    for i=1:steps
        apply!(gates, mpo2; apply_dag=true)
    end

    mpo1 = convertToSpin(mpo1, s)
    normalize!(mpo1)
    normalize!(mpo2)

    @test dot(mpo1,mpo2) ≈ 1
end

@testset "Pauli Commutation" begin

    N = 5
    s = siteinds("S=1/2",N)

    ham = TFIM(s)
    ρ₀ = MPO(s, ["Id" for x in s])
    normalizedm!(ρ₀)
    ρ₀P = convertToPauli(ρ₀)

    μ = siteinds(first, ρ₀P)

    h₀  = ham[N÷2]
    h₀P = convertToPauli(mpo2superoperator(h₀), μ)

    ρ1 = product(h₀P, ρ₀P);
    ρ2 = product(h₀, ρ₀)

    truncate!(ρ2; cutoff=1e-16) # this is necessary for convertToPauli to accurately make a Real tensor network
    ρ2 = convertToPauli(ρ2, μ)

    m1 = Array(reduce(*, ρ1), μ...)
    m2 = Array(reduce(*, ρ2), μ...)

    @test isapprox(m1, m2; atol=1e-15)
end

@testset "Time doubling" begin
    N = 32
    s = siteinds("S=1/2", N)
    τ = 0.5
    χlim = 32
    ttotal = 3.5 #1/γ


    ham = MFIM(s; hz=0.9045)

    ρ₀ = infTempState(s)
    h₀ = ham[N÷2]

    make_measurement(ρ, t) = (localop(ρ, ham), h2t(ρ))

    ρ = product(h₀, ρ₀); # initial energy state
    truncate!(ρ; cutoff=1e-16) # this is necessary for convertToPauli to accurately make a Real tensor network
    ρ = convertToPauli(ρ)

    data = [make_measurement(ρ, 0.)]
    time_evolve!(   ρ,
                    ham,
                    (ρ,t) -> push!(data, make_measurement(ρ,t));
                    t0=0.,
                    ttotal=ttotal,
                    τ=τ,
                    χlim=χlim,
                    cutoff=1e-15,
                    pm=stdout
                )

    (rawε, rawε2t) = collect(zip(data...))
    ε   = permutedims(hcat(rawε...))[1:2:end, :]
    ε2t = permutedims(hcat(rawε2t...))[1:length(data)÷2, N÷2+1:end-N÷2]

    @test all(isapprox.(ε, ε2t; atol=0.02))
end
