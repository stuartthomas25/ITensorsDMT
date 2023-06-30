isinteractive() && using Revise
using Test
using ITensors
import Random
using TEBD
using LinearAlgebra
using Distributions: Uniform

function check_hermiticity(ρ::MPS, s::Vector{<:Index})
    mpo = MPO(ρ, s)
    dot(mpo, dagger(mpo)) / dot(mpo, mpo) ≈ 1.
end

@testset "Superoperator evolution" begin
    """
    Test that superoperator evolution is equivalent to normal evolution
    """
    N = 10
    s = siteinds("S=1/2", N)
    μ = siteinds("PauliOperator", N)
    steps = 4
    τ = 0.5
    W = 0.3

    for method in [DMT(), NaiveTruncation()]
        mpo = MPO(s, [n==N÷2 ? "Z" : "Id" for n=1:N])
        mps = MPS(mpo, μ)

        b = rand(Uniform(-W,W), N)
        ham = Models.TFIM(s; b)

        mpo_gates = make_sweep( TA.ITensor, length(ham), τ) do (j,δt)
            exp(-1.0im * δt * ham[j])
        end
        mps_gates = make_sweep( TA.ITensor, length(ham), τ) do (j,δt)
            gate(ham[j], δt, μ)
        end

        for i=1:steps
            apply!(mpo_gates, mpo; apply_dag=true)
            apply!(mps_gates, mps, method)
        end

        mpo′ = MPO(mps, s)
        normalize!(mpo)
        normalize!(mpo′)

        @test dot(mpo, mpo′) ≈ 1
    end
end

@testset "Identity" begin
    """
    Test that identity is H eigenstate
    """
    N = 10
    s = siteinds("S=1/2", N)
    μ = siteinds("PauliOperator", N)
    steps = 3
    W = 0.4
    τ = 0.5
    ham = Models.TFIM(s; b=rand(Uniform(-W,W), N))

    ρ = MPS(μ, "Id")
    normalize!(ρ)
    ρ′ = deepcopy(ρ)

    gates = make_sweep( TA.Order4, length(ham), τ) do (j,δt)
        gate(ham[j], δt, μ)
    end

    for i=1:steps
        apply!(gates, ρ′)
    end

    @test dot(ρ, ρ′) ≈ 1
end

@testset "Hermiticity" begin
    """
    Test Hermiticity
    """
    N = 10
    s = siteinds("S=1/2", N)
    μ = siteinds("PauliOperator", N)
    steps = 3
    W = 0.4
    τ = 0.5
    ham = Models.TFIM(s; b=rand(Uniform(-W,W), N))

    mpo = MPO(s, "Sy")
    mps = MPS(mpo, μ)

    gates = make_sweep( TA.Order4, length(ham), τ) do (j,δt)
        gate(ham[j], δt, μ)
    end

    for i=1:steps
        apply!(gates, mps)
    end

    @test check_hermiticity(mps, s)
end

@testset "Lindbladian" begin
    """
    Test non-unitary time evolution
    """
    N = 10
    s = siteinds("S=1/2", N)
    μ = siteinds("PauliOperator", N)
    τ = 0.2
    W = 1.0
    γ = 0.4
    steps = 2

    ham = Models.TFIM(s; b=rand(Uniform(-W,W), N))
    clean_ham = Models.TFIM(s)

    ψ = randomMPS(s)
    ρ₀ = MPS(outer(ψ', ψ), μ)

    ρ1, ρ2, ρ3 = [deepcopy(ρ₀) for _ in 1:3]
    g1 = make_sweep( TA.Order1, length(ham), τ) do (j,δt)
        gate(clean_ham[j], δt, μ)
    end
    g2 = make_sweep( TA.Order1, length(ham), τ) do (j,δt)
        gate(clean_ham[j], δt, μ, DepolarizingNoise(γ))
    end
    g3 = make_sweep( TA.Order1, length(ham), τ) do (j,δt)
        gate(ham[j], δt, μ, DepolarizingNoise(γ))
    end

    for i=1:steps
        apply!(g1, ρ1)
        apply!(g2, ρ2)
        apply!(g3, ρ3)
    end


    @test !(dot(ρ1,ρ2) ≈ 0) # not equivalent to unitary evolution
    @test check_hermiticity(ρ1, s)
    @test check_hermiticity(ρ2, s)
    @test trace(ρ2) ≈ trace(ρ₀)
    @test trace(ρ3) ≈ trace(ρ₀)
end

@testset "Trotter algorithms" begin
    """
    Test that superoperator evolution is equivalent to normal evolution
    """
    N = 10
    s = siteinds("S=1/2", N)
    μ = siteinds("PauliOperator", N)
    steps = 3
    τ = 0.1
    W = 0.3
    ham = Models.TFIM(s; b=rand(Uniform(-W,W), N))

    algos = [TA.ITensor,
             TA.Order1,
             TA.Order2,
             TA.Order4]

    ρ₀ = MPS(MPO(s, "Sy"), μ)
    normalize!(ρ₀)

    mpos = map(algos) do algo
        ρ = deepcopy(ρ₀)
        gates = make_sweep( algo, length(ham), τ) do (j,δt)
            gate(ham[j], δt, μ)
        end

        for i=1:steps
            apply!(gates, ρ)
        end
        normalize(ρ)
    end

    ref = mpos[end]
    for mpo in mpos
        @test  isapprox( dot(ref, mpo) , 1., atol=1e-2)
        @test !isapprox( dot(ρ₀, mpo) , 1., atol=1e-2)
    end
end

@testset "n=3 Trotter algorithms" begin
    """
    Test that superoperator evolution is equivalent to normal evolution
    """
    N = 10
    s = siteinds("Fermion", N)
    μ = siteinds("FermionOperator", N)
    steps = 3
    τ = 0.1
    W = 0.3
    ham = Models.MOH(s)

    algos = [TA.Order1n3,
             TA.Order4n3]

    # alternating occupied and empty
    ρ₀ = MPS(MPO(s, [iseven(i) ? "N" : "Id - N" for i in eachindex(s) ]), μ)
    normalize!(ρ₀)

    mpos = map(algos) do algo
        ρ = deepcopy(ρ₀)
        gates = make_sweep( algo, length(ham), τ) do (j,δt)
            gate(ham[j], δt, μ)
        end

        for i=1:steps
            apply!(gates, ρ)
        end
        normalize(ρ)
    end

    ref = mpos[end]
    for mpo in mpos
        @test  isapprox( dot(ref, mpo) , 1., atol=1e-2)
        @test !isapprox( dot(ρ₀, mpo) , 1., atol=1e-2)
    end
end

@testset "DMT" begin
    """
    Test the guarantees of DMT:
    1. Trace is preserved
    2. length 3 operators are preserved
    """
    N         = 128
    τ         = 0.5
    steps     = 8
    s         = siteinds("S=1/2",N)
    μ         = siteinds("PauliOperator", N)
    ham       = Models.MFIM(s)
    ρ₀        = MPS(μ, "Id")

    # a three site operator to test DMT's three site guarantee
    o_f       = probe([op("Z", s[i]) * op("Z", s[i+1]) * op("Z", s[i+2]) for i in 1:N-2], μ)
    ε_f       = probe(ham[N÷2], μ)
    x         = s[N÷2]
    cutoff    = 1e-12
    maxdim = 16
    truncmaxdim = 10

    # verify that tests fail for Naive Truncation but pass for DMT
    for (method, expresult) in zip([DMT(), NaiveTruncation()], [true, false])

        # try normal and traceless cases
        for (V, testtrace) in zip([op("Id",x) + op("Z",x), ham[N÷2]], [true, false])

            ρ = product(superoperator(V,I,μ), ρ₀)

            gates = make_sweep( TA.Order1, length(ham), τ) do (j,δt)
                gate(ham[j], δt, μ)
            end

            for i=1:steps
                apply!(gates, ρ, method; maxdim, cutoff)
            end

            @test all(linkdims(ρ) .<= maxdim)

            # test trace
            if testtrace
                @test isapprox(trace(ρ), 1.; rtol = 1e-8) == expresult
            end

            Oold = o_f(ρ)

            # truncate down to truncmaxdim, but don't time-evolve
            H′ = op("Id", s[N÷2]) * op("Id", s[N÷2+1])
            idGate = gate(H′, 1., μ)
            apply!([idGate], ρ, method; maxdim=truncmaxdim, cutoff)

            Onew = o_f(ρ)

            @test linkdim(ρ, N÷2) <= truncmaxdim

            @test isapprox(Onew, Oold; rtol=1e-10) == expresult
        end
    end
end

@testset "DMT Fermions" begin
    """
    Test the guarantees of DMT:
    1. Trace is preserved
    2. length 2 operators are preserved
    """
    N           = 128
    τ           = 0.5
    steps       = 8
    conserve_nf = true
    s           = siteinds("Fermion",N; conserve_nf)
    μ           = siteinds("FermionOperator", N; conserve_nf)
    ham         = Models.MOH(s)
    ρ₀          = MPS(μ, "Id")
    # O         = op("Z", s[N÷2-1]) * op("Z", s[N÷2+1]) # a three site operator to test DMT's three site guarantee


    # a three site operator to test DMT's three site guarantee
    o_f         = probe([op("N", s[i]) * op("N", s[i+1]) * op("N", s[i+2]) for i in 1:N-2], μ)
    ε_f         = probe(ham[N÷2], μ)
    x           = s[N÷2]
    cutoff      = 1e-12
    maxdim      = 16
    truncmaxdim = 10

    # verify that tests fail for Naive Truncation but pass for DMT
    for (method, expresult) in zip([DMT(), NaiveTruncation()], [true, false])

        # try normal and traceless cases
        for (V, testtrace) in zip([ham[N÷2]], [true])

            ρ = product(superoperator(V,I,μ), ρ₀)

            trace_old = trace(ρ)

            gates = make_sweep( TA.Order1, length(ham), τ) do (j,δt)
                gate(ham[j], δt, μ)
            end

            for i=1:steps
                apply!(gates, ρ, method; maxdim, cutoff)
            end

            @test all(linkdims(ρ) .<= maxdim)

            # test trace
            if testtrace
                @test isapprox(trace(ρ), trace_old; rtol = 1e-8) == expresult
            end

            Oold = o_f(ρ)

            # truncate down to truncmaxdim, but don't time-evolve
            # also test three-site gates
            H′ = op("Id", s[N÷2-1]) * op("Id", s[N÷2]) * op("Id", s[N÷2+1])
            idGate = gate(H′, 1., μ)

            apply!([idGate], ρ, method; maxdim=truncmaxdim, cutoff)

            Onew = o_f(ρ)

            @test linkdim(ρ, N÷2) <= truncmaxdim

            @test isapprox(Onew, Oold; rtol=1e-10) == expresult
        end
    end
end

@testset "Pauli Commutation" begin
    """
    Verify that the the MPS to MPO transformation is natural, i.e.
    follows the commutative diagram below.

           apply h operator
    I::MPO  ----------------->  ρ::MPO

    |                            |
    | MPO                        | MPO
    | to                         | to
    | MPS                        | MPS
    |                            |
    |                            |
    V                            V
       apply h⊗I superoperator
    I::MPS ------------------> ρ::MPS

    """

    N = 5
    s = siteinds("S=1/2",N)
    μ = siteinds("PauliOperator",N)

    ham = Models.MFIM(s)
    ρ = MPO(s, "Id")
    h  = ham[N÷2]

    ψ = MPS(ρ, μ)

    ρ1 = product(superoperator(h,I,μ), ψ)
    ρ2 = product(h, ρ)

    ρ2 = MPS(ρ2, μ)

    m0 = Array(reduce(*, MPS(ρ,μ)), μ...)
    m1 = Array(reduce(*, ρ1), μ...)
    m2 = Array(reduce(*, ρ2), μ...)

    @test  isapprox(m1, m2; atol=1e-14)
    @test !isapprox(m0, m1; atol=1e-14)
    @test !isapprox(m0, m2; atol=1e-14)
end

@testset "Time doubling" begin
    """
    Test that the time-doubling trick matches the normal measurement
    """
    N = 16
    s = siteinds("S=1/2", N)
    μ = siteinds("PauliOperator",N)
    τ = 0.1
    maxdim = 64
    cutoff = 1e-12
    steps = 10
    method = DMT()

    ham = Models.MFIM(s)
    ε_f = probe(ham, μ)

    ρ₀ = MPS(μ, "Id")
    h₀ = ham[N÷2]

    function make_measurement(ρ)
        (ε_f(ρ), Θexpect(ρ, ρ))
    end

    ρ = product(superoperator(h₀,I,μ), ρ₀); # initial energy state

    data = [make_measurement(ρ)]

    gates = make_sweep( TA.Order4, length(ham), τ) do (j,δt)
        gate(ham[j], δt, μ)
    end

    for i=1:steps
        apply!(gates, ρ, method; maxdim, cutoff)
        push!(data, make_measurement(ρ))
    end

    fullε   = hcat(first.(data)...)
    ε       = fullε[:, 1:2:end]
    fullε2t = hcat(last.(data)...)
    ε2t     = fullε2t[N÷2+1:end-N÷2, 1:size(ε, 2)]

    @test all(isapprox.(ε, ε2t; atol=0.02))
end
