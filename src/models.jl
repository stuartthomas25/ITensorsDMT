export Models

module Models
    using ITensors

    function Ising(s::Vector{<:Index})::Vector{ITensor}
        map(1:length(s)-1) do j
            op("Sz",s[j]) * op("Sz",s[j+1])
        end
    end



    function Trivial(s::Vector{<:Index})::Vector{ITensor}
        map(1:length(s)-1) do j
            op("Id",s[j]) * op("Id",s[j+1])
        end
    end

    function TFIM(s::Vector{<:Index}; b::Union{Vector{<:Real},Real}=0.)::Vector{ITensor}
        hs = ITensor[]
        N = length(s)
        if (typeof(b) <: Number) b = fill(b, N) end
        for j=1:N-1
            s1,s2  = s[j], s[j+1]
            hj =                op("Sz",s1) * op("Sz",s2)    +
                        1/2 * op("Sz",s1) * op("Id",s2)    +
                (1+b[j])/2 * op("Sx",s1) * op("Id",s2)    +
                        1/2 * op("Id",s1) * op("Sz",s2)    +
                (1+b[j])/2 * op("Id",s1) * op("Sx",s2)

            if j==1
                hj +=     1/2 * op("Sz",s1) * op("Id",s2)    +
                (1+b[j])/2 * op("Sx",s1) * op("Id",s2)
            elseif j==N-1
                hj +=     1/2 * op("Id",s1) * op("Sz",s2)    +
                (1+b[j])/2 * op("Id",s1) * op("Sx",s2)
            end
            push!(hs, hj)
        end
        hs
    end

    function MFIM(s::Vector{<:Index}; hx::Float64=1.4, hz::Float64=0.9045)::Vector{ITensor}
        hs = ITensor[]
        N = length(s)
        for j=1:N-1
            s1,s2  = s[j], s[j+1]
            hj =            4 * op("Sz",s1) * op("Sz",s2)  +
                        hz * op("Sz",s1) * op("Id",s2)  + # factor of 2 from Pauli matrices cancels with 1/2 from Trotterization
                        hx * op("Sx",s1) * op("Id",s2)  +
                        hz * op("Id",s1) * op("Sz",s2)  +
                        hx * op("Id",s1) * op("Sx",s2)

            if j==1
                hj +=      hz * op("Sz",s1) * op("Id",s2)  +
                        hx * op("Sx",s1) * op("Id",s2)
            elseif j==N-1
                hj +=      hz * op("Id",s1) * op("Sz",s2)   +
                        hx * op("Id",s1) * op("Sx",s2)
            end
            push!(hs, hj)
        end
        hs
    end

    function XXX(s::Vector{Index{Int64}};J::Float64=1.0)
        hs = ITensor[]
        N = length(s)
        for j=1:N-1
            s1,s2  = s[j], s[j+1]
            hj =             J * op("Sx",s1) * op("Sx",s2)    +
                            J * op("Sy",s1) * op("Sy",s2)    +
                            J * op("Sz",s1) * op("Sz",s2)
            push!(hs, hj)
        end
        hs
    end

    function MOH(s::Vector{<:Index}; t::Float64=1., t′::Float64=1., V::Float64=1.)::Vector{ITensor}
        hs = ITensor[]
        N = length(s)
        for j=2:N-1
            srange = s[j-1:j+1]
            s1,s2,s3  = srange
            hj = ITensor(0., dag(srange)..., srange'...)

            hj += -t/2 * ( op("c†", s1) * op("c",  s2) * op("Id", s3) +
                        op("c†", s2) * op("c",  s1) * op("Id", s3) +
                        op("Id", s1) * op("c†", s2) * op("c",  s3) +
                        op("Id", s1) * op("c†", s3) * op("c",  s2) )

            hj += -t′  * ( op("c†", s1) * op("F",  s2) * op("c",  s3) +
                        op("c†", s3) * op("F", s2) * op("c",  s1) )

            hj +=  V/2 * ( op("N",  s1) * op("N",  s2) * op("Id", s3) +
                        op("Id", s1) * op("N",  s2) * op("N",  s3) )


            if j == 2
                hj += -t/2 * ( op("c†", s1) * op("c",  s2) * op("Id", s3) +
                            op("c†", s2) * op("c",  s1) * op("Id", s3) )
                hj +=  V/2 * ( op("N",  s1) * op("N",  s2) * op("Id", s3) )
            end

            if j == N-1
                hj += -t/2 * ( op("Id", s1) * op("c†", s2) * op("c",  s3) +
                            op("Id", s1) * op("c†", s3) * op("c",  s2) )
                hj +=  V/2 * ( op("Id", s1) * op("N",  s2) * op("N",  s3) )
            end

            push!(hs, hj)
        end
        hs
    end

    function free_fermion(s::Vector{<:Index}; t::Float64=1.)::Vector{ITensor}
        hs = ITensor[]
        N = length(s)
        for j=1:N-1
            srange= s[j:j+1]
            s1,s2 = srange
            hj = ITensor(0., dag(srange)..., srange'...)

            hj += -t * ( op("c†", s1) * op("c",  s2) +
                         op("c†", s2) * op("c",  s1) )
                         # op("c", s1) * op("c†",  s2) )

            push!(hs, hj)
        end
        hs
    end

    function test(s::Vector{<:Index})::Vector{ITensor}
        hs = ITensor[]
        N = length(s)
        for x=s
            hj = op("N", x)
            push!(hs, hj)
        end
        hs
    end

end
