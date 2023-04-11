function Θ!(ρ::MPS)
    A = diagm([1,1,-1,1])
    μ = siteinds(ρ)
    for i in eachindex(ρ)
        U = ITensor(A, μ[i], μ[i]')
        ρ[i] *= U
        ITensors.replaceprime!(ρ[i], 1, 0)
        ρ[i] .= conj(ρ[i])
    end
    ρ
end

"""Take the inner product, but offset second MPSs but a certain number of sites"""
function offset_dot(M1::MPS, M2::MPS, Δx::Int)
    Δx<0 && return conj(offset_dot(M2, M1, -Δx))

    s = siteinds(M1)
    M1dag = mapprime(dag(M1), 0, 1; tags="Link")
    O = one(ITensor)
    for j in 1:Δx
        O *= M1dag[j] * state(s[j], 1)
    end
    for j in eachindex(M1)[Δx+1:end]
        shiftedM2T = replaceinds(M2[j-Δx], [s[j-Δx]], [s[j]])
        O = (O * M1dag[j]) * shiftedM2T
    end

    for j in eachindex(M2)[end-Δx+1:end]
        O *= M2[j] * state(s[j], 1)
    end

    O[]
end

function h2t(ρ::MPS)
    h = ρ
    Θh = Θ!(deepcopy(ρ))
    N = length(ρ)

    map((-N+1):(N-1)) do Δx
        offset_dot(h, Θh, Δx)
    end
end

export h2t
