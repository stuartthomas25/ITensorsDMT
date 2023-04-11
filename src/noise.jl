abstract type Noise end

struct NoNoise <: Noise end
dissipator(dn::NoNoise, s::Site) = []

struct DepolarizingNoise <: Noise
    γ::Float64
end
dissipator(dn::DepolarizingNoise, s::Site) = [sqrt(dn.γ) * op("X",s), sqrt(dn.γ) * op("Y",s), sqrt(dn.γ) * op("Z",s)]

struct DephasingNoise <: Noise
    γ::Float64
end
dissipator(dn::DephasingNoise, s::Site) = [sqrt(dn.γ) * op("Z",s)]

export NoNoise,
    DepolarizingNoise,
    DephasingNoise
