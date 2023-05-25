

module TA

    struct TrotterSweep
        τ::Float64
        step::Int64
        offset::Int64
    end

    const ITensor = [TrotterSweep(0.5, 1, 0), TrotterSweep(0.5, -1, 0)]
    const Order1 = [TrotterSweep(1., 2, 0), TrotterSweep(1., -2, 1)]
    const Order2 = [TrotterSweep(0.5, 2, 1), TrotterSweep(1., -2, 0), TrotterSweep(0.5, 2, 1)]
    const Order4 = begin # Bartel & Zhang, Annals of Physics 418, 168165 (2020) Eq. 32
        τ2 = 1/(4-4^(1/3))
        τ2b = 1/(4-4^(1/3))/2
        τ3 = (1-4/(4-4^(1/3)))
        τ4 = (1-3/(4-4^(1/3)))/2
        TS = TrotterSweep
        [
            TS(τ2b, 2, 1),
            TS(τ2, -2, 0),
            TS(τ2,  2, 1),
            TS(τ2, -2, 0),
            TS(τ4,  2, 1),
            TS(τ3, -2, 0),
            TS(τ4,  2, 1),
            TS(τ2, -2, 0),
            TS(τ2,  2, 1),
            TS(τ2, -2, 0),
            TS(τ2b, 2, 1)
        ]
    end

    const Order1n3 = [TrotterSweep(1., 2, 0), TrotterSweep(1., -2, 1)]
    const Order4n3 = begin # Bartel & Zhang, Annals of Physics 418, 168165 (2020) Eq. 53
        u = BigFloat("0.095968145884398107402")
        q1 = BigFloat("0.43046123580897338276")
        r1 = BigFloat("-0.075403897922216340661")
        q2 = BigFloat("-0.12443549678124729963")
        r2 = BigFloat("0.5") - (u + r1)
        u1 = u
        v1 = q1 - u1
        u2 = r1 - v1
        v2 = q2 - u2
        u3 = r2 - v2 # = 1/2 -q2 - q1

        TS = TrotterSweep
        [
            TS(u,    3, 0),
            TS(u,   -3, 1),
            TS(q1  , 3, 2),
            TS(v1  ,-3, 1),
            TS(r1  , 3, 0),
            TS(u2  ,-3, 1),
            TS(q2  , 3, 2),
            TS(v2  ,-3, 1),
            TS(r2  , 3, 0),
            TS(u3  ,-3, 1),
            TS(2*u3, 3, 2),
            TS(u3  ,-3, 1),
            TS(r2  , 3, 0),
            TS(v2  ,-3, 1),
            TS(q2  , 3, 2),
            TS(u2  ,-3, 1),
            TS(r1  , 3, 0),
            TS(v1  ,-3, 1),
            TS(q1  , 3, 2),
            TS(u,   -3, 1),
            TS(u,    3, 0)
        ]
    end
end

# """
# optimized better fourth order from Barthel
# https://arxiv.org/pdf/1901.04974.pdf
# also at
# https://doi.org/10.1016/j.aop.2020.168165
# aka
# https://www.sciencedirect.com/science/article/abs/pii/S0003491620300981?via%3Dihub
# recommended by Barthel for 4th order with 3 terms
# Optimized (m = 21, type SE, ν = 5)
# Steps take the form:
# (u1 ABC)(v1 CBA)(u2 ABC)(v2 CBA)(u3 ABC)(u3 CBA)(v2 ABC)(u2 CBA)(v1 ABC)(u1 CBA)
# which simplifies to
# (u1 A)(u1 B)(u1+v1 C)(v1 B)(v1+u2 A)(u2 B)(u2+v2 C)(v2 B)(v2+u3 A)(u3 B)
# (2u3 C)(u3 B)(u3+v2 A)(v2 B)(v2+u2 C)(u2 B)(u2+v1 A)(v1 B)(v1+u1 C)(u1 B)(u1 A)
# """
# function fourth_order_trotter_threeterm() where T
#     u = BigFloat("0.095968145884398107402")
#     q1 = BigFloat("0.43046123580897338276")
#     r1 = BigFloat("-0.075403897922216340661")
#     q2 = BigFloat("-0.12443549678124729963")
#     r2 = BigFloat("0.5") - (u + r1)
#     u1 = u
#     v1 = q1 - u1
#     u2 = r1 - v1
#     v2 = q2 - u2
#     u3 = r2 - v2 # = 1/2 -q2 - q1
#     # steps = [u1, u1+v1, v1, v1+u2, u2, u2+v2, v2, v2+u3, u3, 2*u3]
#     stepsizes = Float64[u, q1, v1, r1, u2, q2, v2, r2, u3, 2*u3]
#     steppattern = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
#     termpattern = [:A, :B, :C, :B, :A, :B, :C, :B, :A, :B, :C, :B, :A, :B, :C, :B, :A, :B, :C, :B, :A]
#     NS = length(termpattern)
#     @assert NS == length(steppattern)
#     @assert sum(stepsizes[steppattern[termpattern .== :A]]) ≈ 1
#     @assert sum(stepsizes[steppattern[termpattern .== :B]]) ≈ 1
#     @assert sum(stepsizes[steppattern[termpattern .== :C]]) ≈ 1
#     return [(termpattern[i], stepsizes[steppattern[i]]) for i in 1:NS]
# end

const TrotterAlgorithm = Vector{TA.TrotterSweep}

export TA, TrotterAlgorithm, fourth_order_trotter_threeterm
