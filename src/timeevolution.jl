
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
    l = sort([inds(h, plev=0)...], by=sitepos)
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

function _check_form(M::AbstractArray)
    full_error = "The Wᴵ and Wᴵᴵ methods require the Hamiltonian MPO to be in a specific gauge form. The Hamiltonian provided does not meet this requirement."
    M[1,1,:,:] ≈ I || throw("$full_error ($(M[1,1,:,:]) is not identity (1,1))")
    for col in 2:size(M)[2]
        M[1,col,:,:] ≈ 0 * I || throw("$full_error ($(M[1,col,:,:]) is not zero (1,$col))")
    end

    M[end,end,:,:] ≈ I || throw("$full_error ($(M[end,end,:,:]) is not identity (N,N))")
    for row in 1:(size(M)[1]-1)
        M[row,end,:,:] ≈ 0 * I || throw("$full_error ($(M[row,end,:,:]) is not zero ($row, N))")
    end
    return true
end

function _clear_dummy_indices(T::ITensor)
    for i in filter(hastags("Dummy"), inds(T))
        T *= onehot(i=>1)
    end
    T
end

""" multiply MPO by scalar without changing the block triangular canonical form"""
function triangular_scalar_multiply(H::MPO, λ::Number)::MPO
    map(H) do T
        if T===H[1]
            rl = filter(hastags("Link"), inds(T)) |> only
            ll = Index(1, "Link,Dummy")
            T *= onehot(ll=>1)
        elseif T===H[end]
            ll = filter(hastags("Link"), inds(T)) |> only
            rl = Index(1, "Link,Dummy")
            T *= onehot(rl=>1)
        else
            ll, rl = filter(hastags("Link"), inds(T))
        end
        L = δ(ll, ll') |> dense
        # if dim(ll)>1
        L[end, end] *= λ
        # end
        R = δ(rl, rl') |> dense
        if dim(rl)>1
            R[end, end] *= λ
        end
        newT = (L × T × R) |> _clear_dummy_indices
        newT
    end
end
function _Wᴵ_mat(δt::Float64, A::T, B::T, C::T, D::T)::T where {T<:AbstractArray}
    d = D |> length |> isqrt
    Id = reshape(Matrix(I, d, d), 1,1,d,d)
    [Id + δt * D;    √(δt) * B;;
       √(δt) * C;            A ]
end


function _Wᴵᴵ_mat(δt::Float64, A::T, B::T, C::T, D::T)::T where {T<:AbstractArray}
    d = D |> length |> isqrt
    WA = zero(A)
    WB = zero(B)
    WC = zero(C)
    WD = zero(D)
    for (i,j) in Iterators.product(1:size(B)[1], 1:size(C)[2])
        Aa = A[i, j, :, :]
        Ba = B[i, 1, :, :]
        Ca = C[1, j, :, :]
        Da = D[1, 1, :, :]
        ∅ = fill(0., d, d)
        F = [
            δt * Da    ; ∅          ; ∅          ; ∅    ;;
            √(δt) * Ca ; δt * Da    ; ∅          ; ∅    ;;
            √(δt) * Ba ; ∅          ; δt * Da    ; ∅    ;;
            Aa         ; √(δt) * Ba ; √(δt) * Ca ; δt * Da
        ]
        expF = exp(F)

        WA[i,j,:,:] = expF[(3d+1):4d , 1:d]
        WB[i,1,:,:] = expF[(2d+1):3d , 1:d]
        WC[1,j,:,:] = expF[(d+1):2d  , 1:d]
        WD[1,1,:,:] = expF[1:d       , 1:d]
    end

    return  [WD; WB;;
             WC; WA ]
end

Wᴵ(H::MPO, δt::Float64)::MPO = _Wᴵ_Wᴵᴵ(H, (mats...) -> _Wᴵ_mat(δt, mats...))
Wᴵᴵ(H::MPO, δt::Float64)::MPO = _Wᴵ_Wᴵᴵ(H, (mats...) -> _Wᴵᴵ_mat(δt, mats...))


function _Wᴵ_Wᴵᴵ(H::MPO, W_mat_f)::MPO
    N = length(H)
    prev_link = Index(2, "Dummy,Link")
    s = siteinds(first, H; plev=0)
    Wᴵs = map(eachindex(H)) do i
        T = H[i]
        # add an extra dim to include zero-values A and B matrices. I.e. put it in triangular canonical form.
        if i==1
            rl = linkinds(H)[1]
            ll = Index(3, "Dummy,Link")
            T *= onehot(ll=>dim(ll))
            T += onehot(ll=>1) * onehot(rl=>1) * δ(s[i], s[i]')
        elseif i == N
            ll = linkinds(H)[end]
            rl = Index(3, "Dummy,Link")
            T *= onehot(rl=>1)
            T += onehot(ll=>dim(ll)) * onehot(rl=>dim(rl)) * δ(s[i], s[i]')
        else
            ll, rl = linkinds(H)[(i-1):i]
        end

        x = s[i]
        M = Array(T, ll, rl, x, x')
        _check_form(M)

        A = M[2:end-1, 2:end-1, :, :]
        B = M[2:end-1, 1:1,   :, :]
        C = M[end:end, 2:end-1,   :, :]
        D = M[end:end, 1:1,     :, :]

        Wᴵ_m = W_mat_f(A, B, C, D)

        next_link = Index(size(Wᴵ_m)[2], tags="Link,l=$i")
        Wᴵ = ITensor(Wᴵ_m, prev_link, next_link, x, x')

        if i==1
            Wᴵ *= onehot(prev_link=>1)
        elseif i == N
            Wᴵ *= onehot(next_link=>1)
        end

        prev_link = next_link
        Wᴵ
    end
    MPO(Wᴵs, 1, length(H))
end

export
    gate,
    make_sweep,
    Wᴵ,
    Wᴵᴵ
