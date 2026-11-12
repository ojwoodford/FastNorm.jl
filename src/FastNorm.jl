module FastNorm

using LoopVectorization, PrecompileTools, StaticArrays

export sqnorm, fast_norm, norm1!!, vecmeannorm, mean0norm1!!

"""
    sqnorm(x::Number)
    sqnorm(arr::AbstractArray)

Computes the squared Euclidean norm (`abs2(x)` for a scalar, or `sum(abs2, arr)` for an array of any dimension).
Real arrays use SIMD vectorization via `LoopVectorization.@turbo`; complex arrays fall back
to `sum(abs2, arr)`.

!!! warning "Safe range"
    No overflow/underflow protection is applied. Element magnitudes should lie roughly within
    `sqrt(floatmin(T))` and `sqrt(floatmax(T))` (about `1e-154` to `1e154` for `Float64`, `1e-19` to
    `1e19` for `Float32`). Above this range squared sums overflow to `Inf`; below it they
    underflow to zero and lose all precision. Use `LinearAlgebra.norm` for data that may be out of range.
"""
@inline sqnorm(x::Number) = abs2(x)
@inline sqnorm(arr::AbstractArray{<:Complex}) = sum(abs2, arr)
@inline function sqnorm(vec::AbstractVector{<:Real})
    total = zero(eltype(vec))
    @turbo for i in eachindex(vec)
        total += vec[i] * vec[i]
    end
    return total
end
# Matrices and higher-dimensional arrays: dense (contiguous) arrays are reduced as a flat vector
@inline sqnorm(arr::DenseArray{<:Real, N}) where N = N == 1 ? invoke(sqnorm, Tuple{AbstractVector{<:Real}}, arr) : sqnorm(vec(arr))
@inline sqnorm(arr::AbstractArray{<:Real}) = sum(abs2, arr)

@inline function sqnorm(vec::StaticVector{N, T}) where {N, T}
    s = zero(T)
    @fastmath @inbounds for i in 1:N
        s += abs2(vec[i])
    end
    return s
end

"""
    fast_norm(arr::AbstractArray)

Computes the Euclidean (Frobenius, for matrices) 2-norm `sqrt(sqnorm(arr))` in native SIMD Julia without calling multi-threaded BLAS.
"""
@inline fast_norm(arr::AbstractArray) = sqrt(sqnorm(arr))

_ismutable(::Type{V}) where V = ismutabletype(V)
# Wrappers that are immutable structs but write through to their parent
_ismutable(::Type{<:SubArray{<:Any, <:Any, P}}) where P = _ismutable(P)
_ismutable(::Type{<:Base.ReshapedArray{<:Any, <:Any, P}}) where P = _ismutable(P)
_ismutable(::Type{<:PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, P}}) where P = _ismutable(P)
_ismutable(::Type{<:SizedArray{<:Any, <:Any, <:Any, <:Any, P}}) where P = _ismutable(P)

"""
    norm1!!(vec::V, minsqnorm::Real) where {T<:AbstractFloat, V<:AbstractVector{T}}
    norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where {T<:AbstractFloat}

Normalize a vector (of `AbstractFloat` elements; integer vectors are not supported) to have unit Euclidean norm (`norm(vec) ≈ 1`).
Uses `sqnorm` to compute normalization factor in a single pass, then scales the elements in 
a second pass.
For mutable vectors (such as `Vector` or views of mutable arrays), the operation is performed
in-place and `vec` is returned. For immutable vectors (such as `StaticArrays.SVector`), a new
normalized vector is returned.

!!! warning "Safe range"
    No overflow/underflow protection is applied. Element magnitudes should lie roughly within
    `sqrt(floatmin(T))` and `sqrt(floatmax(T))` (about `1e-154` to `1e154` for `Float64`, `1e-19` to
    `1e19` for `Float32`). Above this range squared sums overflow to `Inf` (so `norm1!!` returns zeros); below it they
    underflow to zero and are clamped to `minsqnorm`. Use `LinearAlgebra.norm` for data that may be out of range.
"""
@inline function norm1!!(vec::V, minsqnorm::Real) where {T<:AbstractFloat, V<:AbstractVector{T}}
    vecnorm = @fastmath T(1) / sqrt(max(sqnorm(vec), T(minsqnorm)))
    if _ismutable(V)
        @inbounds @simd for i in eachindex(vec)
            vec[i] *= vecnorm
        end
    else
        vec = vec .* vecnorm
    end
    return vec
end
@inline norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where {T<:AbstractFloat} = norm1!!(vec, minsqnorm)

@inline function norm1!!(vec::StaticVector{N, T}, minsqnorm::Real) where {N, T<:AbstractFloat}
    vecnorm = @fastmath T(1) / sqrt(max(sqnorm(vec), T(minsqnorm)))
    if _ismutable(typeof(vec))
        @inbounds @simd for i in 1:N
            vec[i] *= vecnorm
        end
        return vec
    else
        return vec * vecnorm
    end
end
@inline norm1!!(vec::StaticVector{N, T}; minsqnorm::Real=floatmin(T)) where {N, T<:AbstractFloat} = norm1!!(vec, minsqnorm)

"""
    vecmeannorm(vec::AbstractVector{T}, minsqnorm::Real) where {T<:AbstractFloat}
    vecmeannorm(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where {T<:AbstractFloat}

Compute the mean and normalization factor (`1 / norm(vec .- mean)`) of a vector in a single pass.
Uses the algebraic identity:
    sum((x_i - mean)^2) = sum(x_i^2) - mean * sum(x_i)
which requires only one memory pass over `vec`, trading a small amount of numerical accuracy
under extreme cancellation for reduced memory traffic and execution speed.
The squared norm is floored at `max(minsqnorm, length(vec) * eps(T) * sum(vec .^ 2))`, i.e. at the
level of rounding noise in the identity above, which prevents division by zero, negative square
roots, and garbage output for constant or near-constant vectors (these normalize to ~zero rather
than to noise amplified by `1 / sqrt(floatmin(T))`).
Accuracy degrades as `std(vec) / abs(mean(vec))` approaches `sqrt(eps(T))`; for such low-contrast
data, subtract a representative value (e.g. `vec[1]`) first or use a two-pass method.
Returns `(vecmean, vecnorm)`. An empty vector gives `(zero(T), one(T))`.

!!! warning "Safe range"
    No overflow/underflow protection is applied. Element magnitudes should lie roughly within
    `sqrt(floatmin(T))` and `sqrt(floatmax(T))` (about `1e-154` to `1e154` for `Float64`, `1e-19` to
    `1e19` for `Float32`). Above this range squared sums overflow to `Inf` (so the returned normalization factor is zero); below it they
    underflow to zero and are clamped to the noise floor. Use `LinearAlgebra.norm` for data that may be out of range.
"""
@inline function vecmeannorm(vec::AbstractVector{T}, minsqnorm::Real) where {T<:AbstractFloat}
    isempty(vec) && return (zero(T), one(T))
    x = zero(T)
    xx = zero(T)
    @turbo for i in eachindex(vec)
        v = vec[i]
        x  += v
        xx += v * v
    end
    vecmean = x / length(vec)
    vecnorm = @fastmath T(1) / sqrt(max(xx - vecmean * x, T(minsqnorm), length(vec) * eps(T) * xx))
    return (vecmean, vecnorm)
end
@inline vecmeannorm(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where {T<:AbstractFloat} = vecmeannorm(vec, minsqnorm)

@inline function vecmeannorm(vec::StaticVector{N, T}, minsqnorm::Real) where {N, T<:AbstractFloat}
    N == 0 && return (zero(T), one(T))
    x = zero(T)
    xx = zero(T)
    @fastmath @inbounds for i in 1:N
        v = vec[i]
        x  += v
        xx += v * v
    end
    vecmean = x / N
    vecnorm = @fastmath T(1) / sqrt(max(xx - vecmean * x, T(minsqnorm), N * eps(T) * xx))
    return (vecmean, vecnorm)
end
@inline vecmeannorm(vec::StaticVector{N, T}; minsqnorm::Real=floatmin(T)) where {N, T<:AbstractFloat} = vecmeannorm(vec, minsqnorm)

"""
    mean0norm1!!(vec::V, minsqnorm::Real) where {T<:AbstractFloat, V<:AbstractVector{T}}
    mean0norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where {T<:AbstractFloat}

Normalize a vector (of `AbstractFloat` elements; integer vectors are not supported) to have zero mean and unit Euclidean norm (`norm(vec) ≈ 1`).
Uses `vecmeannorm` to compute the mean and normalization factor in a single pass, then
scales the elements in a second pass.
For mutable vectors (such as `Vector` or views of mutable arrays), the operation is performed
in-place and `vec` is returned. For immutable vectors (such as `StaticArrays.SVector`), a new
normalized vector is returned.

!!! warning "Safe range"
    No overflow/underflow protection is applied. Element magnitudes should lie roughly within
    `sqrt(floatmin(T))` and `sqrt(floatmax(T))` (about `1e-154` to `1e154` for `Float64`, `1e-19` to
    `1e19` for `Float32`). Above this range squared sums overflow to `Inf` (so `mean0norm1!!` returns zeros); below it they
    underflow to zero and are clamped to the noise floor. Use `LinearAlgebra.norm` for data that may be out of range.
"""
@inline function mean0norm1!!(vec::V, minsqnorm::Real) where {T<:AbstractFloat, V<:AbstractVector{T}}
    vecmean, vecnorm = vecmeannorm(vec, minsqnorm)
    if _ismutable(V)
        @inbounds @simd for i in eachindex(vec)
            vec[i] = (vec[i] - vecmean) * vecnorm
        end
    else
        vec = (vec .- vecmean) .* vecnorm
    end
    return vec
end
@inline mean0norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where {T<:AbstractFloat} = mean0norm1!!(vec, minsqnorm)

@inline function mean0norm1!!(vec::StaticVector{N, T}, minsqnorm::Real) where {N, T<:AbstractFloat}
    vecmean, vecnorm = vecmeannorm(vec, minsqnorm)
    if _ismutable(typeof(vec))
        @inbounds @simd for i in 1:N
            vec[i] = (vec[i] - vecmean) * vecnorm
        end
        return vec
    else
        return (vec .- vecmean) * vecnorm
    end
end
@inline mean0norm1!!(vec::StaticVector{N, T}; minsqnorm::Real=floatmin(T)) where {N, T<:AbstractFloat} = mean0norm1!!(vec, minsqnorm)

include("precompile.jl")

end # module FastNorm
