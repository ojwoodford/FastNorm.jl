module FastNorm

using LoopVectorization, PrecompileTools, StaticArrays

export sqnorm, fast_norm, norm1!!, vecmeannorm, mean0norm1!!

"""
    sqnorm(x::Number)
    sqnorm(vec::AbstractVector)

Computes the squared Euclidean norm (`x * x` for a scalar, or `sum(abs2, vec)` for a vector)
using SIMD vectorization via `LoopVectorization.@turbo`.
"""
@inline sqnorm(x::Number) = @fastmath x * x
@inline function sqnorm(vec::AbstractVector)
    total = zero(eltype(vec))
    @turbo for i in eachindex(vec)
        total += vec[i] * vec[i]
    end
    return total
end

@inline function sqnorm(vec::StaticVector{N, T}) where {N, T}
    s = zero(T)
    @fastmath @inbounds for i in 1:N
        s += vec[i] * vec[i]
    end
    return s
end

"""
    fast_norm(vec::AbstractVector)

Computes the Euclidean 2-norm `sqrt(sqnorm(vec))` in native SIMD Julia without calling multi-threaded BLAS.
"""
@inline fast_norm(vec::AbstractVector) = sqrt(sqnorm(vec))

_ismutable(::Type{V}) where V = ismutabletype(V)
_ismutable(::Type{<:SubArray{<:Any, <:Any, P}}) where P = _ismutable(P)

"""
    norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T
    norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T

Normalize a vector to have unit Euclidean norm (`norm(vec) ≈ 1`).
Uses `sqnorm` to compute normalization factor in a single pass, then scales the elements in 
a second pass.
For mutable vectors (such as `Vector` or views of mutable arrays), the operation is performed
in-place and `vec` is returned. For immutable vectors (such as `StaticArrays.SVector`), a new
normalized vector is returned.
"""
@inline function norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T
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
@inline norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T = norm1!!(vec, minsqnorm)

@inline function norm1!!(vec::StaticVector{N, T}, minsqnorm::Real) where {N, T}
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
@inline norm1!!(vec::StaticVector{N, T}; minsqnorm::Real=floatmin(T)) where {N, T} = norm1!!(vec, minsqnorm)

"""
    vecmeannorm(vec::AbstractVector{T}, minsqnorm::Real) where T
    vecmeannorm(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T

Compute the mean and normalization factor (`1 / norm(vec .- mean)`) of a vector in a single pass.
Uses the algebraic identity:
    sum((x_i - mean)^2) = sum(x_i^2) - mean * sum(x_i)
which requires only one memory pass over `vec`, trading a small amount of numerical accuracy
under extreme cancellation for reduced memory traffic and execution speed.
Clamps the squared norm to `minsqnorm` to prevent division by zero or negative square roots.
Returns `(vecmean, vecnorm)`.
"""
@inline function vecmeannorm(vec::AbstractVector{T}, minsqnorm::Real) where T
    x = zero(T)
    xx = zero(T)
    @turbo for i in eachindex(vec)
        v = vec[i]
        x  += v
        xx += v * v
    end
    vecmean = x / length(vec)
    vecnorm = @fastmath T(1) / sqrt(max(xx - vecmean * x, T(minsqnorm)))
    return (vecmean, vecnorm)
end
@inline vecmeannorm(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T = vecmeannorm(vec, minsqnorm)

@inline function vecmeannorm(vec::StaticVector{N, T}, minsqnorm::Real) where {N, T}
    x = zero(T)
    xx = zero(T)
    @fastmath @inbounds for i in 1:N
        v = vec[i]
        x  += v
        xx += v * v
    end
    vecmean = x / N
    vecnorm = @fastmath T(1) / sqrt(max(xx - vecmean * x, T(minsqnorm)))
    return (vecmean, vecnorm)
end
@inline vecmeannorm(vec::StaticVector{N, T}; minsqnorm::Real=floatmin(T)) where {N, T} = vecmeannorm(vec, minsqnorm)

"""
    mean0norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T
    mean0norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T

Normalize a vector to have zero mean and unit Euclidean norm (`norm(vec) ≈ 1`).
Uses `vecmeannorm` to compute the mean and normalization factor in a single pass, then
scales the elements in a second pass.
For mutable vectors (such as `Vector` or views of mutable arrays), the operation is performed
in-place and `vec` is returned. For immutable vectors (such as `StaticArrays.SVector`), a new
normalized vector is returned.
"""
@inline function mean0norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T
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
@inline mean0norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T = mean0norm1!!(vec, minsqnorm)

@inline function mean0norm1!!(vec::StaticVector{N, T}, minsqnorm::Real) where {N, T}
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
@inline mean0norm1!!(vec::StaticVector{N, T}; minsqnorm::Real=floatmin(T)) where {N, T} = mean0norm1!!(vec, minsqnorm)

include("precompile.jl")

end # module FastNorm
