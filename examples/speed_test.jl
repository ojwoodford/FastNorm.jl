# Speed test comparing FastNorm primitives to standard Julia / LinearAlgebra implementations
using FastNorm
using LinearAlgebra
using Random
using StaticArrays

"""
    bench_call(f::F, iters::Int, trials::Int=5) where F

Time execution of `f()` across `trials` independent runs of `iters` iterations each,
returning the minimum time per call in nanoseconds.
"""
@noinline function bench_call(f::F, iters::Int, trials::Int=5) where F
    # Warmup
    Base.inferencebarrier(f())
    best = Inf
    for _ in 1:trials
        t0 = time_ns()
        for _ in 1:iters
            Base.inferencebarrier(f())
        end
        t = (time_ns() - t0) / iters
        best = min(best, t)
    end
    return best
end

@noinline read_elem(arr) = arr[1]

function format_time(t_ns::Float64)
    if t_ns < 1000
        return string(round(t_ns, digits=2), " ns")
    elseif t_ns < 1_000_000
        return string(round(t_ns / 1000, digits=2), " μs")
    else
        return string(round(t_ns / 1_000_000, digits=2), " ms")
    end
end

function standard_vecmeannorm(vec::AbstractVector{T}) where T
    m = sum(vec) / length(vec)
    ss = zero(T)
    @simd for v in vec
        ss += (v - m)^2
    end
    n = T(1) / sqrt(max(ss, floatmin(T)))
    return (m, n)
end

function standard_mean0norm1!(vec::AbstractVector{T}) where T
    m = sum(vec) / length(vec)
    vec .-= m
    n = norm(vec)
    vec ./= max(n, floatmin(T))
    return vec
end

function standard_svecmeannorm(s::SVector{N, T}) where {N, T}
    m = sum(s) / N
    ss = sum((s .- m).^2)
    n = T(1) / sqrt(max(ss, floatmin(T)))
    return (m, n)
end

function standard_smean0norm1(s::SVector{N, T}) where {N, T}
    m = sum(s) / N
    centered = s .- m
    n = norm(centered)
    return centered ./ max(n, floatmin(T))
end

"""
    run_speed_test(trials=5)

Run comprehensive benchmarks comparing all FastNorm functions against standard implementations
and print Markdown-formatted performance summary tables.
"""
function run_speed_test(trials::Int=5)
    Random.seed!(42)

    println("=" ^ 88)
    println("FastNorm.jl Speed Test: FastNorm vs Standard Julia / LinearAlgebra")
    println("Trials per test: $trials (taking minimum time per call)")
    println("=" ^ 88)

    # =========================================================================
    # Part 1: Standard Vectors (Vector{Float64})
    # =========================================================================
    println("\n## Standard Dense Vectors (`Vector{Float64}`)\n")
    sizes = [4, 16, 64, 256, 1024, 4096]

    # 1. Euclidean Norm: fast_norm vs LinearAlgebra.norm
    println("### 1. Euclidean Norm: `fast_norm` vs `LinearAlgebra.norm`\n")
    println("| Vector Size | Standard (`LinearAlgebra.norm`) | FastNorm (`fast_norm`) | Speedup | Time Reduction |")
    println("|:-----------:|:-------------------------------:|:----------------------:|:-------:|:--------------:|")

    for n in sizes
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        # Verification
        @assert fast_norm(a) ≈ norm(a)

        t_std = bench_call(() -> norm(a), iters, trials)
        t_fast = bench_call(() -> fast_norm(a), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 2. Squared Euclidean Norm: sqnorm vs LinearAlgebra.dot(a, a)
    println("\n### 2. Squared Euclidean Norm: `sqnorm` vs `LinearAlgebra.dot(a, a)`\n")
    println("| Vector Size | Standard (`dot(a, a)`) | FastNorm (`sqnorm`) | Speedup | Time Reduction |")
    println("|:-----------:|:----------------------:|:-------------------:|:-------:|:--------------:|")

    for n in sizes
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        # Verification
        @assert sqnorm(a) ≈ dot(a, a)

        t_std = bench_call(() -> dot(a, a), iters, trials)
        t_fast = bench_call(() -> sqnorm(a), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 3. Unit-Norm Normalization: norm1!! vs LinearAlgebra.normalize!
    println("\n### 3. Unit-Norm Normalization: `norm1!!` vs `LinearAlgebra.normalize!`\n")
    println("| Vector Size | Standard (`normalize!`) | FastNorm (`norm1!!`) | Speedup | Time Reduction |")
    println("|:-----------:|:-----------------------:|:--------------------:|:-------:|:--------------:|")

    for n in sizes
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        buf_std = copy(a)
        buf_fast = copy(a)

        # Verification
        normalize!(buf_std)
        norm1!!(buf_fast)
        @assert isapprox(buf_fast, buf_std, atol=1e-10)

        t_std = bench_call(() -> normalize!(buf_std), iters, trials)
        t_fast = bench_call(() -> norm1!!(buf_fast), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 4. Mean and Normalizer: vecmeannorm vs Standard 2-Pass
    println("\n### 4. Mean and Normalizer: `vecmeannorm` vs Standard 2-Pass\n")
    println("| Vector Size | Standard (2-pass) | FastNorm (`vecmeannorm`) | Speedup | Time Reduction |")
    println("|:-----------:|:-----------------:|:------------------------:|:-------:|:--------------:|")

    for n in sizes
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        # Verification
        m_fast, inv_norm_fast = vecmeannorm(a)
        m_std, inv_norm_std = standard_vecmeannorm(a)
        @assert m_fast ≈ m_std
        @assert inv_norm_fast ≈ inv_norm_std

        t_std = bench_call(() -> standard_vecmeannorm(a), iters, trials)
        t_fast = bench_call(() -> vecmeannorm(a), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 5. In-Place Zero-Mean Unit-Norm Normalization: mean0norm1!! vs Standard
    println("\n### 5. Zero-Mean Unit-Norm: `mean0norm1!!` vs Standard Multi-Pass\n")
    println("| Vector Size | Standard (Multi-pass) | FastNorm (`mean0norm1!!`) | Speedup | Time Reduction |")
    println("|:-----------:|:---------------------:|:---------------------------:|:-------:|:--------------:|")

    for n in sizes
        a = randn(n)
        iters = max(1000, 500_000 ÷ n)

        buf_std = copy(a)
        buf_fast = copy(a)

        # Verification
        standard_mean0norm1!(buf_std)
        mean0norm1!!(buf_fast)
        @assert isapprox(buf_fast, buf_std, atol=1e-10)

        t_std = bench_call(() -> standard_mean0norm1!(buf_std), iters, trials)
        t_fast = bench_call(() -> mean0norm1!!(buf_fast), iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # =========================================================================
    # Part 2: Static Vectors (SVector{N, Float64} up to length 32)
    # =========================================================================
    println("\n## Static Vectors (`SVector{N, Float64}` up to length 32)\n")
    svec_sizes = [2, 3, 4, 8, 16, 32]
    svec_iters = 200_000

    # 1. SVector Euclidean Norm: fast_norm vs LinearAlgebra.norm
    println("### 1. SVector Euclidean Norm: `fast_norm` vs `LinearAlgebra.norm`\n")
    println("| Vector Size | Standard (`LinearAlgebra.norm`) | FastNorm (`fast_norm`) | Speedup | Time Reduction |")
    println("|:-----------:|:-------------------------------:|:----------------------:|:-------:|:--------------:|")

    for n in svec_sizes
        s = SVector{n, Float64}(randn(n)...)
        box = [s]

        # Verification
        @assert fast_norm(s) ≈ norm(s)

        t_std = bench_call(() -> norm(read_elem(box)), svec_iters, trials)
        t_fast = bench_call(() -> fast_norm(read_elem(box)), svec_iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 2. SVector Squared Euclidean Norm: sqnorm vs LinearAlgebra.dot
    println("\n### 2. SVector Squared Norm: `sqnorm` vs `LinearAlgebra.dot(s, s)`\n")
    println("| Vector Size | Standard (`dot(s, s)`) | FastNorm (`sqnorm`) | Speedup | Time Reduction |")
    println("|:-----------:|:----------------------:|:-------------------:|:-------:|:--------------:|")

    for n in svec_sizes
        s = SVector{n, Float64}(randn(n)...)
        box = [s]

        # Verification
        @assert sqnorm(s) ≈ dot(s, s)

        t_std = bench_call(() -> dot(read_elem(box), read_elem(box)), svec_iters, trials)
        t_fast = bench_call(() -> sqnorm(read_elem(box)), svec_iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 3. SVector Unit-Norm Normalization: norm1!! vs LinearAlgebra.normalize
    println("\n### 3. SVector Unit-Norm Normalization: `norm1!!` vs `LinearAlgebra.normalize`\n")
    println("| Vector Size | Standard (`LinearAlgebra.normalize`) | FastNorm (`norm1!!`) | Speedup | Time Reduction |")
    println("|:-----------:|:------------------------------------:|:--------------------:|:-------:|:--------------:|")

    for n in svec_sizes
        s = SVector{n, Float64}(randn(n)...)
        box = [s]

        # Verification
        @assert norm1!!(s) ≈ normalize(s)

        t_std = bench_call(() -> normalize(read_elem(box)), svec_iters, trials)
        t_fast = bench_call(() -> norm1!!(read_elem(box)), svec_iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 4. SVector Mean and Normalizer: vecmeannorm vs Standard 2-Pass
    println("\n### 4. SVector Mean and Normalizer: `vecmeannorm` vs Standard 2-Pass\n")
    println("| Vector Size | Standard (2-pass) | FastNorm (`vecmeannorm`) | Speedup | Time Reduction |")
    println("|:-----------:|:-----------------:|:------------------------:|:-------:|:--------------:|")

    for n in svec_sizes
        s = SVector{n, Float64}(randn(n)...)
        box = [s]

        # Verification
        m_fast, inv_norm_fast = vecmeannorm(s)
        m_std, inv_norm_std = standard_svecmeannorm(s)
        @assert m_fast ≈ m_std
        @assert inv_norm_fast ≈ inv_norm_std

        t_std = bench_call(() -> standard_svecmeannorm(read_elem(box)), svec_iters, trials)
        t_fast = bench_call(() -> vecmeannorm(read_elem(box)), svec_iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    # 5. SVector Zero-Mean Unit-Norm Normalization: mean0norm1!! vs Standard Multi-Pass
    println("\n### 5. SVector Zero-Mean Unit-Norm: `mean0norm1!!` vs Standard Multi-Pass\n")
    println("| Vector Size | Standard (Multi-pass) | FastNorm (`mean0norm1!!`) | Speedup | Time Reduction |")
    println("|:-----------:|:---------------------:|:---------------------------:|:-------:|:--------------:|")

    for n in svec_sizes
        s = SVector{n, Float64}(randn(n)...)
        box = [s]

        # Verification
        @assert mean0norm1!!(s) ≈ standard_smean0norm1(s)

        t_std = bench_call(() -> standard_smean0norm1(read_elem(box)), svec_iters, trials)
        t_fast = bench_call(() -> mean0norm1!!(read_elem(box)), svec_iters, trials)

        speedup = t_std / t_fast
        reduction = (1.0 - t_fast / t_std) * 100.0
        println("| $n | $(format_time(t_std)) | $(format_time(t_fast)) | $(round(speedup, digits=1))x | $(round(reduction, digits=1))% |")
    end

    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_speed_test()
end
