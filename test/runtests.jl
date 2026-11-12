using FastNorm, LinearAlgebra, StaticArrays, Test

@testset "FastNorm.jl" begin

    @testset "sqnorm" begin
        # Scalars
        @test sqnorm(-3.0) == 9.0
        @test sqnorm(-3.0f0) == 9.0f0
        @test sqnorm(4) == 16
        @test sqnorm(0.0) == 0.0
        @test sqnorm(0.0f0) == 0.0f0

        # Complex: |x|^2, not x*x
        @test sqnorm(1im) == 1
        @test sqnorm(3.0 + 4.0im) == 25.0
        @test sqnorm([1.0im, 3.0 + 4.0im]) == 26.0
        @test fast_norm([3.0 + 4.0im]) == 5.0
        @test sqnorm(SVector(3.0 + 4.0im, 1.0im)) == 26.0

        # Matrices and higher-dimensional arrays
        M = randn(7, 9)
        @test sqnorm(M) ≈ sum(abs2, M)
        @test sqnorm(view(M, 2:5, 3:8)) ≈ sum(abs2, view(M, 2:5, 3:8))
        @test sqnorm(randn(Float32, 3, 4, 5)) isa Float32
        @test fast_norm(M) ≈ norm(M)
        @test sqnorm(randn(ComplexF64, 4, 4)) isa Float64
        @test sqnorm(reshape(1:12, 3, 4)) == sum(abs2, 1:12)

        # Float64 Vectors
        a64 = randn(1001)
        @test sqnorm(a64) ≈ dot(a64, a64)
        @test sqnorm(view(a64, 1:500)) ≈ dot(view(a64, 1:500), view(a64, 1:500))
        @test sqnorm(Float64[]) == 0.0
        @test sqnorm(SVector(3.0, 4.0)) == 25.0

        # Float32 Vectors
        a32 = randn(Float32, 1001)
        @test sqnorm(a32) ≈ dot(a32, a32)
        @test sqnorm(view(a32, 1:500)) ≈ dot(view(a32, 1:500), view(a32, 1:500))
        @test sqnorm(Float32[]) == 0.0f0
        @test sqnorm(SVector(3.0f0, 4.0f0)) == 25.0f0
    end

    @testset "fast_norm" begin
        # Float64
        a64 = randn(1001)
        @test fast_norm(a64) ≈ norm(a64)
        @test fast_norm(view(a64, 1:500)) ≈ norm(view(a64, 1:500))
        @test fast_norm(Float64[]) == 0.0
        @test fast_norm(SVector(3.0, 4.0)) == 5.0

        # Float32
        a32 = randn(Float32, 1001)
        @test fast_norm(a32) ≈ norm(a32)
        @test fast_norm(view(a32, 1:500)) ≈ norm(view(a32, 1:500))
        @test fast_norm(Float32[]) == 0.0f0
        @test fast_norm(SVector(3.0f0, 4.0f0)) == 5.0f0
    end

    @testset "norm1!!" begin
        for T in (Float64, Float32)
            tol = T == Float64 ? 1e-11 : 1e-4

            # Mutable dense vectors (lengths 16, 1024)
            for n in (16, 1024)
                A = randn(T, n)
                expected = A ./ norm(A)

                # In-place verification
                A_copy = copy(A)
                res = norm1!!(A_copy)
                @test res === A_copy  # Mutated in-place
                @test isapprox(res, expected, atol=tol, rtol=tol)
                @test isapprox(norm(res), one(T), atol=tol)
            end

            # SubArray (view into mutable array)
            parent_arr = randn(T, 64)
            v_view = view(parent_arr, 1:32)
            expected_view = v_view ./ norm(v_view)
            norm1!!(v_view)
            @test isapprox(v_view, expected_view, atol=tol, rtol=tol)
            @test isapprox(norm(v_view), one(T), atol=tol)

            # Static vectors (immutable SVector)
            s_data = randn(T, 16)
            B = SVector{16, T}(s_data)
            expected_s = SVector{16, T}(s_data ./ norm(s_data))
            B_norm = norm1!!(B)
            @test B_norm isa SVector{16, T}
            @test isapprox(B_norm, expected_s, atol=tol, rtol=tol)
            @test isapprox(norm(B_norm), one(T), atol=tol)

            # Zero vector: tests default floatmin(T) clamping prevents NaN/Inf
            z = zeros(T, 16)
            z_norm = norm1!!(copy(z))
            @test all(z_norm .== zero(T))
            @test !any(isnan, z_norm)

            # Custom minsqnorm keyword and positional arguments
            c = randn(T, 16)
            c_kw = norm1!!(copy(c); minsqnorm=1e-6)
            c_pos = norm1!!(copy(c), 1e-6)
            @test isapprox(c_kw, c_pos, atol=tol)
            @test isapprox(norm(c_kw), one(T), atol=tol)

            # Immutable AbstractVector (non-StaticVector)
            r = T(1):T(1):T(16)
            expected_r = r ./ norm(r)
            res_r = norm1!!(r)
            @test isapprox(res_r, expected_r, atol=tol, rtol=tol)
            @test isapprox(norm(res_r), one(T), atol=tol)
            res_r_kw = norm1!!(r; minsqnorm=1e-6)
            res_r_pos = norm1!!(r, 1e-6)
            @test isapprox(res_r_kw, res_r_pos, atol=tol)

            lr = LinRange(T(1), T(10), 16)
            expected_lr = lr ./ norm(lr)
            res_lr = norm1!!(lr)
            @test isapprox(res_lr, expected_lr, atol=tol, rtol=tol)
            @test isapprox(norm(res_lr), one(T), atol=tol)

            # SubArray of immutable vector
            v_imm = view(r, 1:8)
            expected_v_imm = v_imm ./ norm(v_imm)
            res_v_imm = norm1!!(v_imm)
            @test isapprox(res_v_imm, expected_v_imm, atol=tol, rtol=tol)
            @test isapprox(norm(res_v_imm), one(T), atol=tol)

            v_lr = view(lr, 1:8)
            expected_v_lr = v_lr ./ norm(v_lr)
            res_v_lr = norm1!!(v_lr)
            @test isapprox(res_v_lr, expected_v_lr, atol=tol, rtol=tol)
            @test isapprox(norm(res_v_lr), one(T), atol=tol)

            # Zero immutable vector
            z_imm = LinRange(zero(T), zero(T), 16)
            z_imm_norm = norm1!!(z_imm)
            @test all(z_imm_norm .== zero(T))
            @test !any(isnan, z_imm_norm)
        end
    end

    @testset "vecmeannorm" begin
        # Helper to compute reference (mean, norm)
        function ref_vecmeannorm(vec::AbstractVector{T}, minsqnorm=floatmin(T)) where T
            m = sum(vec) / length(vec)
            sq_diff = sum((v - m)^2 for v in vec)
            n = T(1) / sqrt(max(sq_diff, T(minsqnorm)))
            return (m, n)
        end

        for T in (Float64, Float32)
            tol = T == Float64 ? 1e-11 : 1e-4

            # Various sizes
            for n in (16, 128, 1024)
                v = randn(T, n)
                m, inv_norm = vecmeannorm(v)
                ref_m, ref_inv_norm = ref_vecmeannorm(v)
                @test m isa T
                @test inv_norm isa T
                @test isapprox(m, ref_m, atol=tol, rtol=tol)
                @test isapprox(inv_norm, ref_inv_norm, atol=tol, rtol=tol)

                # SubArray view
                v_sub = view(v, 1:(n ÷ 2))
                m_sub, inv_norm_sub = vecmeannorm(v_sub)
                ref_m_sub, ref_inv_norm_sub = ref_vecmeannorm(v_sub)
                @test isapprox(m_sub, ref_m_sub, atol=tol, rtol=tol)
                @test isapprox(inv_norm_sub, ref_inv_norm_sub, atol=tol, rtol=tol)
            end

            # StaticArrays
            s = SVector{16, T}(randn(T, 16))
            m_s, inv_norm_s = vecmeannorm(s)
            ref_m_s, ref_inv_norm_s = ref_vecmeannorm(s)
            @test m_s isa T
            @test inv_norm_s isa T
            @test isapprox(m_s, ref_m_s, atol=tol, rtol=tol)
            @test isapprox(inv_norm_s, ref_inv_norm_s, atol=tol, rtol=tol)

            # Constant vector: tests minsqnorm clamping prevents NaN/Inf
            c = fill(T(5.0), 32)
            # Default minsqnorm = floatmin(T)
            m_c_def, inv_norm_c_def = vecmeannorm(c)
            @test m_c_def ≈ T(5.0)
            @test !isnan(inv_norm_c_def) && !isinf(inv_norm_c_def)

            # Custom positional minsqnorm (above the relative rounding-noise floor)
            m_c, inv_norm_c = vecmeannorm(c, T(1e-2))
            @test m_c ≈ T(5.0)
            @test inv_norm_c ≈ T(1) / sqrt(T(1e-2))
            @test !isnan(inv_norm_c) && !isinf(inv_norm_c)

            # minsqnorm as keyword argument
            m_kw, inv_norm_kw = vecmeannorm(c; minsqnorm=1e-1)
            @test inv_norm_kw ≈ T(1) / sqrt(T(1e-1))
        end
    end

    @testset "low-contrast inputs" begin
        for T in (Float64, Float32)
            for (val, n) in ((T(0.1), 1000), (T(123.456), 100))
                c = fill(val, n)
                @test maximum(abs, mean0norm1!!(copy(c))) < sqrt(eps(T))
                @test all(isfinite, mean0norm1!!(c))
                @test maximum(abs, mean0norm1!!(SVector{16, T}(fill(val, 16)))) < sqrt(eps(T))
            end
            # Near-constant: result must not blow up (norm stays O(1), not ~1e150)
            v = T(200) .+ T(1e-3) .* randn(T, 256)
            @test norm(mean0norm1!!(copy(v))) <= T(1.01)
            # Well-conditioned input still normalized exactly
            w = T(5) .+ randn(T, 256)
            @test isapprox(norm(mean0norm1!!(copy(w))), one(T), atol=sqrt(eps(T)))
        end
    end

    @testset "mean0norm1!!" begin
        # Reference normalization (2-pass)
        function ref_normalize(vec)
            centered = vec .- (sum(vec) / length(vec))
            return centered ./ sqrt(dot(centered, centered))
        end

        for T in (Float64, Float32)
            tol = T == Float64 ? 1e-11 : 1e-4

            # Mutable dense vectors (lengths 16, 1024)
            for n in (16, 1024)
                A = randn(T, n)
                expected = ref_normalize(A)

                # In-place verification
                A_copy = copy(A)
                res = mean0norm1!!(A_copy)
                @test res === A_copy  # Mutated in-place
                @test isapprox(res, expected, atol=tol, rtol=tol)
                @test isapprox(sum(res), zero(T), atol=n * eps(T) * 100)
                @test isapprox(norm(res), one(T), atol=tol)
            end

            # SubArray (view into mutable array)
            parent_arr = randn(T, 64)
            v_view = view(parent_arr, 1:32)
            expected_view = ref_normalize(v_view)
            mean0norm1!!(v_view)
            @test isapprox(v_view, expected_view, atol=tol, rtol=tol)

            # Static vectors (immutable SVector)
            s_data = randn(T, 16)
            B = SVector{16, T}(s_data)
            expected_s = SVector{16, T}(ref_normalize(s_data))
            B_norm = mean0norm1!!(B)
            @test B_norm isa SVector{16, T}
            @test isapprox(B_norm, expected_s, atol=tol, rtol=tol)
            @test isapprox(sum(B_norm), zero(T), atol=16 * eps(T) * 100)
            @test isapprox(norm(B_norm), one(T), atol=tol)

            # Constant vector with default minsqnorm = floatmin(T)
            c = fill(T(7.0), 16)
            c_norm_def = mean0norm1!!(copy(c))
            @test all(isapprox.(c_norm_def, zero(T), atol=tol))
            @test !any(isnan, c_norm_def)

            # Constant vector with custom keyword minsqnorm
            c_norm = mean0norm1!!(copy(c); minsqnorm=1e-8)
            @test all(isapprox.(c_norm, zero(T), atol=tol))
            @test !any(isnan, c_norm)

            # Positional minsqnorm
            c_norm_pos = mean0norm1!!(copy(c), 1e-8)
            @test all(isapprox.(c_norm_pos, zero(T), atol=tol))

            # Immutable AbstractVector (non-StaticVector)
            r = T(1):T(1):T(16)
            expected_r = ref_normalize(r)
            res_r = mean0norm1!!(r)
            @test isapprox(res_r, expected_r, atol=tol, rtol=tol)
            @test isapprox(sum(res_r), zero(T), atol=16 * eps(T) * 100)
            @test isapprox(norm(res_r), one(T), atol=tol)

            lr = LinRange(T(1), T(10), 16)
            expected_lr = ref_normalize(lr)
            res_lr = mean0norm1!!(lr)
            @test isapprox(res_lr, expected_lr, atol=tol, rtol=tol)
            @test isapprox(sum(res_lr), zero(T), atol=16 * eps(T) * 100)
            @test isapprox(norm(res_lr), one(T), atol=tol)

            # SubArray of immutable vector
            v_imm = view(r, 1:8)
            expected_v_imm = ref_normalize(v_imm)
            res_v_imm = mean0norm1!!(v_imm)
            @test isapprox(res_v_imm, expected_v_imm, atol=tol, rtol=tol)
            @test isapprox(sum(res_v_imm), zero(T), atol=8 * eps(T) * 100)
            @test isapprox(norm(res_v_imm), one(T), atol=tol)

            v_lr = view(lr, 1:8)
            expected_v_lr = ref_normalize(v_lr)
            res_v_lr = mean0norm1!!(v_lr)
            @test isapprox(res_v_lr, expected_v_lr, atol=tol, rtol=tol)
            @test isapprox(sum(res_v_lr), zero(T), atol=8 * eps(T) * 100)
            @test isapprox(norm(res_v_lr), one(T), atol=tol)

            # Constant immutable vector
            c_imm = LinRange(T(7.0), T(7.0), 16)
            c_imm_norm = mean0norm1!!(c_imm)
            @test all(isapprox.(c_imm_norm, zero(T), atol=tol))
            @test !any(isnan, c_imm_norm)

            c_imm_kw = mean0norm1!!(c_imm; minsqnorm=1e-8)
            @test all(isapprox.(c_imm_kw, zero(T), atol=tol))
            c_imm_pos = mean0norm1!!(c_imm, 1e-8)
            @test all(isapprox.(c_imm_pos, zero(T), atol=tol))
        end
    end

    @testset "mutation through wrapper types" begin
        for T in (Float64, Float32)
            base = randn(T, 16)
            expected = base ./ norm(base)
            c = base .- sum(base) / 16
            expected_m = c ./ norm(c)

            # Immutable wrappers around mutable memory must be normalized in place
            for wrap in (v -> reshape(v, 2, 8) |> x -> reshape(x, 16),
                         v -> SizedVector{16}(v),
                         v -> view(reshape(v, 16), 1:16),
                         v -> reshape(view(v, 1:16), 16))
                a = copy(base); w = wrap(a)
                @test FastNorm._ismutable(typeof(w))
                norm1!!(w)
                @test isapprox(a, expected, atol=1e-5)
                a = copy(base); w = wrap(a)
                mean0norm1!!(w)
                @test isapprox(a, expected_m, atol=1e-5)
            end

            # Wrappers around immutable parents are still treated as immutable
            r = T(1):T(1):T(16)
            @test !FastNorm._ismutable(typeof(reshape(r, 16)))
            @test !FastNorm._ismutable(typeof(view(r, 1:4)))
        end
    end

    @testset "empty input" begin
        for T in (Float64, Float32)
            @test vecmeannorm(T[]) === (zero(T), one(T))
            @test vecmeannorm(SVector{0, T}()) === (zero(T), one(T))
            @test isempty(mean0norm1!!(T[]))
            @test isempty(norm1!!(T[]))
            @test isempty(mean0norm1!!(SVector{0, T}()))
        end
    end

    @testset "integer input" begin
        @test sqnorm([1, 2, 3]) == 14
        @test fast_norm([3, 4]) == 5.0
        # Normalizing functions are float-only: a clear MethodError, not an InexactError
        @test_throws MethodError norm1!!([1, 2, 3])
        @test_throws MethodError vecmeannorm([1, 2, 3])
        @test_throws MethodError mean0norm1!!([1, 2, 3])
    end

    @testset "StaticVector up to length 32" begin
        for len in (2, 3, 4, 8, 16, 32)
            s = SVector{len, Float64}(randn(len)...)
            m = MVector{len, Float64}(s)

            # sqnorm & fast_norm
            @test sqnorm(s) ≈ dot(s, s)
            @test sqnorm(m) ≈ dot(m, m)
            @test fast_norm(s) ≈ norm(s)
            @test fast_norm(m) ≈ norm(m)

            # norm1!!
            s_n = norm1!!(s)
            @test s_n isa SVector{len, Float64}
            @test norm(s_n) ≈ 1.0
            @test isapprox(norm1!!(s, 1e-6), norm1!!(s; minsqnorm=1e-6), atol=1e-11)
            m_n = norm1!!(m)
            @test m_n === m
            @test norm(m) ≈ 1.0
            m_copy = copy(m)
            @test isapprox(norm1!!(m_copy, 1e-6), norm1!!(copy(m); minsqnorm=1e-6), atol=1e-11)

            # vecmeannorm
            m_s, n_s = vecmeannorm(s)
            @test m_s ≈ sum(s) / len
            @test n_s ≈ 1.0 / norm(s .- m_s)
            @test isapprox(vecmeannorm(s, 1e-6)[2], vecmeannorm(s; minsqnorm=1e-6)[2], atol=1e-11)

            # mean0norm1!!
            s_m0 = mean0norm1!!(s)
            @test s_m0 isa SVector{len, Float64}
            @test isapprox(sum(s_m0), 0.0, atol=1e-10)
            @test norm(s_m0) ≈ 1.0
            @test isapprox(mean0norm1!!(s, 1e-6), mean0norm1!!(s; minsqnorm=1e-6), atol=1e-11)

            m2 = MVector{len, Float64}(s)
            m_m0 = mean0norm1!!(m2)
            @test m_m0 === m2
            @test isapprox(sum(m2), 0.0, atol=1e-10)
            @test norm(m2) ≈ 1.0
            @test isapprox(mean0norm1!!(copy(m2), 1e-6), mean0norm1!!(copy(m2); minsqnorm=1e-6), atol=1e-11)
        end
    end
end
