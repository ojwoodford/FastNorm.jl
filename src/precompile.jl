using PrecompileTools

@setup_workload begin
    # Float64 data
    v_dense64 = [1.0, 2.0, 3.0, 4.0]
    v_view64 = view(v_dense64, 1:3)

    # Float32 data
    v_dense32 = Float32[1.0, 2.0, 3.0, 4.0]
    v_view32 = view(v_dense32, 1:3)

    @compile_workload begin
        # 1. Vector sqnorm (Float64 and Float32)
        sqnorm(v_dense64)
        sqnorm(v_view64)
        sqnorm(v_dense32)
        sqnorm(v_view32)

        # 2. vecmeannorm (Float64 and Float32)
        vecmeannorm(v_dense64, 1e-8)
        vecmeannorm(v_view64, 1e-8)
        vecmeannorm(v_dense32, 1e-8)
        vecmeannorm(v_view32, 1e-8)

        # 3. mean0norm1!! (Float64 and Float32)
        mean0norm1!!(v_dense64, 1e-8)
        mean0norm1!!(v_view64, 1e-8)
        mean0norm1!!(v_dense32, 1e-8)
        mean0norm1!!(v_view32, 1e-8)

        # 4. norm1!! (Float64 and Float32)
        norm1!!(v_dense64, 1e-8)
        norm1!!(v_view64, 1e-8)
        norm1!!(v_dense32, 1e-8)
        norm1!!(v_view32, 1e-8)
    end
end
