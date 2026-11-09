# FastNorm.jl

[![Build Status](https://github.com/ojwoodford/FastNorm.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/FastNorm.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/FastNorm.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/FastNorm.jl)

High-performance norms and normalization methods in Julia that trade a small amount of numerical precision for significant speedups and reduced memory passes.

## Why FastNorm.jl?

Standard Julia linear algebra is built around standard BLAS and LAPACK libraries via `LinearAlgebra.jl`. While standard implementations are designed to guarantee full IEEE-754 precision and protect against extreme floating-point underflow or overflow, they introduce significant overhead in performance-critical inner loops involving small- to medium-sized data:

1. **BLAS Call and Thread-Checking Overhead**:
   Standard routines such as `LinearAlgebra.norm` invoke external C/Fortran BLAS subroutines (`dnrm2`). On every call, this incurs `ccall` overhead, foreign function interface (FFI) boundary transitions, argument marshalling, and multithreading synchronization checks. For small- to medium-sized vectors, this administrative overhead dwarfs the actual floating-point work. FastNorm executes native Julia SIMD loops directly in CPU registers without leaving the Julia runtime.

2. **Over-Defensive Multi-Pass Scaling in Standard Norms**:
   `LinearAlgebra.norm` uses BLAS `dnrm2` (or `generic_norm2`), which executes defensive multi-pass scaling (e.g., Blue's algorithm) with extra conditional branches and scaling factors to protect against extreme underflow or overflow. When data magnitudes are well within normal IEEE-754 dynamic range, this defensive handling adds unnecessary latency. `fast_norm` and `sqnorm` compute norms directly in a single pass, delivering **up to 17x speedups** over `LinearAlgebra.norm`.

3. **In-Place Unit-Norm Normalization (`norm1!!`)**:
   Standard in-place normalization (`LinearAlgebra.normalize!`) delegates to BLAS `dnrm2` on vectors of size $\ge 32$ and does not protect against division by zero (producing `NaN`s for zero vectors). `norm1!!` uses single-pass SIMD `sqnorm` and normalizes the vector in-place (or returns a new static vector) using reciprocal multiplication. To prevent division by zero or NaN creation on degenerate vectors without artificially restricting floating-point dynamic range, `norm1!!` defaults to `minsqnorm = floatmin(T)` (the smallest positive normal number for type `T`), delivering **up to 6.9x speedups** over `LinearAlgebra.normalize!`.

4. **Single-Pass Mean and Normalizer (`vecmeannorm`)**:
   Standard calculations of the mean $\bar{x}$ and norm of deviations $\sqrt{\sum_i (x_i - \bar{x})^2}$ require two sequential memory passes over the vector: pass 1 computes the mean, and pass 2 computes the squared differences from the mean. `vecmeannorm` computes both the mean and the normalization factor in a **single pass** using the algebraic identity:
   $$\sum_{i=1}^N (x_i - \bar{x})^2 = \sum_{i=1}^N x_i^2 - \bar{x} \sum_{i=1}^N x_i$$
   Clamping with the default minimum normalization constant `floatmin(T)` protects against division by zero or NaN results when variance is zero. Halving the memory traffic and leveraging SIMD vectorization yields **up to 3.3x speedups**.

5. **Two-Pass In-Place Zero-Mean Unit-Norm Normalization (`mean0norm1!!`)**:
   Normalizing a vector to have zero mean and unit Euclidean norm ($\mu = 0$, $\|v\|_2 = 1$) conventionally requires 4 memory passes:
   1. Compute mean (pass 1: read vector)
   2. Subtract mean (pass 2: write vector)
   3. Compute Euclidean norm via BLAS (pass 3: read vector)
   4. Divide by norm (pass 4: write vector)

   `mean0norm1!!` uses `vecmeannorm` to compute both the mean and the normalization factor in a single read pass, and then applies the affine transformation $(v_i - \bar{x}) \cdot \text{scale}$ in a second pass. Memory is traversed only twice rather than four times. For mutable arrays (such as standard `Vector` or views), the operation modifies elements in-place with **zero allocations** using an inlined SIMD loop; for immutable arrays (such as `StaticArrays.SVector`), a new normalized static vector is returned. Clamped with `floatmin(T)` by default, it delivers **up to 9.0x speedups** over standard multi-pass normalization pipelines.

## Performance Evaluation

A benchmark script is provided in [`examples/speed_test.jl`](examples/speed_test.jl). It benchmarks all FastNorm functions against standard implementations across multiple input sizes using warmup cycles, minimum-time filtering over multiple trials, and compiler barriers to prevent constant folding.

To run the speed test:

```bash
julia --project=. examples/speed_test.jl
```

### Benchmark Results

Results collected on an Apple Silicon (M-series) CPU:

### Standard Dense Vectors (`Vector{Float64}`)

#### 1. Euclidean Norm (`fast_norm` vs `LinearAlgebra.norm`)

| Vector Size | Standard (`LinearAlgebra.norm`) | FastNorm (`fast_norm`) | Speedup | Time Reduction |
|:-----------:|:-------------------------------:|:----------------------:|:-------:|:--------------:|
| 4 | ~5.4 ns | ~1.9 ns | ~2.9x | ~65.9% |
| 16 | ~13.3 ns | ~2.9 ns | ~4.6x | ~78.4% |
| 64 | ~60.8 ns | ~5.7 ns | ~10.6x | ~90.5% |
| 256 | ~275.5 ns | ~17.5 ns | ~15.7x | ~93.6% |
| 1024 | ~1.24 μs | ~71.3 ns | ~17.3x | ~94.2% |
| 4096 | ~5.08 μs | ~312.4 ns | ~16.3x | ~93.8% |

#### 2. Squared Euclidean Norm (`sqnorm` vs `LinearAlgebra.dot(a, a)`)

| Vector Size | Standard (`dot(a, a)`) | FastNorm (`sqnorm`) | Speedup | Time Reduction |
|:-----------:|:----------------------:|:-------------------:|:-------:|:--------------:|
| 4 | ~9.3 ns | ~1.9 ns | ~5.0x | ~80.0% |
| 16 | ~13.0 ns | ~2.8 ns | ~4.7x | ~78.6% |
| 64 | ~14.6 ns | ~6.5 ns | ~2.3x | ~55.6% |
| 256 | ~39.0 ns | ~16.0 ns | ~2.4x | ~59.0% |
| 1024 | ~138.2 ns | ~67.0 ns | ~2.1x | ~51.5% |
| 4096 | ~539.4 ns | ~312.5 ns | ~1.7x | ~42.1% |

#### 3. Unit-Norm Normalization (`norm1!!` vs `LinearAlgebra.normalize!`)

| Vector Size | Standard (`normalize!`) | FastNorm (`norm1!!`) | Speedup | Time Reduction |
|:-----------:|:-----------------------:|:--------------------:|:-------:|:--------------:|
| 4 | ~7.2 ns | ~15.6 ns | ~0.5x | -117.2% |
| 16 | ~27.6 ns | ~19.1 ns | ~1.4x | ~31.0% |
| 64 | ~104.3 ns | ~23.1 ns | ~4.5x | ~77.8% |
| 256 | ~283.7 ns | ~54.9 ns | ~5.2x | ~80.6% |
| 1024 | ~1.26 μs | ~194.0 ns | ~6.5x | ~84.6% |
| 4096 | ~5.12 μs | ~741.8 ns | ~6.9x | ~85.5% |

#### 4. Mean and Normalizer (`vecmeannorm` vs Standard 2-Pass)

| Vector Size | Standard (2-pass) | FastNorm (`vecmeannorm`) | Speedup | Time Reduction |
|:-----------:|:-----------------:|:------------------------:|:-------:|:--------------:|
| 4 | ~4.7 ns | ~2.6 ns | ~1.8x | ~45.7% |
| 16 | ~10.8 ns | ~4.7 ns | ~2.3x | ~56.9% |
| 64 | ~20.9 ns | ~8.2 ns | ~2.5x | ~60.6% |
| 256 | ~65.8 ns | ~25.4 ns | ~2.6x | ~61.4% |
| 1024 | ~272.1 ns | ~86.2 ns | ~3.2x | ~68.3% |
| 4096 | ~1.09 μs | ~344.6 ns | ~3.1x | ~68.2% |

#### 5. Zero-Mean Unit-Norm Normalization (`mean0norm1!!` vs Standard Multi-Pass)

| Vector Size | Standard (Multi-pass) | FastNorm (`mean0norm1!!`) | Speedup | Time Reduction |
|:-----------:|:---------------------:|:---------------------------:|:-------:|:--------------:|
| 4 | ~26.6 ns | ~20.2 ns | ~1.3x | ~24.0% |
| 16 | ~47.9 ns | ~22.3 ns | ~2.1x | ~53.4% |
| 64 | ~136.8 ns | ~27.8 ns | ~4.9x | ~79.7% |
| 256 | ~455.3 ns | ~64.3 ns | ~7.1x | ~85.9% |
| 1024 | ~1.69 μs | ~195.4 ns | ~8.6x | ~88.4% |
| 4096 | ~6.66 μs | ~741.9 ns | ~9.0x | ~88.9% |

### Static Vectors (`SVector{N, Float64}` up to length 32)

#### 1. SVector Euclidean Norm (`fast_norm` vs `LinearAlgebra.norm`)

| Vector Size | Standard (`LinearAlgebra.norm`) | FastNorm (`fast_norm`) | Speedup | Time Reduction |
|:-----------:|:-------------------------------:|:----------------------:|:-------:|:--------------:|
| 2 | ~2.8 ns | ~1.2 ns | ~2.3x | ~55.7% |
| 3 | ~2.8 ns | ~1.2 ns | ~2.3x | ~55.6% |
| 4 | ~2.8 ns | ~1.2 ns | ~2.3x | ~55.8% |
| 8 | ~3.6 ns | ~2.1 ns | ~1.7x | ~42.1% |
| 16 | ~5.8 ns | ~3.2 ns | ~1.8x | ~45.3% |
| 32 | ~14.9 ns | ~7.4 ns | ~2.0x | ~50.0% |

#### 2. SVector Squared Euclidean Norm (`sqnorm` vs `LinearAlgebra.dot(s, s)`)

| Vector Size | Standard (`dot(s, s)`) | FastNorm (`sqnorm`) | Speedup | Time Reduction |
|:-----------:|:----------------------:|:-------------------:|:-------:|:--------------:|
| 2 | ~2.2 ns | ~1.2 ns | ~1.8x | ~42.9% |
| 3 | ~2.2 ns | ~1.2 ns | ~1.8x | ~42.9% |
| 4 | ~2.2 ns | ~1.2 ns | ~1.7x | ~42.8% |
| 8 | ~3.2 ns | ~1.4 ns | ~2.3x | ~56.0% |
| 16 | ~6.1 ns | ~2.9 ns | ~2.1x | ~52.1% |
| 32 | ~12.0 ns | ~6.0 ns | ~2.0x | ~50.0% |

#### 3. SVector Unit-Norm Normalization (`norm1!!` vs `LinearAlgebra.normalize`)

| Vector Size | Standard (`LinearAlgebra.normalize`) | FastNorm (`norm1!!`) | Speedup | Time Reduction |
|:-----------:|:------------------------------------:|:--------------------:|:-------:|:--------------:|
| 2 | ~2.8 ns | ~2.2 ns | ~1.3x | ~20.1% |
| 3 | ~2.8 ns | ~2.9 ns | ~1.0x | -2.1% |
| 4 | ~2.8 ns | ~2.9 ns | ~1.0x | -0.4% |
| 8 | ~4.4 ns | ~4.3 ns | ~1.0x | ~1.5% |
| 16 | ~8.7 ns | ~7.8 ns | ~1.1x | ~10.4% |
| 32 | ~25.0 ns | ~18.4 ns | ~1.4x | ~26.1% |

#### 4. SVector Mean and Normalizer (`vecmeannorm` vs Standard 2-Pass)

| Vector Size | Standard (2-pass) | FastNorm (`vecmeannorm`) | Speedup | Time Reduction |
|:-----------:|:-----------------:|:------------------------:|:-------:|:--------------:|
| 2 | ~1.2 ns | ~2.5 ns | ~0.5x | -100.0% |
| 3 | ~1.2 ns | ~2.6 ns | ~0.5x | -111.2% |
| 4 | ~1.2 ns | ~3.7 ns | ~0.3x | -200.1% |
| 8 | ~4.5 ns | ~3.6 ns | ~1.3x | ~20.3% |
| 16 | ~10.3 ns | ~6.4 ns | ~1.6x | ~37.8% |
| 32 | ~29.9 ns | ~10.8 ns | ~2.8x | ~64.1% |

#### 5. SVector Zero-Mean Unit-Norm Normalization (`mean0norm1!!` vs Standard Multi-Pass)

| Vector Size | Standard (Multi-pass) | FastNorm (`mean0norm1!!`) | Speedup | Time Reduction |
|:-----------:|:---------------------:|:---------------------------:|:-------:|:--------------:|
| 2 | ~2.8 ns | ~2.5 ns | ~1.1x | ~11.7% |
| 3 | ~4.3 ns | ~3.0 ns | ~1.5x | ~31.5% |
| 4 | ~4.2 ns | ~2.9 ns | ~1.5x | ~31.1% |
| 8 | ~9.1 ns | ~4.5 ns | ~2.0x | ~50.7% |
| 16 | ~25.5 ns | ~8.0 ns | ~3.2x | ~68.7% |
| 32 | ~69.9 ns | ~19.9 ns | ~3.5x | ~71.5% |

## Externally Available Methods

### `sqnorm`

```julia
sqnorm(x::Number) -> Number
sqnorm(vec::AbstractVector) -> eltype(vec)
```

Computes the squared L2 norm $x^2$ (for a `Number`) or $\sum_i v_i^2$ (for an `AbstractVector`) using `LoopVectorization.@turbo`.

### `fast_norm`

```julia
fast_norm(vec::AbstractVector) -> Real
```

Computes the Euclidean L2 norm $\sqrt{\text{sqnorm}(\text{vec})}$ directly in native Julia without calling BLAS.

### `norm1!!`

```julia
norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T -> V
norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T -> AbstractVector{T}
```

Normalizes a vector to unit Euclidean norm ($\|v\|_2 \approx 1$). Computes the squared norm via `sqnorm` in a single pass, then scales the vector in a second pass. For mutable vectors (`Vector`, views of mutable arrays), the operation modifies `vec` in-place (`vec === norm1!!(vec)`). For immutable arrays (`StaticArrays.SVector`), it returns a new normalized vector. Clamping to `minsqnorm` (defaulting to `floatmin(T)`) prevents division by zero and NaN generation for zero vectors.

### `vecmeannorm`

```julia
vecmeannorm(vec::AbstractVector{T}, minsqnorm::Real) where T -> Tuple{T, T}
vecmeannorm(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T -> Tuple{T, T}
```

Computes the mean $\bar{x}$ and normalization factor $1 / \sqrt{\max(\sum (x_i - \bar{x})^2, \text{minsqnorm})}$ in a single memory pass over `vec`. Returns `(vecmean, vecnorm)`. Clamping to `minsqnorm` (defaulting to `floatmin(T)`) guards against division by zero for zero-variance inputs.

### `mean0norm1!!`

```julia
mean0norm1!!(vec::V, minsqnorm::Real) where V <: AbstractVector{T} where T -> V
mean0norm1!!(vec::AbstractVector{T}; minsqnorm::Real=floatmin(T)) where T -> AbstractVector{T}
```

Normalizes a vector to zero mean and unit Euclidean norm ($\mu = 0, \|v\|_2 = 1$). Uses `vecmeannorm` to compute the mean and normalizer in one pass, then applies the scaling in a second pass. For mutable vectors (`Vector`, views of mutable arrays), the operation is in-place and returns the modified array (`vec === mean0norm1!!(vec)`). For immutable arrays (`StaticArrays.SVector`), it returns a new normalized vector. Defaults to `minsqnorm = floatmin(T)`.

## Example

```julia
using FastNorm, StaticArrays

# Standard dense vectors
a = randn(100)
s = sqnorm(a)
n = fast_norm(a)

# In-place unit-norm normalization
norm1!!(a)

# Single-pass mean and normalizer
mu, normalizer = vecmeannorm(a)

# In-place zero-mean unit-norm normalization
mean0norm1!!(a)

# Immutable static vectors
svec = SVector{4, Float64}(1.0, 2.0, 3.0, 4.0)
svec_unit = norm1!!(svec)
svec_norm = mean0norm1!!(svec)
```
