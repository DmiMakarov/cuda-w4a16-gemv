# w4a16 GEMV in CUDA

A fused dequantize + matrix-vector product for weight-only int4 with per-group scales, batch = 1.
The int4 matrix is never expanded to fp16 in memory: each thread reads packed weights, dequantizes
in registers and accumulates.

```
y[i] = Σ_g  s[i][g] · Σ_{j ∈ g} (q[i][j] − z[i][g]) · x[j]
```

On an RTX 3090 with a 16384 × 16384 matrix, G = 32:

- **3.6× faster than cuBLAS GEMV on fp16 weights.** 3.37× of that is simply fewer bytes to read
  (159 MB instead of 537 MB); the other 1.07× is a higher read rate.
- **36× faster than dequantize + `matmul` in PyTorch.**
- The final kernel reads 813 GB/s. Nsight Compute shows DRAM at 90% of peak, so the kernel is memory-bound.

## Data format

| Array | Type | Size | Content |
|---|---|---|---|
| `w` | `uint32` | `rows × w_stride` | 8 weights per word, first weight in the low bits |
| `s` | fp16 (stored as `uint16` bits) | `rows × cols/G` | scale per group |
| `z` | `uint8` | `rows × cols/G` | zero-point per group, 0..15 |

`x` is fp16; the accumulator and `y` are fp32. `cols` must be divisible by `G` and `G` by 8
(`warp_w4` needs `G` divisible by 32). `w_stride` is rounded up to 32 words, so every row starts on a 128-byte boundary.
At G = 32 a weight costs 4.75 bits, which caps the size advantage over fp16 at 3.37×.

## Kernels

Each kernel adds one idea to the previous one.

| Kernel | Idea |
|---|---|
| `naive` | one thread per row |
| `warp` | one warp per row: 32 lanes read 32 adjacent words, then a `__shfl_down_sync` reduction |
| `warp_xvec` | 8 values of `x` per 16-byte load instead of eight 2-byte loads |
| `warp_w4` | 4 words (32 weights) per 16-byte load; `s` and `z` are read once per 32 weights |

## Results

RTX 3090, driver 535.86.10, CUDA 12.8, GCC 13.4. Median of 10 repetitions (cv below 1%), PyTorch excepted, see below.
Weights and vectors are already in device memory; one iteration is launch + kernel + synchronization.
Uploads are not included.

### 16384 × 16384, G = 32

| | time, µs | matrix, MB | read rate, GB/s | speedup vs. cuBLAS fp16 |
|---|---:|---:|---:|---:|
| `naive` | 783.8 | 159 | 203 | 0.90× |
| `warp` | 285.6 | 159 | 558 | 2.47× |
| `warp_xvec` | 200.5 | 159 | 795 | 3.52× |
| **`warp_w4`** | **196.1** | 159 | **813** | **3.60×** |
| cuBLAS fp16 (`cublasHSSgemvBatched`) | 705.5 | 537 | 761 | 1.00× |
| cuBLAS fp32 (`cublasSgemv`) | 1215.4 | 1074 | 883 | 0.58× |
| PyTorch 2.0.1, fp16 `matmul` | 696.3 | 537 | | 1.01× |
| PyTorch 2.0.1, dequantize + `matmul` | 7134.1 | 159 | | 0.10× |

The fp32 row checks that the fp16 bar is not unfairly slow: PyTorch fp16 lands within 2% of cuBLAS fp16.
PyTorch was timed with its own script, 50 iterations.

### 2048 × 2048, G = 32

| | time, µs |
|---|---:|
| `naive` | 47.3 |
| `warp` | 10.7 |
| `warp_xvec` | 8.9 |
| **`warp_w4`** | **8.4** |
| cuBLAS fp16 | 21.3 |
| cuBLAS fp32 | 30.2 |
| PyTorch, fp16 `matmul` | 40.5 |
| PyTorch, dequantize + `matmul` | 166.0 |

Read this one with care: the int4 matrix (2.5 MB) fits in the 6 MB L2 while the fp16 one (8.4 MB) does not, the GPU is
not filled (0.52 waves per SM for `warp_w4`), and launch and synchronization are a visible part of 8 µs.

## What limited each version

Nsight Compute, 16384 × 16384. Profiled times are not comparable with the tables above: clocks differ.

| | `naive` | `warp` | `warp_xvec` | `warp_w4` |
|---|---:|---:|---:|---:|
| DRAM throughput, % of peak | 20 | 52 | 82 | **90** |
| SM busy, % | 23 | 58 | 78 | 67 |
| Memory busy, % | 57 | **95** | 37 | 83 |
| Achieved occupancy, % | **13** | 91 | 90 | 91 |
| L1 hit rate, % | 87 | 96 | 78 | 87 |
| Warp instructions, M | 86.0 | 89.3 | 75.3 | 57.9 |

All four read about the same from DRAM (163 MB for `naive`, 160 MB for the rest). What changed was everything else.

- **`naive`: not enough threads.** 16 384 threads is one per row, against a capacity of about 126 000. The GPU was 13% occupied.
  The 2.7× gain of `warp` mixes better occupancy with better locality; the profile does not separate the two.
- **`warp`: too many small loads.** Lanes are 16 bytes apart, so each 2-byte load of `x` touches 16 sectors where 2 would do,
  and it repeats 8 times per word. Nsight reports 117 440 512 excess sectors, which is exactly
  `16384 rows × 64 steps × 8 loads × 14 sectors`. These hit in L1, which is why a 96% hit rate did not mean good performance.
- **`warp_xvec`: DRAM.** One 16-byte load removed that bottleneck and DRAM rose to 82%.
- **`warp_w4`: DRAM.** 23% fewer instructions bought only 2% of time, because instructions were no longer the limit.

## What was not done, and why

| Idea | Reason |
|---|---|
| Precomputed `Σx` per group (`Σq·x − z·Σx`), more FMA | fewer instructions; the kernel is DRAM-bound. May pay off on GPUs with faster memory relative to compute |
| `x`, `s` or `z` in shared memory | `x` is served mostly by L1; `s` and `z` are each read once |
| Fewer registers, other block sizes | occupancy is already 91% |
| 4-bit zero-points | at most 2.6% fewer bytes |
| G = 64, symmetric quantization | at most 8% and 5% fewer bytes, at the cost of accuracy, which is not measured here |

The remaining headroom for this format is roughly 8%, judging by the 883 GB/s that cuBLAS fp32 reaches on the same card.
Processing several vectors per matrix read (GEMM) would raise throughput, but not lower the latency of a single vector.

## Verification

- CPU reference (`cpu::gemv`, double accumulation) tested against an independent per-element dequantization.
- 29 CPU tests; 13 GPU tests compare every kernel with the CPU reference, including padded rows, uneven
  last warp step and one-hot inputs. All pass on the RTX 3090.
- `compute-sanitizer` on the GPU tests: 0 errors.
- The benchmark compares each int4 kernel with the CPU reference and reports an error instead of a time if they differ.

## Limitations

- One GPU and two matrix sizes; random weights N(0, 0.02).
- No accuracy measurement on a real model: this project is about kernel speed, not quantization quality.
- cuBLAS was not profiled, so why its fp16 path reads 7% slower than `warp_w4` is unknown.

## Build and run

CMake 3.25+, GCC 13+ (`std::format`), GoogleTest; for CUDA, toolkit 12.4+ and Google Benchmark 1.7+.

```bash
# CPU only
cmake -S . -B build && cmake --build build -j && ./build/tests

# With CUDA; set the architecture of your GPU (86 = RTX 30xx)
cmake -S . -B build-cuda -DQUANT_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j
./build-cuda/tests_gpu       # skipped tests mean no GPU was found

./build-cuda/bench_gemv --benchmark_repetitions=10 --benchmark_report_aggregates_only=true
python3 bench/bench_torch.py 16384 16384 32
```

## Layout

```
include/   matrix format, CPU and GPU declarations
src/cpu/   quantization, fp16 conversion, reference GEMV
src/gpu/   the four kernels
test/      CPU and GPU tests
bench/     Google Benchmark and PyTorch baselines
results/   raw benchmark output and Nsight Compute reports
```
