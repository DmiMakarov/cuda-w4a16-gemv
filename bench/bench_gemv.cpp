// GEMV benchmarks, batch = 1. Matrices and vectors are already on the GPU; one iteration is
// launch + kernel + synchronization. Upload/* is separate: what it costs to load the weights.
//
// Usage: bench_gemv [benchmark flags] [rows cols G]
// Without a shape: 2048 x 2048 and 16384 x 16384, G = 32.

#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu/float_fp16.hpp"
#include "cpu/gemv.hpp"
#include "cpu/quantize.hpp"
#include "gpu/cuda_matrix_quant.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/gemv_naive.hpp"
#include "gpu/gemv_warp.hpp"
#include "gpu/gemv_warp_w4.hpp"
#include "gpu/gemv_warp_xvec.hpp"

namespace
{

using quant::gpu::CudaMatrixQuant;
using quant::gpu::DeviceBuffer;
using quant::gpu::detail::check;

using GemvFn = void (*)(const CudaMatrixQuant&, const DeviceBuffer<std::uint16_t>&, DeviceBuffer<float>&);

struct Shape
{
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t G;
};

void check_blas(cublasStatus_t status, const char* what)
{
    if(status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::format("{}: cuBLAS status {}", what, static_cast<int>(status)));
}

void sync()
{
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

struct CublasHandle
{
    CublasHandle() { check_blas(cublasCreate(&h), "cublasCreate"); }
    ~CublasHandle() { cublasDestroy(h); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t h = nullptr;
};

// --- Host data ------------------------------------------------------------------
// Google Benchmark calls a benchmark function many times, so the slow setup is cached.

struct HostData
{
    Shape shape{};
    std::vector<float> a;            // fp32 weights
    std::vector<std::uint16_t> ah;   // fp16 weights
    std::vector<float> xf;
    std::vector<std::uint16_t> x;
    quant::cpu::MatrixQuant m{};
    std::vector<float> ref;          // cpu::gemv

    explicit HostData(Shape s) : shape(s)
    {
        const std::size_t n = static_cast<std::size_t>(s.rows) * s.cols;
        std::mt19937 rng(42);
        std::normal_distribution<float> weights(0.0f, 0.02f), inputs(0.0f, 1.0f);

        a.resize(n);
        for(float& v : a)
            v = weights(rng);
        ah.resize(n);
        std::transform(a.begin(), a.end(), ah.begin(), quant::cpu::float_to_fp16_bits);

        x.resize(s.cols);
        xf.resize(s.cols);
        for(std::size_t j = 0; j < s.cols; j++)
        {
            x[j] = quant::cpu::float_to_fp16_bits(inputs(rng));
            xf[j] = quant::cpu::fp16_bits_to_float(x[j]);
        }

        m = quant::cpu::quantize(a, s.rows, s.cols, s.G);
        ref.resize(s.rows);
        quant::cpu::gemv(m.view(), x, ref);
    }
};

// Keeps one shape at a time: the 16k one takes gigabytes of host memory.
const HostData& host_data(Shape s)
{
    static std::unique_ptr<HostData> cached;
    if(!cached || cached->shape.rows != s.rows || cached->shape.cols != s.cols || cached->shape.G != s.G)
    {
        cached.reset();
        cached = std::make_unique<HostData>(s);
    }
    return *cached;
}

// --- Counters ---------------------------------------------------------------------

// Device-to-device copy, read + write. An estimate of what the memory delivers, not a hard limit.
double copy_bandwidth()
{
    static const double value = [] {
        const std::size_t bytes = std::size_t{256} << 20;
        DeviceBuffer<std::uint8_t> src(bytes), dst(bytes);
        const auto copy = [&] {
            check(cudaMemcpy(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice), "copy");
        };

        std::vector<double> seconds(50);
        for(int n = 0; n < 10; n++)
            copy();
        sync();
        for(double& t : seconds)
        {
            const auto start = std::chrono::steady_clock::now();
            copy();
            sync();
            t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        std::sort(seconds.begin(), seconds.end());
        return 2.0 * static_cast<double>(bytes) / seconds[seconds.size() / 2];
    }();
    return value;
}

void set_bytes(benchmark::State& state, double bytes)
{
    state.counters["MB"] = bytes / 1e6;
    state.counters["bytes/s"] = benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate);
}

void set_kernel_counters(benchmark::State& state, double bytes)
{
    set_bytes(state, bytes);
    state.counters["pct_copy_bw"] =
        benchmark::Counter(100.0 * bytes / copy_bandwidth(), benchmark::Counter::kIsIterationInvariantRate);
}

// Bytes a w4 kernel has to read: weights without padding, s (fp16) and z (u8).
double w4_bytes(Shape s)
{
    const double n = static_cast<double>(s.rows) * s.cols;
    return n / 2.0 + n / s.G * 3.0;
}

// --- Benchmarks -------------------------------------------------------------------

template <typename F>
void run_gemv(benchmark::State& state, F&& launch)
{
    for(auto _ : state)
    {
        launch();
        sync();
    }
}

void w4a16(benchmark::State& state, Shape shape, GemvFn fn)
{
    const HostData& d = host_data(shape);
    const CudaMatrixQuant dm(d.m);
    DeviceBuffer<std::uint16_t> dx(shape.cols);
    DeviceBuffer<float> dy(shape.rows);
    dx.copy_from_host(d.x);

    run_gemv(state, [&] { fn(dm, dx, dy); });

    // Do not report timings of a wrong result.
    std::vector<float> y(shape.rows);
    dy.copy_to_host(y);
    float max_ref = 0.0f;
    bool ok = true;
    for(const float v : d.ref)
        max_ref = std::max(max_ref, std::fabs(v));
    for(std::size_t i = 0; i < y.size(); i++)
        ok = ok && std::fabs(d.ref[i] - y[i]) <= 1e-3f * max_ref;  // false for NaN
    if(!ok)
    {
        state.SkipWithError("result does not match cpu::gemv");
        return;
    }

    set_kernel_counters(state, w4_bytes(shape));
}

// Row-major rows x cols is column-major cols x rows, hence CUBLAS_OP_T below.

void cublas_fp16(benchmark::State& state, Shape shape)
{
    const HostData& d = host_data(shape);
    DeviceBuffer<std::uint16_t> da(d.ah.size()), dx(shape.cols);
    DeviceBuffer<float> dy(shape.rows);
    da.copy_from_host(d.ah);
    dx.copy_from_host(d.x);

    // The batched API takes arrays of pointers that live on the device.
    const auto* a_ptr = reinterpret_cast<const __half*>(da.data());
    const auto* x_ptr = reinterpret_cast<const __half*>(dx.data());
    float* y_ptr = dy.data();
    DeviceBuffer<const __half*> a_arr(1), x_arr(1);
    DeviceBuffer<float*> y_arr(1);
    a_arr.copy_from_host(std::span<const __half* const>(&a_ptr, 1));
    x_arr.copy_from_host(std::span<const __half* const>(&x_ptr, 1));
    y_arr.copy_from_host(std::span<float* const>(&y_ptr, 1));

    const CublasHandle handle;
    const float one = 1.0f, zero = 0.0f;
    const int c = static_cast<int>(shape.cols), r = static_cast<int>(shape.rows);

    run_gemv(state, [&] {
        check_blas(cublasHSSgemvBatched(handle.h, CUBLAS_OP_T, c, r, &one, a_arr.data(), c, x_arr.data(), 1, &zero,
                                        y_arr.data(), 1, 1),
                   "cublasHSSgemvBatched");
    });
    set_kernel_counters(state, static_cast<double>(d.ah.size()) * 2.0);
}

// Guards against an unfairly slow bar: the batched fp16 path may be poor at batch = 1.
void cublas_fp32(benchmark::State& state, Shape shape)
{
    const HostData& d = host_data(shape);
    DeviceBuffer<float> da(d.a.size()), dx(shape.cols), dy(shape.rows);
    da.copy_from_host(d.a);
    dx.copy_from_host(d.xf);

    const CublasHandle handle;
    const float one = 1.0f, zero = 0.0f;
    const int c = static_cast<int>(shape.cols), r = static_cast<int>(shape.rows);

    run_gemv(state, [&] {
        check_blas(cublasSgemv(handle.h, CUBLAS_OP_T, c, r, &one, da.data(), c, dx.data(), 1, &zero, dy.data(), 1),
                   "cublasSgemv");
    });
    set_kernel_counters(state, static_cast<double>(d.a.size()) * 4.0);
}

// Host -> device, pageable memory, buffers allocated once.
void upload_w4(benchmark::State& state, Shape shape)
{
    const auto& m = host_data(shape).m;
    DeviceBuffer<std::uint32_t> w(m.w.size());
    DeviceBuffer<std::uint16_t> s(m.s.size());
    DeviceBuffer<std::uint8_t> z(m.z.size());

    run_gemv(state, [&] {
        w.copy_from_host(m.w);
        s.copy_from_host(m.s);
        z.copy_from_host(m.z);
    });
    set_bytes(state, static_cast<double>(m.w.size() * 4 + m.s.size() * 2 + m.z.size()));
}

void upload_fp16(benchmark::State& state, Shape shape)
{
    const auto& ah = host_data(shape).ah;
    DeviceBuffer<std::uint16_t> da(ah.size());

    run_gemv(state, [&] { da.copy_from_host(ah); });
    set_bytes(state, static_cast<double>(ah.size()) * 2.0);
}

// An exception out of a benchmark would end the process; report it and go on.
template <typename F>
void add(const std::string& name, Shape shape, F fn)
{
    const std::string full = std::format("{}/{}x{}/G{}", name, shape.rows, shape.cols, shape.G);
    benchmark::RegisterBenchmark(full.c_str(), [fn, shape](benchmark::State& state) {
        try
        {
            fn(state, shape);
        }
        catch(const std::exception& e)
        {
            state.SkipWithError(e.what());
        }
    })
        ->UseRealTime()  // the CPU only waits, its time says nothing
        ->Unit(benchmark::kMicrosecond)
        ->MinWarmUpTime(0.1);
}

void add_all(Shape shape)
{
    // A new kernel is added here.
    add("w4a16_naive", shape, [](benchmark::State& st, Shape s) { w4a16(st, s, quant::gpu::gemv_naive); });
    add("w4a16_warp", shape, [](benchmark::State& st, Shape s) { w4a16(st, s, quant::gpu::gemv_warp); });
    add("w4a16_warp_xvec", shape, [](benchmark::State& st, Shape s) { w4a16(st, s, quant::gpu::gemv_warp_xvec); });
    if(shape.G % 32 == 0)
        add("w4a16_warp_w4", shape, [](benchmark::State& st, Shape s) { w4a16(st, s, quant::gpu::gemv_warp_w4); });

    add("cublas_fp16", shape, cublas_fp16);
    add("cublas_fp32", shape, cublas_fp32);
    add("Upload/w4", shape, upload_w4);
    add("Upload/fp16", shape, upload_fp16);
}

} // namespace

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);  // removes the flags it knows

    std::vector<Shape> shapes{{2048, 2048, 32}, {16384, 16384, 32}};
    if(argc == 4)
    {
        const auto arg = [&](int i) { return static_cast<std::uint32_t>(std::strtoul(argv[i], nullptr, 10)); };
        shapes = {{arg(1), arg(2), arg(3)}};
    }
    else if(argc != 1)
    {
        std::fprintf(stderr, "usage: bench_gemv [benchmark flags] [rows cols G]\n");
        return 2;
    }

    cudaDeviceProp prop{};
    if(const cudaError_t err = cudaGetDeviceProperties(&prop, 0); err != cudaSuccess)
    {
        std::fprintf(stderr, "no usable CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    benchmark::AddCustomContext("gpu", prop.name);
    benchmark::AddCustomContext("gpu_l2_mb", std::format("{:.1f}", prop.l2CacheSize / 1e6));
    try
    {
        benchmark::AddCustomContext("copy_bw_gb_per_s", std::format("{:.1f}", copy_bandwidth() / 1e9));
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    for(const Shape s : shapes)
        add_all(s);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
