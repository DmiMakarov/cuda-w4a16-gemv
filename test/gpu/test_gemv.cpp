#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "cpu/gemv.hpp"
#include "cpu/quantize.hpp"
#include "gpu/cuda_matrix_quant.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/gemv_naive.hpp"
#include "gpu/gemv_warp.hpp"
#include "gpu/gpu_test.hpp"
#include "test_utils.hpp"

namespace
{

using quant::gpu::CudaMatrixQuant;
using quant::gpu::DeviceBuffer;
using test_utils::Params;
using test_utils::dequantize_at;
using test_utils::make_input;

using GemvFn = void (*)(const CudaMatrixQuant&, const DeviceBuffer<std::uint16_t>&, DeviceBuffer<float>&);

// Runs a GPU kernel; y starts as NaN so that an unwritten element cannot pass.
std::vector<float> run(GemvFn fn, const quant::cpu::MatrixQuant& m, const std::vector<std::uint16_t>& x)
{
    const CudaMatrixQuant dm(m);
    DeviceBuffer<std::uint16_t> dx(x.size());
    dx.copy_from_host(x);

    std::vector<float> y(m.rows, std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<float> dy(y.size());
    dy.copy_from_host(y);

    fn(dm, dx, dy);
    dy.copy_to_host(y);  // synchronizes with the kernel
    return y;
}

// A new kernel is added here and gets every test below.
struct Kernel
{
    const char* name;
    GemvFn fn;
};

const Kernel kernels[] = {
    {"naive", quant::gpu::gemv_naive},
    {"warp", quant::gpu::gemv_warp},
};

class GpuGemv : public ::testing::TestWithParam<Params> {};

} // namespace

TEST_P(GpuGemv, MatchesCpu)
{
    SKIP_WITHOUT_GPU();

    const Params p = GetParam();
    const auto m = quant::cpu::quantize(make_input(p), p.rows, p.cols, p.G);
    const auto v = m.view();
    const auto x = test_utils::make_x(p.cols);

    std::vector<float> ref(p.rows);
    quant::cpu::gemv(v, x, ref);

    for(const Kernel& k : kernels)
    {
        const auto y = run(k.fn, m, x);
        for(std::size_t i = 0; i < p.rows; i++)
        {
            double abs_sum = 0.0;
            for(std::size_t j = 0; j < p.cols; j++)
                abs_sum += std::abs(static_cast<double>(dequantize_at(v, i, j))
                                    * quant::cpu::fp16_bits_to_float(x[j]));

            // fp32 accumulation; NaN fails this too.
            ASSERT_NEAR(y[i], ref[i], 1e-5 * abs_sum) << k.name << " i=" << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, GpuGemv,
                         ::testing::Values(Params{7, 1000, 8, 0.0f, 1.0f},       // padding, few rows
                                           Params{300, 1024, 32, 0.0f, 0.02f},   // rows not a multiple of the block
                                           Params{16, 4096, 32, 0.0f, 0.02f},
                                           Params{5, 1024, 64, 3.0f, 0.1f},      // positive only
                                           Params{5, 1024, 128, -3.0f, 0.1f}));  // negative only

TEST(GpuGemvBasic, OneHotSelectsColumn)
{
    SKIP_WITHOUT_GPU();

    const Params p{3, 80, 16, 0.0f, 1.0f};
    const auto m = quant::cpu::quantize(make_input(p), p.rows, p.cols, p.G);
    const auto v = m.view();

    for(const Kernel& k : kernels)
        for(std::size_t j = 0; j < p.cols; j++)
        {
            std::vector<std::uint16_t> x(p.cols, 0);
            x[j] = 0x3C00;

            const auto y = run(k.fn, m, x);
            for(std::size_t i = 0; i < p.rows; i++)
                ASSERT_FLOAT_EQ(y[i], dequantize_at(v, i, j)) << k.name << " i=" << i << " j=" << j;
        }
}

TEST(GpuGemvBasic, RejectsWrongSizes)
{
    SKIP_WITHOUT_GPU();

    const Params p{3, 80, 16, 0.0f, 1.0f};
    const CudaMatrixQuant m(quant::cpu::quantize(make_input(p), p.rows, p.cols, p.G));
    DeviceBuffer<std::uint16_t> x(p.cols), bad_x(p.cols - 1);
    DeviceBuffer<float> y(p.rows), bad_y(p.rows + 1);

    for(const Kernel& k : kernels)
    {
        EXPECT_THROW(k.fn(m, bad_x, y), std::invalid_argument) << k.name;
        EXPECT_THROW(k.fn(m, x, bad_y), std::invalid_argument) << k.name;
    }
}
