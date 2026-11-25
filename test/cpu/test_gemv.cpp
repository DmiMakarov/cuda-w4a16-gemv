#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "cpu/gemv.hpp"
#include "cpu/float_fp16.hpp"
#include "cpu/matrix_quant.hpp"
#include "matrix_quant_view.hpp"
#include "cpu/quantize.hpp"
#include "test_utils.hpp"

using quant::cpu::fp16_bits_to_float;

namespace
{

using test_utils::Params;
using test_utils::dequantize_at;
using test_utils::make_input;

class GEMV : public ::testing::TestWithParam<Params> {};

} //namespace

TEST_P(GEMV, MatchesDequantizedReference)
{
    const Params p = GetParam();
    
    std::vector<float> a = make_input(p);
    const auto m = quant::cpu::quantize(a, p.rows, p.cols, p.G);
    const auto v = m.view();

    std::vector<float> y(p.rows);
    const auto x = test_utils::make_x(p.cols);

    quant::cpu::gemv(v, x, y);
    
    for(std::size_t i = 0; i < m.rows; i++)
    {
        double ref = 0.0;
        double abs_sum = 0.0;

        for(std::size_t j = 0; j < m.cols; j++)
        {
            const double term = static_cast<double>(dequantize_at(v, i, j)) * fp16_bits_to_float(x[j]);
            ref += term;
            abs_sum += std::abs(term);
        }

        EXPECT_NEAR(ref, y[i], 1e-6 * abs_sum) << "i=" << i;
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, GEMV, 
    ::testing::Values(Params{.rows=7, .cols=1000, .G=8, .mean=3.0f, .sd=1.0f},
                      Params{16, 4096, 32, 0.0f, 0.02f},
                      Params{5, 1024, 64, 3.0f, 0.1f},
                      Params{5, 1024, 128, -3.0f, 0.1f}));

// Codes are 0..15, z = 5, so with x = 1: y = s * (120 - 16 * 5) = 40 * s.
TEST(GemvBasic, HandComputedExample)
{
    std::vector<float> a(16);
    for(std::size_t j = 0; j < a.size(); j++)
        a[j] = -1.0f + 0.2f * static_cast<float>(j);

    const auto m = quant::cpu::quantize(a, 1, 16, 16);
    const std::vector<std::uint16_t> x(16, 0x3C00);
    std::vector<float> y(1);

    quant::cpu::gemv(m.view(), x, y);

    EXPECT_FLOAT_EQ(y[0], 40.0f * fp16_bits_to_float(0x3266));
}

// x = e_j must pick column j: catches any mix-up in the indices of w, s, z or x.
TEST(GemvBasic, OneHotSelectsColumn)
{
    const Params p{3, 80, 16, 0.0f, 1.0f};
    const auto m = quant::cpu::quantize(make_input(p), p.rows, p.cols, p.G);
    const auto v = m.view();
    std::vector<float> y(p.rows);

    for(std::size_t j = 0; j < p.cols; j++)
    {
        std::vector<std::uint16_t> x(p.cols, 0);
        x[j] = 0x3C00;

        quant::cpu::gemv(v, x, y);

        for(std::size_t i = 0; i < p.rows; i++)
            ASSERT_EQ(y[i], dequantize_at(v, i, j)) << "i=" << i << " j=" << j;
    }
}

TEST(GemvBasic, OverwritesOutput)
{
    const Params p{3, 80, 16, 0.0f, 1.0f};
    const auto m = quant::cpu::quantize(make_input(p), p.rows, p.cols, p.G);
    const std::vector<std::uint16_t> x(p.cols, 0);
    std::vector<float> y(p.rows, 123.0f);

    quant::cpu::gemv(m.view(), x, y);

    for(const float value : y)
        EXPECT_EQ(value, 0.0f);
}

TEST(GemvBasic, RejectsWrongSizes)
{
    const Params p{3, 80, 16, 0.0f, 1.0f};
    const auto m = quant::cpu::quantize(make_input(p), p.rows, p.cols, p.G);
    const auto v = m.view();
    const std::vector<std::uint16_t> x(p.cols), bad_x(p.cols - 1);
    std::vector<float> y(p.rows), bad_y(p.rows + 1);

    EXPECT_THROW(quant::cpu::gemv(v, bad_x, y), std::invalid_argument);
    EXPECT_THROW(quant::cpu::gemv(v, x, bad_y), std::invalid_argument);
}
