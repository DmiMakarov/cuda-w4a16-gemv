#include <vector>
#include <random>
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

using quant::MatrixQuantView;
using quant::cpu::fp16_bits_to_float;
using quant::cpu::float_to_fp16_bits;

namespace
{

std::size_t group_index(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    return i * (m.cols / m.G) + j / m.G;
}

std::uint32_t code_at(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    return (m.w[i * m.w_stride + j / 8] >> (4 * (j % 8))) & 0xFu;
}

float scale_at(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    return fp16_bits_to_float(m.s[group_index(m, i, j)]);
}

float dequantize_at(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    const int q = static_cast<int>(code_at(m, i, j));
    const int z = m.z[group_index(m, i, j)];
    return scale_at(m, i, j) * static_cast<float>(q - z);
}

struct Params
{
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t G;
    float mean;
    float sd;
};

std::vector<float> make_input(const Params& p)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(p.mean, p.sd);
    std::vector<float> a(static_cast<std::size_t>(p.rows) * p.cols);
    for(float& v : a)
        v = dist(rng);
    return a;
}

class GEMV : public ::testing::TestWithParam<Params> {};

} //namespace

TEST_P(GEMV, MatchesDequantizedReference)
{
    const Params p = GetParam();
    
    std::vector<float> a = make_input(p);
    const auto m = quant::cpu::quantize(a, p.rows, p.cols, p.G);
    const auto v = m.view();

    std::vector<float> y(p.rows);
    std::vector<std::uint16_t> x(p.cols);
    std::mt19937 rng(7);  // not the seed of make_input, otherwise x repeats row 0
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for(std::size_t i = 0; i < p.cols; i++)
    {
        x[i] = float_to_fp16_bits(dist(rng));
    }

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
