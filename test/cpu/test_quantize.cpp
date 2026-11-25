#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "cpu/float_fp16.hpp"
#include "cpu/quantize.hpp"
#include "test_utils.hpp"

using quant::cpu::fp16_bits_to_float;
using quant::cpu::quantize;

namespace
{

using test_utils::Params;
using test_utils::dequantize_at;
using test_utils::make_input;
using test_utils::scale_at;

class QuantizeRandom : public ::testing::TestWithParam<Params> {};

} // namespace

// Worked by hand: pins down nibble order, scale and zero-point.
TEST(Quantize, HandComputedExample)
{
    std::vector<float> a(16);
    for(std::size_t j = 0; j < a.size(); j++)
        a[j] = -1.0f + 0.2f * static_cast<float>(j);

    const auto m = quantize(a, 1, 16, 16);

    EXPECT_EQ(m.w_stride, 32u);
    EXPECT_EQ(m.s.at(0), 0x3266);
    EXPECT_EQ(m.z.at(0), 5);
    EXPECT_EQ(m.w.at(0), 0x76543210u);
    EXPECT_EQ(m.w.at(1), 0xFEDCBA98u);
}

TEST_P(QuantizeRandom, ShapeAndPadding)
{
    const Params p = GetParam();
    const auto m = quantize(make_input(p), p.rows, p.cols, p.G);

    const std::size_t rows = p.rows;
    EXPECT_EQ(m.rows, p.rows);
    EXPECT_EQ(m.cols, p.cols);
    EXPECT_EQ(m.G, p.G);
    EXPECT_EQ(m.w_stride % 32, 0u);
    EXPECT_GE(m.w_stride, p.cols / 8);
    EXPECT_LT(m.w_stride, p.cols / 8 + 32);
    ASSERT_EQ(m.w.size(), rows * m.w_stride);
    ASSERT_EQ(m.s.size(), rows * (p.cols / p.G));
    ASSERT_EQ(m.z.size(), rows * (p.cols / p.G));

    for(const auto z : m.z)
        ASSERT_LE(z, 15);

    for(std::size_t i = 0; i < rows; i++)
        for(std::size_t k = p.cols / 8; k < m.w_stride; k++)
            ASSERT_EQ(m.w[i * m.w_stride + k], 0u) << "i=" << i << " k=" << k;
}

TEST_P(QuantizeRandom, ErrorIsAboutHalfScale)
{
    const Params p = GetParam();
    const auto a = make_input(p);
    const auto m = quantize(a, p.rows, p.cols, p.G);
    const auto v = m.view();

    // 0.5 from rounding, the rest covers fp16 scale rounding and clamping.
    for(std::size_t i = 0; i < p.rows; i++)
        for(std::size_t j = 0; j < p.cols; j++)
            ASSERT_LE(std::fabs(a[i * p.cols + j] - dequantize_at(v, i, j)), 0.6f * scale_at(v, i, j))
                << "i=" << i << " j=" << j;
}

INSTANTIATE_TEST_SUITE_P(Shapes, QuantizeRandom,
                         ::testing::Values(Params{7, 1000, 8, 0.0f, 1.0f},       // padding
                                           Params{16, 4096, 32, 0.0f, 0.02f},    // typical weights
                                           Params{5, 1024, 64, 3.0f, 0.1f},      // positive only
                                           Params{5, 1024, 128, -3.0f, 0.1f}));  // negative only

TEST(Quantize, GroupsAreIndependent)
{
    // Two rows of two groups; group n holds 0..7 times 10^n.
    const float mult[] = {1.0f, 10.0f, 100.0f, 1000.0f};
    std::vector<float> a;
    for(const float f : mult)
        for(int j = 0; j < 8; j++)
            a.push_back(f * static_cast<float>(j));

    const auto m = quantize(a, 2, 16, 8);

    ASSERT_EQ(m.s.size(), 4u);
    for(std::size_t n = 0; n < 4; n++)
    {
        const float expected = 7.0f * mult[n] / 15.0f;
        EXPECT_NEAR(fp16_bits_to_float(m.s[n]), expected, 1e-3f * expected) << "group " << n;
    }
}

TEST(Quantize, ZeroGroup)
{
    const std::vector<float> a(32, 0.0f);
    const auto m = quantize(a, 1, 32, 32);

    EXPECT_EQ(m.s.at(0), 0x3C00);
    EXPECT_EQ(m.z.at(0), 0);
    for(const auto word : m.w)
        EXPECT_EQ(word, 0u);
}

TEST(Quantize, ConstantGroups)
{
    const auto pos = quantize(std::vector<float>(8, 3.0f), 1, 8, 8);
    EXPECT_EQ(pos.z.at(0), 0);
    EXPECT_EQ(pos.w.at(0), 0xFFFFFFFFu);

    const auto neg = quantize(std::vector<float>(8, -3.0f), 1, 8, 8);
    EXPECT_EQ(neg.z.at(0), 15);
    EXPECT_EQ(neg.w.at(0), 0u);
}

TEST(Quantize, ZeroIsExact)
{
    const std::vector<float> a{-0.7f, 0.0f, 1.3f, 0.2f, -0.1f, 0.0f, 0.9f, 0.5f};
    const auto m = quantize(a, 1, 8, 8);

    EXPECT_EQ(dequantize_at(m.view(), 0, 1), 0.0f);
    EXPECT_EQ(dequantize_at(m.view(), 0, 5), 0.0f);
}

TEST(Quantize, RejectsBadArguments)
{
    const std::vector<float> a(32, 1.0f);

    EXPECT_THROW(quantize(a, 1, 32, 0), std::invalid_argument);
    EXPECT_THROW(quantize(a, 1, 32, 24), std::invalid_argument);  // cols % G != 0
    EXPECT_THROW(quantize(a, 1, 32, 4), std::invalid_argument);   // G % 8 != 0
    EXPECT_THROW(quantize(a, 2, 32, 32), std::invalid_argument);  // wrong a.size()
}

TEST(Quantize, RejectsScaleOverflow)
{
    const std::vector<float> a(32, 1e7f);
    EXPECT_THROW(quantize(a, 1, 32, 32), std::runtime_error);
}

// Exact scale is 1.6 * 2^-24, which fp16 stores as 2 * 2^-24.
// Codes must be computed with the stored scale: 24 / 2 = 12, not 24 / 1.6 = 15.
TEST(Quantize, UsesStoredScale)
{
    const float unit = std::ldexp(1.0f, -24);
    const auto m = quantize(std::vector<float>(8, 24.0f * unit), 1, 8, 8);

    EXPECT_EQ(m.s.at(0), 0x0002);
    EXPECT_EQ(m.w.at(0), 0xCCCCCCCCu);
}
