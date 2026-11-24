#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "cpu/float_fp16.hpp"

using quant::cpu::float_to_fp16_bits;
using quant::cpu::fp16_bits_to_float;

TEST(Fp16, ExactValuesBothWays)
{
    const struct { float value; std::uint16_t bits; } cases[] = {
        {0.0f, 0x0000},
        {1.0f, 0x3C00},
        {-2.0f, 0xC000},
        {65504.0f, 0x7BFF},               // largest finite fp16
        {std::ldexp(1.0f, -24), 0x0001},  // smallest subnormal fp16
    };

    for(const auto& c : cases)
    {
        EXPECT_EQ(float_to_fp16_bits(c.value), c.bits) << c.value;
        EXPECT_EQ(fp16_bits_to_float(c.bits), c.value) << c.bits;
    }
}

TEST(Fp16, RoundsToNearestEven)
{
    EXPECT_EQ(float_to_fp16_bits(0.2f), 0x3266);
    EXPECT_EQ(float_to_fp16_bits(1.0f + std::ldexp(1.0f, -11)), 0x3C00);  // tie, down to even
    EXPECT_EQ(float_to_fp16_bits(1.0f + std::ldexp(3.0f, -11)), 0x3C02);  // tie, up to even
}

TEST(Fp16, OverflowAndUnderflow)
{
    EXPECT_EQ(float_to_fp16_bits(65520.0f), 0x7C00);
    EXPECT_EQ(float_to_fp16_bits(1e-9f), 0x0000);
}

TEST(Fp16, RoundTripAllBitPatterns)
{
    for(std::uint32_t h = 0; h <= 0xFFFFu; h++)
    {
        const auto bits = static_cast<std::uint16_t>(h);
        const float value = fp16_bits_to_float(bits);
        const std::uint16_t back = float_to_fp16_bits(value);

        const bool is_nan = (h & 0x7FFFu) > 0x7C00u;
        ASSERT_EQ(std::isnan(value), is_nan) << h;
        if(is_nan)
            ASSERT_GT(back & 0x7FFFu, 0x7C00u) << h;
        else
            ASSERT_EQ(back, bits) << h;
    }
}
