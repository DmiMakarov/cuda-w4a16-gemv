#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/device_buffer.hpp"
#include "gpu/gpu_test.hpp"

using quant::gpu::DeviceBuffer;

static_assert(!std::is_copy_constructible_v<DeviceBuffer<float>>);
static_assert(!std::is_copy_assignable_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<float>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer<float>>);

TEST(DeviceBuffer, RoundTrip)
{
    SKIP_WITHOUT_GPU();

    const std::vector<std::uint32_t> src{1, 2, 3, 0xFFFFFFFFu, 5};
    DeviceBuffer<std::uint32_t> buf(src.size());
    ASSERT_EQ(buf.size(), src.size());

    buf.copy_from_host(src);
    std::vector<std::uint32_t> dst(src.size());
    buf.copy_to_host(dst);

    EXPECT_EQ(dst, src);
}

TEST(DeviceBuffer, RejectsWrongSizes)
{
    SKIP_WITHOUT_GPU();

    DeviceBuffer<float> buf(8);
    std::vector<float> small(7), big(9);

    EXPECT_THROW(buf.copy_from_host(small), std::invalid_argument);
    EXPECT_THROW(buf.copy_from_host(big), std::invalid_argument);
    EXPECT_THROW(buf.copy_to_host(small), std::invalid_argument);
    EXPECT_THROW(buf.copy_to_host(big), std::invalid_argument);
}

TEST(DeviceBuffer, MoveTransfersOwnership)
{
    SKIP_WITHOUT_GPU();

    DeviceBuffer<float> a(8);
    float* const ptr = a.data();

    DeviceBuffer<float> b(std::move(a));
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(b.size(), 8u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);

    DeviceBuffer<float> c(4);
    c = std::move(b);
    EXPECT_EQ(c.data(), ptr);
    EXPECT_EQ(c.size(), 8u);
    EXPECT_EQ(b.data(), nullptr);
}
