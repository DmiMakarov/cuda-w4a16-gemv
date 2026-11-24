#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "cpu/matrix_quant.hpp"
#include "matrix_quant_view.hpp"

using quant::MatrixQuantView;
using quant::cpu::MatrixQuant;

template <typename T>
concept HasView = requires(T&& m) { std::forward<T>(m).view(); };

// The view is passed to the kernel by value.
static_assert(std::is_trivially_copyable_v<MatrixQuantView>);

// A view of a temporary would dangle.
static_assert(HasView<MatrixQuant&>);
static_assert(HasView<const MatrixQuant&>);
static_assert(!HasView<MatrixQuant>);
static_assert(!HasView<const MatrixQuant>);

TEST(MatrixQuant, ViewMatchesOwner)
{
    const MatrixQuant m{.w = {1, 2},
                        .s = {3},
                        .z = {4},
                        .rows = 5,
                        .cols = 6,
                        .G = 7,
                        .w_stride = 8};

    const MatrixQuantView v = m.view();

    EXPECT_EQ(v.w, m.w.data());
    EXPECT_EQ(v.s, m.s.data());
    EXPECT_EQ(v.z, m.z.data());
    EXPECT_EQ(v.rows, 5u);
    EXPECT_EQ(v.cols, 6u);
    EXPECT_EQ(v.G, 7u);
    EXPECT_EQ(v.w_stride, 8u);
}
