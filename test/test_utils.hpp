#ifndef QUANT_TEST_UTILS_HPP
#define QUANT_TEST_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "cpu/float_fp16.hpp"
#include "matrix_quant_view.hpp"

// Independent, per-element access to the packed format: deliberately not shared with src/.
namespace test_utils
{

using quant::MatrixQuantView;

inline std::size_t group_index(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    return i * (m.cols / m.G) + j / m.G;
}

inline std::uint32_t code_at(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    return (m.w[i * m.w_stride + j / 8] >> (4 * (j % 8))) & 0xFu;
}

inline float scale_at(const MatrixQuantView& m, std::size_t i, std::size_t j)
{
    return quant::cpu::fp16_bits_to_float(m.s[group_index(m, i, j)]);
}

inline float dequantize_at(const MatrixQuantView& m, std::size_t i, std::size_t j)
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

inline std::vector<float> make_input(const Params& p)
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(p.mean, p.sd);
    std::vector<float> a(static_cast<std::size_t>(p.rows) * p.cols);
    for(float& v : a)
        v = dist(rng);
    return a;
}

// Not the seed of make_input, otherwise x would repeat row 0.
inline std::vector<std::uint16_t> make_x(std::uint32_t cols)
{
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::uint16_t> x(cols);
    for(auto& v : x)
        v = quant::cpu::float_to_fp16_bits(dist(rng));
    return x;
}

} // namespace test_utils

#endif // QUANT_TEST_UTILS_HPP
