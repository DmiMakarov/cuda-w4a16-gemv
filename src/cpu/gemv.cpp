#include <span>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <format>

#include "matrix_quant_view.hpp"
#include "cpu/float_fp16.hpp"
#include "cpu/gemv.hpp"

void quant::cpu::gemv(const MatrixQuantView& m, std::span<const std::uint16_t> x, std::span<float> y)
{
    if(x.size() != m.cols) throw std::invalid_argument(std::format("In cpu::gemv invalid size of x:  x.size() = {}, matrix.cols  = {}", x.size(), m.cols));
    if(y.size() != m.rows) throw std::invalid_argument(std::format("In cpu::gemv invalid size of y:  y.size() = {}, matrix.rows  = {}", y.size(), m.rows));

    std::size_t cols_g = m.cols / m.G;
    std::size_t g_8 = m.G / 8;

    for(std::size_t i = 0; i < m.rows; i++)
    {
        double acc = 0.0;

        for(std::size_t g = 0; g < cols_g; g++)
        {
            float scale = fp16_bits_to_float(m.s[i * cols_g + g]);
            float z = m.z[i * cols_g + g];
            double sum = 0;

            for(std::size_t k = 0; k < g_8; k++)
            {
                const auto word = m.w[i * m.w_stride + g * g_8 + k]; //here 8 numbers

                for(std::size_t p = 0; p < 8; p++)
                {
                    const auto q = (word >> (4 * p)) & 0xFu;

                    sum += static_cast<double>((q - z)) * fp16_bits_to_float(x[g * m.G + 8 * k + p]);
                }
            }

            acc += scale * sum;
        }

        y[i] = static_cast<float>(acc);
    }
}
