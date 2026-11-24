#ifndef QUANT_CPU_MATRIX_QUANT
#define QUANT_CPU_MATRIX_QUANT

#include <vector>
#include <cstdint>

#include "matrix_quant_view.hpp"

namespace quant::cpu
{

/*
MatrixQuant - structure for quantized matrix with group factor G;

w - int4 quantized weights. w[i] - 8 weights of quantized matricies stored in one word
s - scale, s.size = rows * cols / G
z - zero-point, z.size = rows * cols / G

a ≈ s · (q − z), a - initial weights
*/

struct MatrixQuant
{
    std::vector<std::uint32_t> w;
    std::vector<std::uint16_t> s;
    std::vector<std::uint8_t> z;
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t G;
    std::uint32_t w_stride;

    [[nodiscard]] MatrixQuantView view() const &
    {
        return MatrixQuantView{.w=w.data(),
                               .s=s.data(),
                               .z=z.data(),
                               .rows=rows,
                               .cols=cols,
                               .G=G,
                               .w_stride=w_stride};
    }

    MatrixQuantView view() const && = delete;
};

} //quant::cpu

#endif //QUANT_CPU_MATRIX_QUANT
