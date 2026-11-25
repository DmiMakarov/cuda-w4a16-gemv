#ifndef QUANT_MATRIX_QUANT_VIEW
#define QUANT_MATRIX_QUANT_VIEW

#include <cstdint>

namespace quant
{

/*
MatrixQuantView - structure for quantized matrix with group factor G;

w - int4 quantized weights. w[i] - 8 weights of quantized matricies stored in one word
s - scale, s.size = rows * cols / G
z - zero-point, z.size = rows * cols / G

a ≈ s · (q − z), a - initial weights
*/

struct MatrixQuantView
{
    const std::uint32_t* w;
    const std::uint16_t* s;
    const std::uint8_t* z;

    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t G;
    std::uint32_t w_stride;
};

} //namespace quant

#endif //QUANT_MATRIX_QUANT_VIEW
