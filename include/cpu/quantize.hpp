#ifndef QUANT_CPU_QUANTIZE
#define QUANT_CPU_QUANTIZE


#include <cstdint>
#include <span>

#include "cpu/matrix_quant.hpp"

namespace quant::cpu
{

MatrixQuant quantize(std::span<const float> a, std::uint32_t rows, std::uint32_t cols, std::uint32_t G);

} //namespace quant::cpu

#endif //QUANT_CPU_QUANTIZE
