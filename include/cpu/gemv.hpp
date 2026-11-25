#ifndef QUANT_CPU_GEMV_HPP
#define QUANT_CPU_GEMV_HPP

#include <span>
#include <cstdint>

#include "matrix_quant_view.hpp"

namespace quant::cpu
{
    void gemv(const MatrixQuantView& m, std::span<const std::uint16_t> x, std::span<float> y);
} //namespace quant::cpu

#endif //QUANT_CPU_GEMV_HPP