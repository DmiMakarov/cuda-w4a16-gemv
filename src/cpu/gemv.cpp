#include <span>
#include <cstdint>

#include "cpu/matrix_quant_view.hpp"

void quant::cpu::gemv(const MatrixQuantView& m, std::span<const std::uint16_t> x, std::span<float> y)
{}
