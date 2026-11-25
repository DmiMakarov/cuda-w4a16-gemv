#ifndef QUANT_GPU_GEMV_NAIVE_HPP
#define QUANT_GPU_GEMV_NAIVE_HPP

#include <cstdint>

#include "gpu/cuda_matrix_quant.hpp"
#include "gpu/device_buffer.hpp"

namespace quant::gpu
{

void gemv_naive(const CudaMatrixQuant& m, const DeviceBuffer<std::uint16_t>& x, DeviceBuffer<float>& y);

} //namespace quant::gpu

#endif //QUANT_GPU_GEMV_NAIVE_HPP