#ifndef GPU_CUDA_MATRIX_QUANT_HPP
#define GPU_CUDA_MATRIX_QUANT_HPP

#include  <cstdint>

#include "matrix_quant_view.hpp"
#include "cpu/matrix_quant.hpp"
#include "gpu/device_buffer.hpp"

namespace quant::gpu
{

class CudaMatrixQuant
{
public:
    explicit CudaMatrixQuant(const quant::cpu::MatrixQuant& m);
    
    [[nodiscard]] quant::MatrixQuantView view() const&;
    MatrixQuantView view() const && = delete;

private:
    DeviceBuffer<std::uint32_t> w_;
    DeviceBuffer<std::uint16_t> s_;
    DeviceBuffer<std::uint8_t> z_;
    
    std::uint32_t rows_, cols_, G_, w_stride_;
};

} //namespace quant::gpu

#endif //GPU_CUDA_MATRIX_QUANT_HPP