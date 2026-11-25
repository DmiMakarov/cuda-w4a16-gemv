#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <cuda_fp16.h>

#include "gpu/gemv_naive.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/cuda_matrix_quant.hpp"


namespace quant::gpu
{

namespace
{

__global__
void gemv_naive_kernel(quant::MatrixQuantView m, const std::uint16_t* x, float* y)
{    
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t cols_g = m.cols / m.G;
    std::uint32_t g_8 = m.G / 8;

    if(i < m.rows)
    {     
        const __half* xh = reinterpret_cast<const __half*>(x);
        const __half* sh = reinterpret_cast<const __half*>(m.s);
        float acc = 0.0;

        for( std::uint32_t g = 0; g < cols_g; g++)
        {
            const float scale = __half2float(sh[i * cols_g + g]);
            const float z = static_cast<float>(m.z[i * cols_g + g]);
            float sum = 0;

            for( std::uint32_t k = 0; k < g_8; k++)
            {
                const auto word = m.w[static_cast<std::size_t>(i) * m.w_stride + g * g_8 + k];
                
                for(std::uint32_t p = 0; p < 8; p++)
                {
                    const auto q = (word >> (4 * p)) & 0xFu;

                    sum += (q - z) * __half2float(xh[g * m.G + 8 * k + p]);
                }
            }

            acc += scale * sum;
        }

        y[i] = acc;
    }
}
}

void gemv_naive(const CudaMatrixQuant& m, const DeviceBuffer<std::uint16_t>& x, DeviceBuffer<float>& y)
{
    //all structures already allocated and moved to Device
    const auto v = m.view();

    if(x.size() != v.cols)
        throw std::invalid_argument(std::format("Sizes of matrix and x doesnt match in gemv_naive: matrix = [{}, {}], x = {}",
                                                v.rows, v.cols, x.size()));
    if(y.size() != v.rows)
        throw std::invalid_argument(std::format("Sizes of matrix and y doesnt match in gemv_naive: matrix = [{}, {}], y = {}",
                                                v.rows, v.cols, y.size()));

    std::size_t block_size = 128;
    const dim3 grid((y.size() + block_size - 1) / block_size); 
    const dim3 block(block_size);

    gemv_naive_kernel<<<grid, block>>>(v, x.data(), y.data());

    detail::check(cudaGetLastError(), "gemv_naive");
    //no need to free memory because we dont own them
}

} //namespace quant::gpu