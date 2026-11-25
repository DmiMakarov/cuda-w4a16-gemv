#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <cuda_fp16.h>

#include "gpu/gemv_warp_xvec.hpp"
#include "gpu/device_buffer.hpp"
#include "gpu/cuda_matrix_quant.hpp"


namespace quant::gpu
{

namespace
{

__global__
void gemv_warp_xvec_kernel(quant::MatrixQuantView m, const std::uint16_t* __restrict__ x, float* __restrict__ y)
{    
    const std::uint32_t lane = threadIdx.x % 32; //index of thread inside warp
    const std::uint32_t warp = threadIdx.x / 32;
    const std::uint32_t i = blockIdx.x * (blockDim.x / 32 ) + warp; //line i that is common for the warp;

    const std::uint32_t cols_g = m.cols / m.G;
    const std::uint32_t g_8 = m.G / 8;
    const std::uint32_t words = m.cols / 8;

    if(i < m.rows)
    {
        const __half* __restrict__ sh = reinterpret_cast<const __half*>(m.s);
        const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);

        float acc = 0;

        for(std::uint32_t k = lane; k < words; k += 32)
        {
            float sum = 0.0f;

            const std::uint32_t g = k / g_8;
            const auto word = m.w[static_cast<std::size_t>(i) * m.w_stride + k];
            const float scale = __half2float(sh[i * cols_g + g]);
            const float z = static_cast<float>(m.z[i * cols_g + g]);

            const uint4 x8 = xv[k];
            const std::uint32_t xw[4] = {x8.x, x8.y, x8.z, x8.w};

            for(std::uint32_t p = 0; p < 8; p++)
            {
                const auto q = (word >> (4 * p)) & 0xFu;
                const auto bits = static_cast<unsigned short>(xw[p / 2] >> (16 * (p % 2)));
                sum += (q - z) * __half2float(__ushort_as_half(bits));
            }

            acc += scale * sum;
        }

        for(int shift = 16; shift > 0; shift /= 2)
            acc += __shfl_down_sync(0xffffffffu, acc, shift);

        if(lane == 0)
            y[i] = acc;
    }
}
}

void gemv_warp_xvec(const CudaMatrixQuant& m, const DeviceBuffer<std::uint16_t>& x, DeviceBuffer<float>& y)
{
    //all structures already allocated and moved to Device
    const auto v = m.view();

    if(x.size() != v.cols)
        throw std::invalid_argument(std::format("Sizes of matrix and x doesnt match in gemv_warp_xvec: matrix = [{}, {}], x = {}",
                                                v.rows, v.cols, x.size()));
    if(y.size() != v.rows)
        throw std::invalid_argument(std::format("Sizes of matrix and y doesnt match in gemv_warp_xvec: matrix = [{}, {}], y = {}",
                                                v.rows, v.cols, y.size()));

    constexpr std::uint32_t block_size = 128;
    constexpr std::uint32_t rows_per_block = block_size / 32;
    const dim3 grid((v.rows + rows_per_block  - 1) / rows_per_block); 
    const dim3 block(block_size);

    gemv_warp_xvec_kernel<<<grid, block>>>(v, x.data(), y.data());

    detail::check(cudaGetLastError(), "gemv_warp_xvec");
    //no need to free memory because we dont own them
}

} //namespace quant::gpu