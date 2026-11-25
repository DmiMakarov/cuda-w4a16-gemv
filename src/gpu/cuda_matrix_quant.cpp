#include  <cstdint>

#include "gpu/device_buffer.hpp"
#include "matrix_quant_view.hpp"
#include "cpu/matrix_quant.hpp"
#include "gpu/cuda_matrix_quant.hpp"

quant::gpu::CudaMatrixQuant::CudaMatrixQuant(const quant::cpu::MatrixQuant& m): w_(m.w.size()), s_(m.s.size()), z_(m.z.size()),
                                                                        rows_(m.rows), cols_(m.cols), G_(m.G),
                                                                        w_stride_(m.w_stride)
{
    w_.copy_from_host(m.w);
    s_.copy_from_host(m.s);
    z_.copy_from_host(m.z);
}

quant::MatrixQuantView quant::gpu::CudaMatrixQuant::view() const& 
{
    return MatrixQuantView{.w=w_.data(),
                           .s=s_.data(),
                           .z=z_.data(),
                           .rows=rows_,
                           .cols=cols_,
                           .G=G_,
                           .w_stride=w_stride_};
}