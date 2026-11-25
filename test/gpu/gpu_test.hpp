#ifndef QUANT_GPU_TEST_HPP
#define QUANT_GPU_TEST_HPP

#include <cuda_runtime.h>
#include <gtest/gtest.h>

// Skips (not fails) on machines without a GPU. On the GPU machine check that nothing was skipped.
#define SKIP_WITHOUT_GPU()                                                      \
    do                                                                          \
    {                                                                           \
        int count = 0;                                                          \
        if(cudaGetDeviceCount(&count) != cudaSuccess || count == 0)             \
            GTEST_SKIP() << "no CUDA device";                                   \
    } while(false)

#endif // QUANT_GPU_TEST_HPP
