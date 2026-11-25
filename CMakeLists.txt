cmake_minimum_required(VERSION 3.25)
project(cuda-w4a16-gemv CXX)

option(QUANT_CUDA "Build CUDA kernels" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED)

add_library(quant STATIC
            src/cpu/float_fp16.cpp
            src/cpu/quantize.cpp
            src/cpu/gemv.cpp)
target_include_directories(quant PUBLIC include)
target_compile_options(quant PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(QUANT_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(quant_gpu STATIC 
                    src/gpu/gemv_naive.cu
                    src/gpu/gemv_warp.cu
                    src/gpu/cuda_matrix_quant.cpp)
    target_link_libraries(quant_gpu PUBLIC quant CUDA::cudart)
endif()

enable_testing()
add_executable(tests
               test/cpu/test_float_fp16.cpp
               test/cpu/test_matrix_quant.cpp
               test/cpu/test_quantize.cpp
               test/cpu/test_gemv.cpp)
target_include_directories(tests PRIVATE test)
target_link_libraries(tests PRIVATE quant GTest::gtest_main)
target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME tests COMMAND tests)

if(QUANT_CUDA)
    add_executable(tests_gpu
                   test/gpu/test_device_buffer.cpp
                   test/gpu/test_gemv.cpp)
    target_include_directories(tests_gpu PRIVATE test)
    target_link_libraries(tests_gpu PRIVATE quant_gpu GTest::gtest_main)
    add_test(NAME tests_gpu COMMAND tests_gpu)

    find_package(benchmark REQUIRED)
    add_executable(bench_gemv bench/bench_gemv.cpp)
    target_link_libraries(bench_gemv PRIVATE quant_gpu CUDA::cublas benchmark::benchmark)
endif()

#add_executable(main
#               src/main.cpp)
#target_link_libraries(main PRIVATE quant)


