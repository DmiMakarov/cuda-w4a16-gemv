cmake_minimum_required(VERSION 3.25)
project(cuda-w4a16-gemv CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTest REQUIRED)

add_library(quant STATIC
            src/cpu/float_fp16.cpp
            src/cpu/quantize.cpp)
target_include_directories(quant PUBLIC include)
target_compile_options(quant PUBLIC -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_executable(tests
               test/cpu/test_float_fp16.cpp
               test/cpu/test_matrix_quant.cpp
               test/cpu/test_quantize.cpp)
target_link_libraries(tests PRIVATE quant GTest::gtest_main)
add_test(NAME tests COMMAND tests)

#add_executable(main
#               src/main.cpp)
#target_link_libraries(main PRIVATE quant)


