#ifndef CPU_FLOAT_FP16_HPP
#define CPU_FLOAT_FP16_HPP

#include<cstdint>

namespace quant::cpu
{

std::uint16_t float_to_fp16_bits(float s);
float fp16_bits_to_float(std::uint16_t h);

} //quant::cpu

#endif //CPU_FLOAT_FP16_HPP
