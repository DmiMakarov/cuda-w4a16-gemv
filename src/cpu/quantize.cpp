#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <format>
#include <span>
#include <algorithm>
#include <cmath>
#include <utility>

#include "cpu/matrix_quant.hpp"
#include "cpu/quantize.hpp"
#include "cpu/float_fp16.hpp"

quant::cpu::MatrixQuant quant::cpu::quantize(std::span<const float> a, std::uint32_t rows, std::uint32_t cols, std::uint32_t G){
    
    if(G == 0)
        throw std::invalid_argument("G is equal to 0");
    if((cols % G )!= 0 )
        throw std::invalid_argument(std::format("Cols can not be divided by G without remain: cols = {}, G = {}", cols, G));
    if((G % 8) != 0 )
        throw std::invalid_argument(std::format("8 must be divisor for the G: G = {}", G));
    
    std::size_t rows_s = static_cast<std::size_t>(rows);
    
    if(a.size() != (rows_s * cols))
        throw std::invalid_argument(std::format("Len of a doesnt match with rows and cols: len(a) = {}, rows = {}, cols = {}", a.size(), rows, cols));

    //padding for memory utilization optimization
    std::uint32_t w_stride = cols / 8;
    w_stride = (w_stride + 31) / 32 * 32;

    std::size_t cols_g = cols / G;
    std::size_t g_8 = G / 8;
    std::vector<std::uint32_t> w(rows_s * w_stride, 0);
    std::vector<std::uint16_t> s(rows_s * cols_g);
    std::vector<std::uint8_t> z(rows_s * cols_g);


    for(std::size_t i = 0; i < rows_s; i++)
    {
        for(std::size_t g = 0; g < cols_g; g++)
        {
            auto group = a.subspan(i * cols + g * G, G);
            auto [mn, mx] = std::ranges::minmax(group);
            mn = std::min(mn, 0.0f);
            mx = std::max(mx, 0.0f);
            float scale = (mx - mn) / 15;
            s[i * cols_g + g] = float_to_fp16_bits(scale);
            scale = fp16_bits_to_float(s[i * cols_g + g]);

            if(s[i * cols_g + g] == 0)
            {
                s[i * cols_g + g] = 0x3C00;
                scale = 1.0f;
            }

            if(s[i * cols_g + g] == 0x7C00) throw std::runtime_error("Get infinite scale!");

            const float zp = std::clamp(std::round(-mn / scale), 0.0f, 15.0f);
            z[i * cols_g + g] = static_cast<std::uint8_t>(zp);
        

            for(std::size_t k = 0; k < g_8; k++)
            {
                for(std::size_t p = 0; p < 8; p++)
                {
                    const float v = std::round(group[8 * k + p] / scale) + zp;
                    const auto q = static_cast<std::uint32_t>(std::clamp(v, 0.0f, 15.0f));
                    w[i * w_stride + g * g_8 + k] |= q << (4 * p);
                }   
            }
        }
    }

    return MatrixQuant{ .w=std::move(w),
                        .s=std::move(s),
                        .z=std::move(z),
                        .rows=rows,
                        .cols=cols,
                        .G=G,
                        .w_stride=w_stride};

}