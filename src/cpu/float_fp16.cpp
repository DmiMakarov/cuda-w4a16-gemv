#include <cstdint>
#include <bit>

#include "cpu/float_fp16.hpp"

std::uint16_t quant::cpu::float_to_fp16_bits(float s) {
    std::uint32_t x = std::bit_cast<std::uint32_t>(s);
    std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t exp  = (x >> 23) & 0xFFu;
    std::uint32_t mant = x & 0x007FFFFFu;

    // NaN или Infinity
    if (exp == 0xFFu) {
        if (mant != 0) {
            return static_cast<std::uint16_t>(sign | 0x7E00u | (mant >> 13)); // NaN
        }
        return static_cast<std::uint16_t>(sign | 0x7C00u); // Inf
    }

    std::int32_t new_exp = static_cast<std::int32_t>(exp) - 127 + 15;

    // Переполнение -> Infinity
    if (new_exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }

    // Денормализованные числа или ноль
    if (new_exp <= 0) {
        if (new_exp < -10) {
            return static_cast<std::uint16_t>(sign); // Слишком мало -> +-0
        }
        mant |= 0x00800000u; // Добавляем скрытую единицу
        std::uint32_t shift = 14 - new_exp;
        std::uint32_t sub_mant = mant >> shift;
        // Округление к ближайшему чётному (round-to-nearest-even)
        std::uint32_t rem = mant & ((1u << shift) - 1u);
        std::uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (sub_mant & 1u))) {
            sub_mant++;
        }
        return static_cast<std::uint16_t>(sign | sub_mant);
    }

    // Обычные нормализованные числа + округление к ближайшему чётному
    std::uint32_t h_mant = mant >> 13;
    std::uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h_mant & 1u))) {
        h_mant++;
        if (h_mant == 0x0400u) { // Переполнение мантиссы
            h_mant = 0;
            new_exp++;
            if (new_exp >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7C00u);
            }
        }
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(new_exp) << 10) | h_mant);
}

float quant::cpu::fp16_bits_to_float(std::uint16_t h) {
    std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp  = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x03FFu;

    std::uint32_t f_bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            // +0.0 или -0.0
            f_bits = sign;
        } else {
            // Денормализованное число fp16 -> нормализованное float32
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu; // Убираем скрытую единицу
            std::uint32_t f_exp  = (exp + 127 - 15) << 23;
            std::uint32_t f_mant = mant << 13;
            f_bits = sign | f_exp | f_mant;
        }
    } else if (exp == 0x1Fu) {
        // Infinity или NaN
        std::uint32_t f_exp  = 0xFFu << 23;
        std::uint32_t f_mant = mant << 13;
        // Сохраняем признак quiet NaN, если mant != 0
        if (mant != 0 && f_mant == 0) {
            f_mant = 0x00400000u; 
        }
        f_bits = sign | f_exp | f_mant;
    } else {
        // Обычное нормализованное число
        std::uint32_t f_exp  = (exp + 127 - 15) << 23;
        std::uint32_t f_mant = mant << 13;
        f_bits = sign | f_exp | f_mant;
    }

    return std::bit_cast<float>(f_bits);
}
