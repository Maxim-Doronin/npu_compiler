// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <vpux/utils/core/type/float8_e8m0.hpp>

#include <array>
#include <cmath>
#include <limits>

static_assert(sizeof(vpux::type::float8_e8m0) == 1, "type f8e8m0 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<vpux::type::float8_e8m0, vpux::type::float8_e8m0>::value,
              "should be trivially constructible");
static_assert(std::is_trivially_copyable<vpux::type::float8_e8m0>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<vpux::type::float8_e8m0>::value, "must be trivially destructible");

constexpr uint8_t f32_m_size{23u};                              // f32 mantissa bits size
constexpr uint32_t f32_e_mask{0x7F800000u};                     // f32 exponent bits mask
constexpr uint32_t f32_m_mask{0x007FFFFFu};                     // f32 mantissa bits mask
constexpr uint32_t f32_m_round_even_up{0x00400000u};            // f32 mantissa round even up
constexpr uint32_t f32_m_round_even_up_subnormal{0x00600000u};  // f32 mantissa round even up for subnormal values

/**
 * @brief Converts a 32-bit float value to its corresponding 8-bit f8e8m0 representation.
 *
 * The f8e8m0 format uses:
 * - no sign bit
 * - 8 exponent bits (with bias 127)
 * - 0 mantissa bits
 *
 * @param value The 32-bit float value to convert.
 * @return The 8-bit f8e8m0 representation as a uint8_t.
 */
uint8_t f32_to_f8e8m0_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint8_t f32_exp = static_cast<uint8_t>((bits & f32_e_mask) >> f32_m_size);  // Extract exponent
    const uint32_t f32_mantissa = bits & f32_m_mask;                                  // Extract mantissa

    if (std::signbit(value)) {
        // negative values
        return 0b00000000;
    } else if (f32_exp >= 0b11111110) {
        // infinity, NaNs and normal values >= 2^127
        return f32_exp - static_cast<uint8_t>(std::isinf(value));
    } else if (f32_exp == 0b00000000 && f32_mantissa <= f32_m_round_even_up_subnormal) {
        // zero and subnormal values with significand <= 0.75 (ties to even for 0.75)
        return 0b00000000;
    } else {
        // normal values and subnormal values with significand (same as mantissa in this context) > 0.75
        return f32_exp + static_cast<uint8_t>((f32_mantissa > f32_m_round_even_up) ||    // round to nearest
                                              ((f32_mantissa == f32_m_round_even_up) &&  // ties to even
                                               (f32_exp & 0x1)));
    }
}

vpux::type::float8_e8m0::float8_e8m0(const float value): m_value(f32_to_f8e8m0_bits(value)) {};

vpux::type::float8_e8m0::operator float() const {
    constexpr auto f8e8m0_2_power_negative_127 = std::numeric_limits<vpux::type::float8_e8m0>::min();
    constexpr auto float_2_power_negative_127 = std::numeric_limits<float>::min() / 2;

    if (to_bits() == std::numeric_limits<vpux::type::float8_e8m0>::quiet_NaN().to_bits()) {
        // NaN
        return std::numeric_limits<float>::quiet_NaN();
    } else if (to_bits() == f8e8m0_2_power_negative_127.to_bits()) {
        // we need a denormalized fp32 value to represent 2^(-127); as the implicit exponent is -126, we need to have a
        // significand (and fraction/mantissa) equal to 0.5
        return float_2_power_negative_127;
    } else {
        // remaining values can be represented just by shifting the exponent (both types have 8 bits for it);
        // significand will be 1.0 (i.e. fraction/mantissa 0.0)
        uint32_t bits = m_value << f32_m_size;
        float value;
        std::memcpy(&value, &bits, sizeof(bits));
        return value;
    }
}

uint8_t vpux::type::float8_e8m0::to_bits() const {
    return m_value;
}
