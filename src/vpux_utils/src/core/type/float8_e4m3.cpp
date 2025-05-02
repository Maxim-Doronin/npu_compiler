// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vpux/utils/core/type/float16.hpp>
#include <vpux/utils/core/type/float8_e4m3.hpp>

#include <array>
#include <cmath>
#include <limits>

static_assert(sizeof(vpux::type::float8_e4m3) == 1, "class f8e4m3 must be exactly 1 byte");
static_assert(std::is_trivially_constructible<vpux::type::float8_e4m3, vpux::type::float8_e4m3>::value,
              "should be trivially constructible");
static_assert(std::is_trivially_copyable<vpux::type::float8_e4m3>::value, "must be trivially copyable");
static_assert(std::is_trivially_destructible<vpux::type::float8_e4m3>::value, "must be trivially destructible");
static_assert(std::numeric_limits<vpux::type::float8_e4m3>::is_specialized, "numeric_limits must be specialized");
static_assert(!std::numeric_limits<vpux::type::float8_e4m3>::is_integer, "numeric_limits::is_integer must be false");

constexpr auto float_nan = std::numeric_limits<float>::quiet_NaN();
// Lookup table for conversion f8 -> float. The f8 bit value without sign bit (masked 0x7f) is LUT offset.
static constexpr std::array<float, 128> f8_to_float_lut{
        0.0f,      0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
        0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
        0.03125f,  0.03515625f,  0.0390625f,  0.04296875f,  0.046875f,  0.05078125f,  0.0546875f,  0.05859375f,
        0.0625f,   0.0703125f,   0.078125f,   0.0859375f,   0.09375f,   0.1015625f,   0.109375f,   0.1171875f,
        0.125f,    0.140625f,    0.15625f,    0.171875f,    0.1875f,    0.203125f,    0.21875f,    0.234375f,
        0.25f,     0.28125f,     0.3125f,     0.34375f,     0.375f,     0.40625f,     0.4375f,     0.46875f,
        0.5f,      0.5625f,      0.625f,      0.6875f,      0.75f,      0.8125f,      0.875f,      0.9375f,
        1.0f,      1.125f,       1.25f,       1.375f,       1.5f,       1.625f,       1.75f,       1.875f,
        2.0f,      2.25f,        2.5f,        2.75f,        3.0f,       3.25f,        3.5f,        3.75f,
        4.0f,      4.5f,         5.0f,        5.5f,         6.0f,       6.5f,         7.0f,        7.5f,
        8.0f,      9.0f,         10.0f,       11.0f,        12.0f,      13.0f,        14.0f,       15.0f,
        16.0f,     18.0f,        20.0f,       22.0f,        24.0f,      26.0f,        28.0f,       30.0f,
        32.0f,     36.0f,        40.0f,       44.0f,        48.0f,      52.0f,        56.0f,       60.0f,
        64.0f,     72.0f,        80.0f,       88.0f,        96.0f,      104.0f,       112.0f,      120.0f,
        128.0f,    144.0f,       160.0f,      176.0f,       192.0f,     208.0f,       224.0f,      240.0f,
        256.0f,    288.0f,       320.0f,      352.0f,       384.0f,     416.0f,       448.0f,      float_nan};

constexpr uint32_t three_bytes_shift = 24;

constexpr uint8_t f8e4m3_s_mask = 0x80;  // f8e4m3 sign bit mask
constexpr uint8_t f8e4m3_e_size = 4;     // f8e4m3 exponent bit size
constexpr uint8_t f8e4m3_e_mask = 0x78;  // f8e4m3 exponent bit mask
constexpr uint8_t f8e4m3_e_bias = 7;     // f8e4m3 exponent bias
constexpr uint8_t f8e4m3_e_max = 0x0f;   // f8e4m3 exponent max value
constexpr uint8_t f8e4m3_m_size = 3;     // f8e4m3 mantissa bits size
constexpr uint8_t f8e4m3_m_mask = 0x07;  // f8e4m3 mantissa bit mask

uint8_t f16_to_f8e4m3_bits(const vpux::type::float16 value) {
    constexpr uint16_t f16_s_mask = 0x8000;  // f16 sign bit mask
    constexpr uint16_t f16_e_mask = 0x7C00;  // f16 exponent bits mask
    constexpr uint16_t f16_e_bias = 15;      // f16 exponent bias
    constexpr uint16_t f16_e_size = 5;       // f16 exponent bits size
    constexpr uint16_t f16_m_mask = 0x03ff;  // f16 mantissa bits mask
    constexpr uint16_t f16_m_size = 10;      // f16 mantissa bits size

    constexpr uint8_t byte_shift = 8;

    constexpr uint16_t f8_e_mask = f8e4m3_e_mask << byte_shift;  // f8 exponent bits mask (on u16)
    constexpr uint16_t f8_m_mask = f8e4m3_m_mask << byte_shift;  // f8 mantissa bits mask (on u16)
    constexpr uint16_t f8_m_hidden_one_mask = 0x0800;            // f8 mantissa hidden one bits mask (on u16)

    constexpr uint16_t round_half = 0x01ff;  // value for half to even round for f8
    constexpr uint16_t round_norm = 0x007f;  // value for normal round for f8
    constexpr uint16_t round_even = 0x0080;  // value for half to even round for f8
    constexpr uint16_t round_odd = 0x0180;   // value for an non-half to even round for f8

    // f8 exponent min value for subnormal
    // For f8_e less than -10, the hidden 1 is shifted beyond rounding bit.
    // So the 3 bits in mantissa and rounding bit are all 0, the f8 value is always 0.
    constexpr int16_t f8_e_subnormal_min = -10;

    const uint16_t input = value.to_bits();

    uint8_t f8_bits = static_cast<uint8_t>((input & f16_s_mask) >> byte_shift);

    uint16_t f16_e_field = input & f16_e_mask;

    if (f16_e_field == f16_e_mask) {
        f8_bits |= (f8e4m3_e_mask | f8e4m3_m_mask);
    } else if (f16_e_field != 0) {
        int16_t f8_biased_exp = (f16_e_field >> f16_m_size) - (f16_e_bias - f8e4m3_e_bias);
        uint16_t fractional = (input & f16_m_mask) << (f16_e_size - f8e4m3_e_size);

        // for normalized values round apply rounding change f8 fractional and biased exponent
        if ((fractional & round_half) == round_odd || (fractional & round_norm) != 0) {
            fractional += round_even;
            if (0 != (fractional & f8_e_mask)) {
                fractional &= f8_e_mask;
                ++f8_biased_exp;
            }
        }
        fractional &= f8_m_mask;

        // set exponent and mantissa on f8 bits
        if (f8_biased_exp > f8e4m3_e_max) {
            // Use NAN as this type has no infinity
            f8_bits |= (f8e4m3_e_mask | f8e4m3_m_mask);
        } else if (f8_biased_exp > 0) {
            f8_bits |= (f8_biased_exp << f8e4m3_m_size) | (fractional >> byte_shift);
        } else {
            // Restore the hidden 1 in f8 mantissa for subnormal calculation
            fractional = f8_m_hidden_one_mask | (input & f16_m_mask) << (f16_e_size - f8e4m3_e_size);
            int16_t f8_exp = f8_biased_exp - f8e4m3_e_bias;
            int16_t shift = 1 - f8_exp;
            int16_t sticky_mask = f8_exp < f8_e_subnormal_min ? 0 : ((1 << shift) - 1);
            uint16_t sticky = (fractional & sticky_mask) ? 1 : 0;

            // Subnormal mantissa has less significant bits for smaller exponent
            fractional = f8_exp < f8_e_subnormal_min ? 0 : fractional >> (1 - f8_biased_exp);
            // apply rounding
            if (((fractional & round_half) == round_odd && sticky == 0) || (fractional & round_norm) != 0 ||
                sticky != 0) {
                fractional += round_even;
            }

            f8_bits |= fractional >> byte_shift;
        }
    }

    return f8_bits;
}

#define EXTRACT_F32_SIGN(x) ((x >> 31) & 0x1)
#define EXTRACT_F32_EXP(x) ((x >> 23) & 0xFF)
#define EXTRACT_F32_FRAC(x) (x & 0x007FFFFF)
#define HF8_EXP_BITS (4)
#define HF8_MANTISSA_BITS (3)
#define HF8_SIGN_BIT_POS (HF8_EXP_BITS + HF8_MANTISSA_BITS)
#define PACK_HF8(x, y, z) ((x << HF8_SIGN_BIT_POS) + (y << HF8_MANTISSA_BITS) + (z))

// NaN default
#define HF8_NAN_DEFAULT 0x7F

// exceptions
#define F32_EX_INEXACT 0x00000001
#define F32_EX_DIV_BY_ZERO 0x00000002
#define F32_EX_INVALID 0x00000004
#define F32_EX_UNDERFLOW 0x00000008
#define F32_EX_OVERFLOW 0x00000010

// Exponent bias
#define FP32_EXP_BIAS (127)
#define HF8_EXP_BIAS (7)

// rounding modes
#define F32_RND_NEAREST_EVEN 0
#define F32_RND_MINUS_INF 1
#define F32_RND_PLUS_INF 2
#define F32_RND_TO_ZERO 3
#define F32_RND_NEAREST_AWAY 4

// detect tinyness mode
#define F32_DETECT_TINY_AFTER_RND 0
#define F32_DETECT_TINY_BEFORE_RND 1

unsigned int detect_tiny_mode{F32_DETECT_TINY_BEFORE_RND};

unsigned int f16_shift_right_loss_detect(unsigned int op, unsigned int cnt) {
    unsigned int ret_val;
    if (cnt == 0) {
        ret_val = op;
    } else if (cnt < 16) {
        ret_val = op >> cnt;
        // mark LSB as 1 if we shifted out some ones
        if ((op & ((0x1 << cnt) - 1)) != 0) {
            ret_val |= 0x1;
        }
    } else {
        // mark LSB as 1 if we shifted out some ones
        ret_val = (op != 0) ? 1 : 0;
    }
    return ret_val;
}

unsigned int f32_to_hf8_conv(unsigned int x, unsigned int rnd_mode, unsigned int* exceptions, unsigned int preserveDens,
                             unsigned int clamp) {
    unsigned int result{0};
    unsigned int sign = EXTRACT_F32_SIGN(x);
    int exp = EXTRACT_F32_EXP(x);
    unsigned int frac = EXTRACT_F32_FRAC(x);
    // clear flags
    *exceptions = 0;

    if ((exp == 0) && (frac == 0)) {
        // FP32 number is zero
        result = PACK_HF8(sign, 0, 0);
    } else if ((exp == 0) && !preserveDens) {
        // flushing FP32 input denormals to zero (no exceptions flags)
        // FP32 number is zero
        result = PACK_HF8(sign, 0, 0);
    } else if (exp == 0xFF) {
        if (frac == 0) {
            // Infinity
            // Input Infinity needs to be converted to HF8 NaN, preserving the sign
            result = (sign << HF8_SIGN_BIT_POS) | HF8_NAN_DEFAULT;
        } else {
            // fp32 number is a NaN - return QNaN, raise invalid if
            // SNaN. QNaN assumed to have MSB of significand set
            if ((frac & 0x00400000) == 0)
                *exceptions |= F32_EX_INVALID;

            // sign needs to be preserved for HF8 NaNs
            result = (sign << HF8_SIGN_BIT_POS) | HF8_NAN_DEFAULT;
        }
    } else {
        // FP32 number is normal or denormal

        // Add hidden bit if normal
        if (exp != 0) {
            frac = frac | 0x00800000;
        }

        // Unbias exponent
        exp = exp - FP32_EXP_BIAS;

        // Check if not below HF8 denormal
        if (exp >= -6) {
            // Extract lsb, round and sticky bits
            int round = (frac & 0x00080000) >> 19;  // MSB of discarded FP32 mantissa
            int sticky = ((frac & 0x0007FFFF) == 0) ? 0 : 1;
            frac = frac >> 20;             // Truncate mantissa
            int flsb = frac & 0x00000001;  // LSB of HF8 frac

            // Increment if necessary
            switch (rnd_mode) {
                // Use softfloat mappings (P_CFG will have been mapped before call to CMU
            case F32_RND_NEAREST_EVEN:
                if ((round && flsb) || (round && sticky)) {
                    frac = frac + 1;
                }
                break;
            case F32_RND_TO_ZERO:
                break;
            case F32_RND_PLUS_INF:
                if ((sign == 0) && (round || sticky)) {
                    frac = frac + 1;
                }
                break;
            case F32_RND_MINUS_INF:
                if ((sign == 1) && (round || sticky)) {
                    frac = frac + 1;
                }
                break;
            }

            // Inexact if either round or sticky bit set
            if (round || sticky) {
                *exceptions |= F32_EX_INEXACT;
            }

            // Check if rounding caused mantissa overflow
            if ((frac & 0x00000010))  // Allow for hidden bit
            {
                frac = frac >> 1;
                exp = exp + 1;
            }

            // Add BF8 bias to exponent
            exp = exp + HF8_EXP_BIAS;

            // Check for exponent overflow
            if ((exp > 15) || ((exp == 15) && ((frac & 0x07) > 6))) {
                // Set overflow and inexact flags
                *exceptions |= F32_EX_OVERFLOW;
                *exceptions |= F32_EX_INEXACT;

                if (clamp == 1) {
                    result = PACK_HF8(sign, 0xF, 0x6);  // Largest HF8 finite value
                } else {
                    // Return according to rounding mode
                    switch (rnd_mode) {
                    case F32_RND_NEAREST_EVEN:
                        result = PACK_HF8(sign, 0xF, 0x7);  // Infinity
                        break;
                    case F32_RND_TO_ZERO:
                        result = PACK_HF8(sign, 0xF, 0x6);  // Largest finite #
                        break;
                    case F32_RND_PLUS_INF:
                        result = (sign == 0) ? 0x7F : 0xFE;
                        break;
                    case F32_RND_MINUS_INF:
                        result = (sign == 1) ? 0xFF : 0x7E;
                        break;
                    }
                }
            } else {
                // Remove hidden bit and pack
                frac = frac & 0x07;
                result = PACK_HF8(sign, exp, frac);
            }
        } else {
            // HF8 denormal
            if (preserveDens == 1) {
                bool is_tiny{false};

                frac = f16_shift_right_loss_detect(frac, abs(exp + 6));

                // Extract lsb, round and sticky bits
                int round = (frac & 0x00080000) >> 19;  // MSB of discarded FP32 mantissa
                int sticky = ((frac & 0x0007FFFF) == 0) ? 0 : 1;
                frac = frac >> 20;             // Truncate mantissa
                int flsb = frac & 0x00000001;  // LSB of hF8 frac

                is_tiny = (detect_tiny_mode == F32_DETECT_TINY_BEFORE_RND);  // as per FergalO'C

                // Increment if necessary
                switch (rnd_mode) {
                case F32_RND_NEAREST_EVEN:
                    if ((round && flsb) || (round && sticky)) {
                        frac = frac + 1;
                    }
                    break;
                case F32_RND_TO_ZERO:
                    break;
                case F32_RND_PLUS_INF:
                    if ((sign == 0) && (round || sticky)) {
                        frac = frac + 1;
                    }
                    break;
                case F32_RND_MINUS_INF:
                    if ((sign == 1) && (round || sticky)) {
                        frac = frac + 1;
                    }
                    break;
                }

                exp = 0;
                // Check if mantissa became normal again after rounding
                if (frac & 0x00000008) {
                    exp = exp + 1;
                } else {
                    // Rounded result is tiny
                    is_tiny = true;
                }

                // Inexact if either round or sticky bit set
                if (round || sticky) {
                    *exceptions |= F32_EX_INEXACT;

                    // Underflow if also tiny
                    if (is_tiny) {
                        *exceptions |= F32_EX_UNDERFLOW;
                    }
                }

                // Remove hidden bit and pack
                frac = frac & 0x07;
                result = PACK_HF8(sign, exp, frac);
            } else {
                *exceptions |= (F32_EX_UNDERFLOW | F32_EX_INEXACT);
                // Flushing denormals to zero
                result = PACK_HF8(sign, 0, 0);
            }
        }
    }

    return result;
}

vpux::type::float8_e4m3::float8_e4m3(const uint32_t sign, const uint32_t biased_exponent, const uint32_t fraction)
        : m_value(((sign & 0x01U) << (f8e4m3_e_size + f8e4m3_m_size)) |
                  (biased_exponent & (f8e4m3_e_mask >> f8e4m3_m_size)) << f8e4m3_m_size | (fraction & f8e4m3_m_mask)) {
}

union f32_t {
    float value;
    uint32_t bits;
};

vpux::type::float8_e4m3::float8_e4m3(const float value) {
    const auto input = f32_t{value};
    uint32_t exceptions = 0;
    auto hf8Val = f32_to_hf8_conv(input.bits, F32_RND_NEAREST_EVEN, &exceptions, 1, 0);
    m_value = static_cast<uint8_t>(hf8Val & 0xFF);
}

vpux::type::float8_e4m3::operator float() const {
    auto converted = f32_t{f8_to_float_lut[m_value & (f8e4m3_e_mask | f8e4m3_m_mask)]};
    converted.bits |= (m_value & f8e4m3_s_mask) << three_bytes_shift;
    return converted.value;
}

uint8_t vpux::type::float8_e4m3::to_bits() const {
    return m_value;
}
