//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>
#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/type/float8_e8m0.hpp"

std::vector<float> generateF8E8M0Floats() {
    std::vector<float> powers(256);

    uint32_t nan_bits = 0x7FC00000;  // fp32 positive quiet NaN
    float nan_val;
    std::memcpy(&nan_val, &nan_bits, sizeof(float));
    powers[255] = nan_val;

    std::generate(powers.begin(), powers.end() - 1, [i = -127]() mutable {
        return std::pow(2.0f, i++);
    });
    return powers;
}

template <class TContainer>
std::vector<std::tuple<int, typename TContainer::value_type>> enumerate(const TContainer& values) {
    std::vector<std::tuple<int, typename TContainer::value_type>> enum_values;
    for (const auto& v : values | vpux::indexed) {
        enum_values.emplace_back(v.index(), v.value());
    }
    return enum_values;
}

TEST(float8_e8m0, f32_inf) {
    const auto f8 = vpux::type::float8_e8m0(std::numeric_limits<float>::infinity());
    // f8 is max as there is no infinity
    EXPECT_EQ(f8.to_bits(), 0b11111110);
    EXPECT_EQ(std::numeric_limits<vpux::type::float8_e8m0>::has_infinity, false);
}

TEST(float8_e8m0, f32_minus_inf) {
    const auto f8 = vpux::type::float8_e8m0(-std::numeric_limits<float>::infinity());
    // f8 is min as there is no minus infinity
    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(float8_e8m0, f32_negative) {
    const auto f8 = vpux::type::float8_e8m0(-10.5f);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(float8_e8m0, f32_rounding) {
    const auto f8 = vpux::type::float8_e8m0(1.75f);

    EXPECT_EQ(f8.to_bits(), 0b10000000);
}

TEST(float8_e8m0, f32_ge_f8_max_within_round_to_max) {
    const float f8_as_float = std::pow(2.0, 127) + 10;
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b11111110);
}

TEST(float8_e8m0, f32_ge_f8_max_not_within_round_to_max) {
    const float f8_as_float = std::pow(2.0, 127) * 1.75;
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b11111110);
}

TEST(float8_e8m0, f32_le_f8_lowest_within_round_to_lowest) {
    const float f8_as_float = std::pow(2.0, -126) * 0.4;
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(float8_e8m0, f32_le_f8_lowest_not_within_round_to_lowest) {
    const float f8_as_float = std::pow(2.0, -126) * 0.1;
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(float8_e8m0, stream_operator) {
    std::stringstream s;
    s << vpux::type::float8_e8m0(0.25f);

    EXPECT_EQ(s.str(), "0.25");
}

TEST(float8_e8m0, to_string) {
    const auto f8 = vpux::type::float8_e8m0::from_bits(0b10000110);

    EXPECT_EQ(std::to_string(f8), "128.000000");
}

const auto exp_floats = generateF8E8M0Floats();

using f8m0e8_params = std::tuple<int, float>;
class F8E8M0PTest : public testing::TestWithParam<f8m0e8_params> {};

INSTANTIATE_TEST_SUITE_P(convert, F8E8M0PTest, testing::ValuesIn(enumerate(exp_floats)),
                         testing::PrintToStringParamName());

TEST_P(F8E8M0PTest, f8_bits_to_f32) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<1>(params);
    const auto f8 = vpux::type::float8_e8m0::from_bits(std::get<0>(params));

    if (std::isnan(exp_value)) {
        EXPECT_TRUE(std::isnan(static_cast<float>(f8)));
    } else {
        EXPECT_EQ(static_cast<float>(f8), exp_value);
    }
}

TEST_P(F8E8M0PTest, f32_to_f8_bits) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<0>(params);
    const auto& value = std::get<1>(params);
    const auto f8 = vpux::type::float8_e8m0(value);

    if (exp_value == 0xFF) {  // quiet NaN
        EXPECT_TRUE(std::isnan(f8));
    }

    EXPECT_EQ(f8.to_bits(), exp_value);
}

TEST(float8_e8m0, f8e8m0_num_limits_exp) {
    const auto min_exp = std::numeric_limits<vpux::type::float8_e8m0>::min_exponent;
    const auto min_exp10 = std::numeric_limits<vpux::type::float8_e8m0>::min_exponent10;
    const auto max_exp = std::numeric_limits<vpux::type::float8_e8m0>::max_exponent;
    const auto max_exp10 = std::numeric_limits<vpux::type::float8_e8m0>::max_exponent10;

    EXPECT_EQ(min_exp, -126);
    EXPECT_EQ(min_exp10, -38);
    EXPECT_EQ(max_exp, 128);
    EXPECT_EQ(max_exp10, 38);
}

TEST(float8_e8m0, min_value) {
    const float f8_as_float = std::pow(2.0, -127);
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<vpux::type::float8_e8m0>::min().to_bits());
}

TEST(float8_e8m0, max_value) {
    const float f8_as_float = std::pow(2.0, 127);
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b11111110);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<vpux::type::float8_e8m0>::max().to_bits());
}

TEST(float8_e8m0, lowest_value) {
    const float f8_as_float = std::pow(2.0, -127);
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<vpux::type::float8_e8m0>::lowest().to_bits());
}

TEST(float8_e8m0, denorm_min) {
    const float f8_as_float = std::pow(2.0, -127);
    const auto f8 = vpux::type::float8_e8m0(f8_as_float);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<vpux::type::float8_e8m0>::denorm_min().to_bits());
}

TEST(float8_e8m0, num_limits_is_specialized) {
    const auto val = std::numeric_limits<vpux::type::float8_e8m0>::is_specialized;
    EXPECT_TRUE(val);
}

TEST(float8_e8m0, num_limits_is_signed) {
    const auto val = std::numeric_limits<vpux::type::float8_e8m0>::is_signed;
    EXPECT_FALSE(val);
}

TEST(float8_e8m0, num_limits_is_integer) {
    const auto val = std::numeric_limits<vpux::type::float8_e8m0>::is_integer;
    EXPECT_FALSE(val);
}

TEST(float8_e8m0, num_limits_is_exact) {
    const auto val = std::numeric_limits<vpux::type::float8_e8m0>::is_exact;
    EXPECT_FALSE(val);
}

TEST(float8_e8m0, num_limits_radix) {
    const auto val = std::numeric_limits<vpux::type::float8_e8m0>::radix;
    EXPECT_EQ(val, 2);
}

TEST(float8_e8m0, num_limits_digits) {
    const auto val = std::numeric_limits<vpux::type::float8_e8m0>::digits;
    EXPECT_EQ(val, 1);
}

TEST(float8_e8m0, num_limits_digits10) {
    const auto f8_dig = std::numeric_limits<vpux::type::float8_e8m0>::digits;
    const auto f8_dig10 = std::numeric_limits<vpux::type::float8_e8m0>::digits10;

    EXPECT_EQ(f8_dig10, static_cast<int>((f8_dig - 1) * std::log10(2)));
    EXPECT_EQ(f8_dig10, 0);
}

TEST(float8_e8m0, num_limits_epsilon) {
    const auto f8_1 = vpux::type::float8_e8m0(1.f);
    const auto f8_1_bits = f8_1.to_bits();
    const auto f8_1_next_bits = f8_1_bits + 1u;

    const auto f8_eps = vpux::type::float8_e8m0::from_bits(f8_1_next_bits - f8_1_bits);

    EXPECT_EQ(f8_eps, std::numeric_limits<vpux::type::float8_e8m0>::epsilon());
    EXPECT_EQ(f8_eps.to_bits(), std::numeric_limits<vpux::type::float8_e8m0>::epsilon().to_bits());
}

TEST(float8_e8m0, num_limits_round_error) {
    const auto f8 = vpux::type::float8_e8m0(0.5f);

    EXPECT_EQ(f8, std::numeric_limits<vpux::type::float8_e8m0>::round_error());
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<vpux::type::float8_e8m0>::round_error().to_bits());
}

TEST(float8_e8m0, quiet_nan) {
    const auto has_quiet_nan = std::numeric_limits<vpux::type::float8_e8m0>::has_quiet_NaN;
    EXPECT_TRUE(has_quiet_nan);
    EXPECT_EQ(std::numeric_limits<vpux::type::float8_e8m0>::quiet_NaN().to_bits(), 0b11111111);
}

TEST(float8_e8m0, f32_quiet_nan) {
    const auto f8 = vpux::type::float8_e8m0(std::numeric_limits<float>::quiet_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);
}

TEST(float8_e8m0, f32_sig_nan) {
    const auto f8 = vpux::type::float8_e8m0(std::numeric_limits<float>::signaling_NaN());

    const auto has_sig_nan = std::numeric_limits<vpux::type::float8_e8m0>::has_signaling_NaN;
    EXPECT_FALSE(has_sig_nan);
    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);  // mapped to quiet NaN
    EXPECT_EQ(0, std::numeric_limits<vpux::type::float8_e8m0>::signaling_NaN().to_bits());
}

using rounding_params = std::tuple<float, uint8_t, float>;

class F32ToF8E8M0RoundingTest : public ::testing::TestWithParam<rounding_params> {};

// clang-format off
INSTANTIATE_TEST_SUITE_P(boundary_params,
                         F32ToF8E8M0RoundingTest,
                         ::testing::Values(rounding_params{powf(2.0f, -127) * 1.4999f, 0b00000000, powf(2.0f, -127)},
                                           rounding_params{powf(2.0f, -127) * 1.5000f, 0b00000000, powf(2.0f, -127)},
                                           rounding_params{powf(2.0f, -127) * 1.5001f, 0b00000001, powf(2.0f, -126)},

                                           rounding_params{0.09374f, 0b01111011, 0.0625f},
                                           rounding_params{0.09375f, 0b01111100, 0.125f},
                                           rounding_params{0.09376f, 0b01111100, 0.125f},

                                           rounding_params{0.1874f, 0b01111100, 0.125f},
                                           rounding_params{0.1875f, 0b01111100, 0.125f},
                                           rounding_params{0.1876f, 0b01111101, 0.25f},

                                           rounding_params{0.374f, 0b01111101, 0.25f},
                                           rounding_params{0.375f, 0b01111110, 0.5f},
                                           rounding_params{0.376f, 0b01111110, 0.5f},

                                           rounding_params{0.74f, 0b01111110, 0.5f},
                                           rounding_params{0.75f, 0b01111110, 0.5f},
                                           rounding_params{0.76f, 0b01111111, 1.0f},

                                           rounding_params{1.49f, 0b01111111, 1.0f},
                                           rounding_params{1.50f, 0b10000000, 2.0f},
                                           rounding_params{1.51f, 0b10000000, 2.0f},

                                           rounding_params{2.99f, 0b10000000, 2.0f},
                                           rounding_params{3.00f, 0b10000000, 2.0f},
                                           rounding_params{3.01f, 0b10000001, 4.0f},

                                           rounding_params{5.99f, 0b10000001, 4.0f},
                                           rounding_params{6.00f, 0b10000010, 8.0f},
                                           rounding_params{6.01f, 0b10000010, 8.0f},

                                           rounding_params{powf(2.0f, 126) * 1.4999f, 0b11111101, powf(2.0f, 126)},
                                           rounding_params{powf(2.0f, 126) * 1.5000f, 0b11111110, powf(2.0f, 127)},
                                           rounding_params{powf(2.0f, 126) * 1.5001f, 0b11111110, powf(2.0f, 127)}),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(rounding_params,
                         F32ToF8E8M0RoundingTest,
                         ::testing::Values(rounding_params{0.0f, 0b00000000, powf(2.0f, -127)},
                                           rounding_params{powf(2.0f, -127), 0b00000000, powf(2.0f, -127)},
                                           rounding_params{powf(2.0f, -127) * 1.0625f, 0b00000000, powf(2.0f, -127)},

                                           rounding_params{0.100f, 0b01111100, 0.125f},
                                           rounding_params{0.125f, 0b01111100, 0.125f},
                                           rounding_params{0.150f, 0b01111100, 0.125f},

                                           rounding_params{0.24f, 0b01111101, 0.25f},
                                           rounding_params{0.25f, 0b01111101, 0.25f},
                                           rounding_params{0.26f, 0b01111101, 0.25f},

                                           rounding_params{0.49f, 0b01111110, 0.5f},
                                           rounding_params{0.50f, 0b01111110, 0.5f},
                                           rounding_params{0.51f, 0b01111110, 0.5f},

                                           rounding_params{0.99f, 0b01111111, 1.0f},
                                           rounding_params{1.00f, 0b01111111, 1.0f},
                                           rounding_params{1.01f, 0b01111111, 1.0f},

                                           rounding_params{1.99f, 0b10000000, 2.0f},
                                           rounding_params{2.00f, 0b10000000, 2.0f},
                                           rounding_params{2.01f, 0b10000000, 2.0f},

                                           rounding_params{3.99f, 0b10000001, 4.0f},
                                           rounding_params{4.00f, 0b10000001, 4.0f},
                                           rounding_params{4.01f, 0b10000001, 4.0f},

                                           rounding_params{7.99f, 0b10000010, 8.0f},
                                           rounding_params{8.00f, 0b10000010, 8.0f},
                                           rounding_params{8.50f, 0b10000010, 8.0f},

                                           rounding_params{14.00f, 0b10000011, 16.0f},
                                           rounding_params{16.00f, 0b10000011, 16.0f},
                                           rounding_params{16.01f, 0b10000011, 16.0f},

                                           rounding_params{powf(2.0f, 126) * 1.85f, 0b11111110, powf(2.0f, 127)},
                                           rounding_params{powf(2.0f, 127), 0b11111110, powf(2.0f, 127)},
                                           rounding_params{std::numeric_limits<float>::infinity(), 0b11111110, powf(2.0f, 127)}),
                         ::testing::PrintToStringParamName());
// clang-format on

TEST_P(F32ToF8E8M0RoundingTest, round_behavior) {
    const auto& [input, expected_bits, expected_float] = GetParam();

    const auto f8 = vpux::type::float8_e8m0(input);
    EXPECT_EQ(f8.to_bits(), expected_bits);
    EXPECT_NEAR(static_cast<float>(f8), expected_float, 0.0001f);
}
