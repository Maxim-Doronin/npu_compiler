//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6_decl.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test::utils;
using namespace ov::test;

struct FQAsSubMul {
    size_t num_channels;
    std::vector<int8_t> zp_per_channel;
    std::vector<float> scale_per_channel;
    int8_t zp_min;
    int8_t zp_max;
    float scale_min;
    float scale_max;
};

// Test parameters: <test_case, float_element_type, target_device>
using ZeroPointTableConvParams = std::tuple<FQAsSubMul, ov::element::Type, std::string>;

namespace {
template <typename T>
std::vector<T> generateRange(const size_t num_channels, const T min_val, const T max_val) {
    std::vector<T> result;
    result.reserve(num_channels);

    if (min_val == max_val) {
        // Constant value case
        result.assign(num_channels, min_val);
    } else {
        // Distribute values evenly across the range
        for (size_t i = 0; i < num_channels; ++i) {
            T value = min_val + static_cast<T>((max_val - min_val) * i / (num_channels - 1));
            result.push_back(value);
        }
    }
    return result;
}

FQAsSubMul createTest(const size_t num_channels, const int8_t zp_min, const int8_t zp_max, const float scale_min,
                      const float scale_max) {
    FQAsSubMul result;
    result.num_channels = num_channels;
    result.zp_min = zp_min;
    result.zp_max = zp_max;
    result.scale_min = scale_min;
    result.scale_max = scale_max;
    result.zp_per_channel = generateRange<int8_t>(num_channels, zp_min, zp_max);
    result.scale_per_channel = generateRange<float>(num_channels, scale_min, scale_max);

    return result;
}
}  // namespace

class ZeroPointTableConvTest : public testing::WithParamInterface<ZeroPointTableConvParams>, public VpuOv2LayerTest {
public:
    void SetUp() override {
        const auto& [test_case, float_element_type, device] = GetParam();
        targetDevice = device;

        const ov::Shape weightsShape{test_case.num_channels, test_case.num_channels, 2, 2};
        const size_t totalWeights = weightsShape[0] * weightsShape[1] * weightsShape[2] * weightsShape[3];
        std::vector<int8_t> weights(totalWeights, 10);

        auto i_weights = std::make_shared<ov::op::v0::Constant>(ov::element::i8, weightsShape, weights);
        auto f_weights = std::make_shared<ov::opset6::Convert>(i_weights, float_element_type);

        // Convert i8 zero points to f16, create per-channel zero point tensor and subtract it from weights
        auto i_zp = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{test_case.num_channels, 1, 1, 1},
                                                           test_case.zp_per_channel);
        auto f_zp = std::make_shared<ov::opset6::Convert>(i_zp, float_element_type);
        std::shared_ptr<ov::opset6::Subtract> zero_points_subtracted =
                std::make_shared<ov::opset6::Subtract>(f_weights, f_zp);

        // Create per-channel scale tensor and multiply zero_points_subtracted by it
        auto scale = std::make_shared<ov::op::v0::Constant>(
                float_element_type, ov::Shape{test_case.num_channels, 1, 1, 1}, test_case.scale_per_channel);
        std::shared_ptr<ov::opset6::Multiply> multiplied =
                std::make_shared<ov::opset6::Multiply>(zero_points_subtracted, scale);

        const ov::Shape inputShape{1, test_case.num_channels, 4, 4};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        const ov::ParameterVector conv_params{
                std::make_shared<ov::op::v0::Parameter>(float_element_type, ov::Shape(inputShape))};

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};

        const auto conv = std::make_shared<ov::opset6::Convolution>(conv_params[0], multiplied->output(0), strides,
                                                                    pads_begin, pads_end, dilations);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, conv_params, "ZeroPointTable_Conv");
    }

    static std::string getTestCaseName(testing::TestParamInfo<ZeroPointTableConvParams> obj) {
        const auto& [test_case, precision, _] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "OC=" << test_case.num_channels << sep;
        result << "ZP_" << static_cast<int>(test_case.zp_min);
        if (test_case.zp_min != test_case.zp_max) {
            result << "_to_" << static_cast<int>(test_case.zp_max);
        }
        result << sep;
        result << "precision=" << precision.get_type_name() << sep;
        return result.str();
    }
};

//
// Platform test definition
//

// clang-format off
const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<FQAsSubMul>{
                // 80 channels: ZP in range [1, 3], scale = 1
                createTest(80, 1, 3, 1.0f, 1.0f),
                // 80 channels: ZP in range [-3, 3], scales in range [1, 20]
                createTest(80, -3, 3, 1.0f, 20.0f),
                // 80 channels: ZP = 0, scales in range [1.3, 3.0]
                createTest(80, 0, 0, 1.3f, 3.0f),
                // 80 channels: ZP = 1, scales in range [1.3, 3.0]
                createTest(80, 1, 1, 1.3f, 3.0f),
                // 80 channels: ZP = 20, scale = 1
                createTest(80, 20, 20, 1.0f, 1.0f),

                // 16 channels: ZP in range [1, 3], scale = 1
                createTest(16, 1, 3, 1.0f, 1.0f),
                // 16 channels: ZP in range [-3, 3], scales in range [1, 20]
                createTest(16, -3, 3, 1.0f, 20.0f),
                // 16 channels: ZP = 0, scales in range [1.3, 3.0]
                createTest(16, 0, 0, 1.3f, 3.0f),
                // 16 channels: ZP = 1, scales in range [1.3, 3.0]
                createTest(16, 1, 1, 1.3f, 3.0f),
                // 16 channels: ZP = 20, scale = 1
                createTest(16, 20, 20, 1.0f, 1.0f),

                // 2000 channels: ZP in range [1, 5], scale = 1
                createTest(2000, 1, 5, 1.0f, 1.0f),
                // 2000 channels: ZP in range [-5, 5], scale = 1
                createTest(2000, -5, 5, 1.0f, 1.0f),
                // 2000 channels: ZP in range [-5, 5], scales in range [1.3, 3.0]
                createTest(2000, -6, 6, 1.3f, 3.0f),
                // 2000 channels: ZP = 0, scales in range [1.3, 3.0]
                createTest(2000, 0, 0, 1.3f, 3.0f),
                // 2000 channels: ZP = 1, scales in range [1.3, 3.0]
                createTest(2000, 1, 1, 1.3f, 3.0f),
                // 2000 channels: ZP = 20, scale = 1
                createTest(2000, 20, 20, 1.0f, 1.0f),
            }),
        ::testing::Values(ov::element::f16),
        ::testing::Values(test_utils::TARGET_DEVICE));
// clang-format on

INSTANTIATE_TEST_SUITE_P(zeroPointTableConv, ZeroPointTableConvTest, basicCases,
                         ZeroPointTableConvTest::getTestCaseName);
