//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common/quantization_utils.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/opsets/opset6_decl.hpp"

using namespace ov::test::utils;
using namespace ov::test;

struct QuantizationParams {
    size_t levels;
    FakeQuantizeParams weightsQuant;
};

//      [input]
//         |
//    Convolution -- FakeQuantize (16, 15 levels - sub-byte datatype i|u 4) -- [weights]
//         |
//      [output]

class SubByteQuantizationTransformationLayerTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizationParams> {
    void SetUp() override {
        const auto& quantParams = GetParam();

        init_input_shapes(static_shapes_to_test_representation({ov::Shape{1, 16, 16, 16}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto input = params.at(0)->get_default_output();

        auto weights = buildWeights(input.get_shape(), 5.0f);
        weights = utils::makeFakeQuantize(weights, ov::element::f16, quantParams.levels, quantParams.weightsQuant)
                          ->get_default_output();

        auto conv = buildConv(input, weights);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};

        function = std::make_shared<ov::Model>(results, params, "SubByteQuantizationTransformationTest");
    }

    ov::Output<ov::Node> buildWeights(const ov::Shape& shape, float value) const {
        const auto weightsShape = ov::Shape{16, shape.at(1), 1, 1};
        const size_t totalWeightsSize = weightsShape[0] * weightsShape[1] * weightsShape[2] * weightsShape[3];
        std::vector<float> weights(totalWeightsSize, value);

        return ov::op::v0::Constant::create(ov::element::f16, weightsShape, weights)->get_default_output();
    }

    ov::Output<ov::Node> buildConv(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& weights) const {
        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};

        return std::make_shared<ov::opset6::Convolution>(input, weights, strides, pads_begin, pads_end, dilations)
                ->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizationParams>& obj) {
        const auto& quantParamsCase = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "levels=" << quantParamsCase.levels;
        result << sep << "WQ=" << quantParamsCase.weightsQuant;

        return result.str();
    };
};

TEST_P(SubByteQuantizationTransformationLayerTest, HW) {
    rel_threshold = 0.1;
    run(getTestDeviceId());
}

std::vector<QuantizationParams> getQuantParams() {
    return {
            QuantizationParams{16, FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f})},
            QuantizationParams{16, FakeQuantizeParams({-1.0f}, {1.f}, {-1.0f}, {1.f})},
            QuantizationParams{15, FakeQuantizeParams({0.f}, {100.f}, {0.f}, {100.f})},
            QuantizationParams{15, FakeQuantizeParams({-1.0f}, {1.f}, {-1.0f}, {1.f})},
    };
}

INSTANTIATE_TEST_SUITE_P(subByteQuantizationTransformation, SubByteQuantizationTransformationLayerTest,
                         ::testing::ValuesIn(getQuantParams()),
                         (appendPlatformTypeTestName<SubByteQuantizationTransformationLayerTest>));
