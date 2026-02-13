//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/mvn.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset6.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

namespace ov::test::subgraph {

using MVNEpsMode = ov::op::MVNEpsMode;
using MVNAxes2Params = std::tuple<ov::Shape, std::vector<int64_t>, bool, float, MVNEpsMode>;

class MVNAxes2LayerTest : public VpuOv2LayerTest, public ::testing::WithParamInterface<MVNAxes2Params> {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        for (const auto& shape : targetInputStaticShapes) {
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 10, -5, 1);
            inputs[function->get_parameters().at(0)] = tensor;
        }
    }

protected:
    void SetUp() override {
        auto params = GetParam();
        const auto& dataShape = std::get<0>(params);
        const auto& axes = std::get<1>(params);
        bool normalizeVariance = std::get<2>(params);
        float eps = std::get<3>(params);
        MVNEpsMode epsMode = std::get<4>(params);

        init_input_shapes(static_shapes_to_test_representation({dataShape}));

        const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto axesConst = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);

        const auto mvn = std::make_shared<ov::opset6::MVN>(param, axesConst, normalizeVariance, eps, epsMode);
        const auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(mvn->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MVNAxes2");
    }
};

const std::vector<ov::Shape> inShapesMvnAxes2 = {
        {{1, 32, 2048, 64}},
        {{1, 32, 1024, 64}},
        {{1, 32, 512, 64}},
        {{1, 32, 512, 1}},
};

const std::vector<bool> normalizeVarianceValues = {true};
const std::vector<float> epsValues = {1e-9f};
const std::vector<MVNEpsMode> epsModeValues = {MVNEpsMode::INSIDE_SQRT};

std::vector<MVNAxes2Params> generateMVNAxes2Params(const std::vector<ov::Shape>& shapesAndAxes,
                                                   const std::vector<bool>& normalizeVarianceValues,
                                                   const std::vector<float>& epsValues,
                                                   const std::vector<MVNEpsMode>& epsModeValues) {
    std::vector<MVNAxes2Params> result;
    std::vector<int64_t> axes = {2};
    for (const auto& inputShape : shapesAndAxes) {
        for (bool normVar : normalizeVarianceValues) {
            for (float eps : epsValues) {
                for (MVNEpsMode epsMode : epsModeValues) {
                    result.emplace_back(inputShape, axes, normVar, eps, epsMode);
                }
            }
        }
    }
    return result;
}

TEST_P(MVNAxes2LayerTest, NPU3720) {
    setDefaultHardwareMode();
    enableProfiling();
    run(Platform::NPU3720);
}

TEST_P(MVNAxes2LayerTest, NPU4000) {
    setDefaultHardwareMode();
    enableProfiling();
    run(Platform::NPU4000);
}

TEST_P(MVNAxes2LayerTest, NPU5010) {
    setDefaultHardwareMode();
    enableProfiling();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(MVNAxes2, MVNAxes2LayerTest,
                         ::testing::ValuesIn(generateMVNAxes2Params(inShapesMvnAxes2, normalizeVarianceValues,
                                                                    epsValues, epsModeValues)));

}  // namespace ov::test::subgraph
