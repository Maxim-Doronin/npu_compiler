//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/rms.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

using RMSNormDecompositionParams = std::tuple<ov::Shape,          // input shapes
                                              ov::element::Type,  // input precision
                                              ov::Shape>;         // gamma shape (empty = same as input)

class FuseRMSTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<RMSNormDecompositionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RMSNormDecompositionParams> obj) {
        ov::Shape inputShape;
        ov::element::Type inputPrecision;
        ov::Shape gammaShape;
        std::tie(inputShape, inputPrecision, gammaShape) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape=" << inputShape << sep;
        result << "GammaShape=" << (gammaShape.empty() ? ov::Shape{inputShape.back()} : gammaShape) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes, const ov::Shape& target_shape,
                                             const ov::element::Type input_precision, const ov::Shape& gamma_shape) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0])};

        // x^2
        auto power_const = ov::op::v0::Constant::create(input_precision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(params[0], power_const);

        // ReduceMean(x^2,axes)
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);

        // ReduceMean(x^2,axes)+eps
        auto eps = ov::op::v0::Constant::create(input_precision, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);

        // Sqrt(ReduceMean(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);

        // 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto div_const = ov::op::v0::Constant::create(input_precision, {}, {1});
        auto div = std::make_shared<ov::op::v1::Divide>(div_const, sqrt);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(params[0], div);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
        // empty gamma_shape means use input's last dimension
        ov::Shape actual_gamma_shape = gamma_shape.empty() ? ov::Shape{target_shape.back()} : gamma_shape;

        auto tensor = ov::test::utils::create_and_fill_tensor(input_precision, actual_gamma_shape);
        auto gamma = std::make_shared<ov::op::v0::Constant>(tensor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(gamma, mul1);

        auto comp = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f16);

        return std::make_shared<ov::Model>(ov::OutputVector{comp}, params, "RMSNormDecomposition");
    }
    void SetUp() override {
        ov::Shape input_shapes;
        ov::element::Type input_precision;
        ov::Shape gamma_shape;

        std::tie(input_shapes, input_precision, gamma_shape) = GetParam();
        inType = outType = input_precision;
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shapes}));

        std::vector<ov::PartialShape> partial_shapes = {inputDynamicShapes.front()};
        function = init_subgraph(partial_shapes, input_shapes, input_precision, gamma_shape);
    }
};

TEST_P(FuseRMSTestCommon, NPU3720_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    // TODO E####-203348 - delete unroll after fix
    setBatchCompilerMode("unroll");
    run(Platform::NPU3720);
}

TEST_P(FuseRMSTestCommon, NPU4000_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    // TODO E####-203348 - delete unroll after fix
    setBatchCompilerMode("unroll");
    run(Platform::NPU4000);
}

TEST_P(FuseRMSTestCommon, NPU5010_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    // TODO E####-203348 - delete unroll after fix
    setBatchCompilerMode("unroll");
    run(Platform::NPU5010);
}
TEST_P(FuseRMSTestCommon, NPU5020_HW) {
    abs_threshold = 0.11f;
    setDefaultHardwareMode();
    // TODO E####-203348 - delete unroll after fix
    // TODO E####-159644
    setBatchCompilerMode("unroll");
    run(Platform::NPU5020);
}

namespace {
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};

const std::vector<ov::Shape> input_shapes_basic = {{{1, 2, 6}}, {{2, 2, 6}}};
const std::vector<ov::Shape> input_shapes = {{32}, {{3, 32}}, {{1, 32, 16}}, {{1, 4, 16, 16}}, {{1, 77, 4096}}};

const std::vector<ov::Shape> gamma_shapes_default = {{}};
const std::vector<ov::Shape> gamma_shapes_broadcast = {{1, 1, 1}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(gamma_shapes_default)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(gamma_shapes_default)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS_BroadcastGamma, FuseRMSTestCommon,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 4, 128}),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(gamma_shapes_broadcast)),
                         FuseRMSTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
