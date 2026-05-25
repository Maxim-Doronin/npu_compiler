//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset14.hpp>
#include <openvino/opsets/opset3.hpp>
#include <pretty_test_arguments.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

struct AttentionSinkPatternParams {
    ov::Shape inputQ;
    ov::Shape inputKTranspose;
    ov::Shape inputV;
    ov::Shape inputMask;
    ov::Shape inputSink;
};

class FuseAttentionSinkPatternTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<AttentionSinkPatternParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        const size_t sinkIndex = funcInputs.size() - 1;
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = (i == sinkIndex) ? 4 : 1;
            in_data.resolution = 32768;
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    void SetUp() override {
        auto elementType = ov::element::f16;
        inType = outType = elementType;

        const auto testParams = GetParam();
        const auto& inputQShape = testParams.inputQ;
        const auto& inputKTransposeShape = testParams.inputKTranspose;
        const auto& inputVShape = testParams.inputV;
        const auto& inputMaskShape = testParams.inputMask;
        const auto& inputSinkShape = testParams.inputSink;

        init_input_shapes(ov::test::static_shapes_to_test_representation(
                {inputQShape, inputKTransposeShape, inputVShape, inputMaskShape, inputSinkShape}));

        const auto inputQ = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(0));
        const auto inputKTranspose = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(1));
        const auto inputV = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(2));
        const auto inputMask = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(3));
        auto inputSink = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(4));

        const auto scores = std::make_shared<ov::opset14::MatMul>(inputQ, inputKTranspose, false, true);
        const auto maskedScores =
                std::make_shared<ov::opset14::Add>(scores, inputMask, ov::op::AutoBroadcastType::NUMPY);

        // broadcast sink resolution if need
        std::shared_ptr<ov::op::v0::Concat> scoresWithSink;
        if (inputSinkShape[2] == 1) {
            const auto broadcastShape =
                    ov::Shape{inputSinkShape[0], inputSinkShape[1], inputQShape[2], inputSinkShape[3]};
            const auto target_shape_input =
                    std::make_shared<ov::opset3::Constant>(ov::element::i32, ov::Shape{4}, broadcastShape);
            const auto broadcastedSink = std::make_shared<ov::opset14::Broadcast>(inputSink, target_shape_input);
            broadcastedSink->set_friendly_name("sdp_sink_pattern");
            scoresWithSink = std::make_shared<ov::opset14::Concat>(ov::OutputVector{maskedScores, broadcastedSink}, 3);
        } else {
            scoresWithSink = std::make_shared<ov::opset14::Concat>(ov::OutputVector{maskedScores, inputSink}, 3);
        }

        const auto normalizedScores = std::make_shared<ov::opset14::Softmax>(scoresWithSink, 3);
        const auto qShape = inputQShape;
        const auto kTShape = inputKTransposeShape;
        const auto beginConst =
                ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 0});
        const auto endConst = ov::opset3::Constant::create(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(qShape[0]), static_cast<int64_t>(qShape[1]),
                                     static_cast<int64_t>(qShape[2]), static_cast<int64_t>(kTShape[2])});
        const auto stridesConst =
                ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
        const auto slicedScores = std::make_shared<ov::opset3::StridedSlice>(
                normalizedScores, beginConst, endConst, stridesConst, std::vector<int64_t>{0, 0, 0, 0},
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0},
                std::vector<int64_t>{0, 0, 0, 0});

        const auto output = std::make_shared<ov::opset14::MatMul>(slicedScores, inputV, false, false);
        output->set_friendly_name("sdp_sink_pattern");

        ov::ParameterVector inputParams{inputQ, inputKTranspose, inputV, inputMask, inputSink};
        auto results = ov::ResultVector{std::make_shared<ov::opset3::Result>(output)};

        function = std::make_shared<ov::Model>(results, inputParams, "SDPSinkPattern");
        functionRefs = function->clone();
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<AttentionSinkPatternParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        const auto& p = obj.param;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "Q=" << p.inputQ << sep;
        result << "KT=" << p.inputKTranspose << sep;
        result << "V=" << p.inputV << sep;
        result << "Mask=" << p.inputMask << sep;
        result << "SinkBase=" << p.inputSink;
        return result.str();
    };
};

TEST_P(FuseAttentionSinkPatternTestCommon, NPU5010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(
        smoke_AttentionWithSink_model_scenario, FuseAttentionSinkPatternTestCommon,
        ::testing::ValuesIn({
                AttentionSinkPatternParams{
                        {1, 64, 1024, 64}, {1, 64, 1024, 64}, {1, 64, 1024, 64}, {1, 1, 1024, 1024}, {1, 64, 1024, 1}},
                AttentionSinkPatternParams{
                        {1, 64, 1024, 64}, {1, 64, 1024, 64}, {1, 64, 1024, 64}, {1, 1, 1024, 1024}, {1, 64, 1, 1}},
        }),
        FuseAttentionSinkPatternTestCommon::getTestCaseName);
}  // namespace ov::test::subgraph
