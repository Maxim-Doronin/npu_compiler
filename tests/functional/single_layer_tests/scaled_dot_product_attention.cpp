//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/dimension.hpp>
#include <openvino/opsets/opset14.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

PRETTY_PARAM(Batch, int64_t);            // N
PRETTY_PARAM(SourceSeqLen, int64_t);     // S
PRETTY_PARAM(TargetSeqLen, int64_t);     // L
PRETTY_PARAM(QKEmbeddingSize, int64_t);  // E
PRETTY_PARAM(VEmbeddingSize, int64_t);   // Ev
PRETTY_PARAM(IsCasual, bool);
PRETTY_PARAM(HasAttentionMask, bool);
PRETTY_PARAM(HasScale, bool);

PRETTY_PARAM(InputType, ov::element::Type);

namespace {

std::vector<InputShape> generateInputShapes(Batch batch, SourceSeqLen sourceSeqLen, TargetSeqLen targetSeqLen,
                                            QKEmbeddingSize qkEmbeddingSize, VEmbeddingSize vEmbeddingSize,
                                            HasAttentionMask hasAttentionMask) {
    auto qShape = generateShapes(batch, targetSeqLen, qkEmbeddingSize);
    auto kShape = generateShapes(batch, sourceSeqLen, qkEmbeddingSize);
    auto vShape = generateShapes(batch, sourceSeqLen, vEmbeddingSize);

    auto inputShapes = std::vector<InputShape>{qShape, kShape, vShape};

    if (hasAttentionMask) {
        auto attentionMaskShape = generateShapes(batch, targetSeqLen, sourceSeqLen);
        inputShapes.push_back(attentionMaskShape);
    }

    return inputShapes;
}

ov::ParameterVector generateInputParams(const std::vector<ov::PartialShape>& inputDynamicShapes, InputType inputType,
                                        HasAttentionMask hasAttentionMask, HasScale hasScale) {
    ov::ParameterVector inputParams;

    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[2]));

    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");

    if (hasScale) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(
                inputType, hasAttentionMask ? inputDynamicShapes[3] : ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    } else if (hasAttentionMask) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[3]));
        inputParams.back()->set_friendly_name("attention_mask");
    }

    return inputParams;
}

}  // namespace

class SdpAttentionLayerTestCommon : public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        auto shapes = std::vector<ov::Shape>{targetInputStaticShapes[0], targetInputStaticShapes[1],
                                             targetInputStaticShapes[2]};
        if (hasScale) {
            shapes.push_back(hasAttentionMask ? targetInputStaticShapes[3] : ov::Shape{});
            shapes.push_back(ov::Shape{1});  // emplace_back leads to incorrect results
        } else if (hasAttentionMask) {
            shapes.push_back(targetInputStaticShapes[3]);
        }

        SubgraphBaseTest::generate_inputs(shapes);
    }

public:
    bool hasAttentionMask = false;
    bool hasScale = false;
};

//
// Cross Attention test class
//

using SdpAttentionLayerTestParams = std::tuple<Batch, SourceSeqLen, TargetSeqLen, QKEmbeddingSize, VEmbeddingSize,
                                               IsCasual, HasAttentionMask, HasScale, InputType>;

class SdpAttentionLayerTest :
        public testing::WithParamInterface<SdpAttentionLayerTestParams>,
        public SdpAttentionLayerTestCommon {
protected:
    void SetUp() override {
        const auto& [batch, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, isCasual, hasAttentionMask,
                     hasScale, inputType] = GetParam();

        this->hasAttentionMask = hasAttentionMask;
        this->hasScale = hasScale;

        const auto inputShapes = generateInputShapes(batch, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize,
                                                     hasAttentionMask);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, inputType, hasAttentionMask, hasScale);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto sdp = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCasual);
        sdp->set_friendly_name("sdp");

        auto results = ov::ResultVector();
        for (size_t i = 0; i < sdp->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset14::Result>(sdp->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "SDP");

        // Interpreter backend doesn't implement evaluate method for OP ScaledDotProductAttention
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }
};

//
// Self Attention test class
//

PRETTY_PARAM(SequenceLength, int64_t);  // S == L
PRETTY_PARAM(EmbeddingSize, int64_t);   // E == Ev

using SelfAttentionTestParams =
        std::tuple<Batch, SequenceLength, EmbeddingSize, IsCasual, HasAttentionMask, HasScale, InputType>;

class SelfAttentionLayerTest :
        public testing::WithParamInterface<SelfAttentionTestParams>,
        public SdpAttentionLayerTestCommon {
protected:
    void SetUp() override {
        const auto& [batch, sequenceLength, embeddingSize, isCasual, hasAttentionMask, hasScale, inputType] =
                GetParam();

        this->hasAttentionMask = hasAttentionMask;
        this->hasScale = hasScale;

        auto sourceSeqLen = SourceSeqLen{sequenceLength};
        auto targetSeqLen = TargetSeqLen{sequenceLength};
        auto qkEmbeddingSize = QKEmbeddingSize{embeddingSize};
        auto vEmbeddingSize = VEmbeddingSize{embeddingSize};

        const auto inputShapes = generateInputShapes(batch, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize,
                                                     hasAttentionMask);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, inputType, hasAttentionMask, hasScale);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto sdp = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCasual);
        sdp->set_friendly_name("sdp");

        auto results = ov::ResultVector();
        for (size_t i = 0; i < sdp->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset14::Result>(sdp->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "SDP");

        // Interpreter backend doesn't implement evaluate method for OP ScaledDotProductAttention
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }
};

TEST_P(SdpAttentionLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SelfAttentionLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SdpAttentionLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(SelfAttentionLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<InputType> inputPrecision = {ov::element::f16};

//
// SelfAttentionTests
//

INSTANTIATE_TEST_SUITE_P(smoke, SelfAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Batch{8}),                              // 12, 16
                                            ::testing::Values(SequenceLength{512}),                   // 1k, 2k, 4k, 8k
                                            ::testing::Values(EmbeddingSize{64}),                     // 128, 256
                                            ::testing::ValuesIn(std::vector<IsCasual>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),          //
                                            ::testing::ValuesIn(inputPrecision)                               //
                                            ),
                         PrintTestCaseName());

//
// CrossAttentionTests
//

INSTANTIATE_TEST_SUITE_P(smoke, SdpAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Batch{8}),                //
                                            ::testing::Values(SourceSeqLen{512}),       //
                                            ::testing::Values(TargetSeqLen{256}),       //
                                            ::testing::Values(QKEmbeddingSize{64}),     //
                                            ::testing::Values(VEmbeddingSize{32}),      //
                                            ::testing::Values(IsCasual{false}),         //
                                            ::testing::Values(HasAttentionMask{true}),  //
                                            ::testing::Values(HasScale{true}),          //
                                            ::testing::ValuesIn(inputPrecision)         //
                                            ),
                         PrintTestCaseName());

}  // namespace ov::test
