//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_npu/ops/flash_attention_tile.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <limits>
#include <memory>
#include <random>

namespace ov::test {

namespace {

const int32_t SEED = 42;

PRETTY_PARAM(Heads, BoundedDim);            // H
PRETTY_PARAM(SourceSeqLen, BoundedDim);     // S
PRETTY_PARAM(TargetSeqLen, BoundedDim);     // L
PRETTY_PARAM(QKEmbeddingSize, BoundedDim);  // E
PRETTY_PARAM(VEmbeddingSize, BoundedDim);   // Ev

// Absent      - no attention mask
// Broadcasted - [1, 1, L, S]
// Full        - [1, H, L, S]
enum struct Mask { Absent, Broadcasted, Full };
PRETTY_PARAM(AttentionMask, Mask);

PRETTY_PARAM(IsHead, bool);
PRETTY_PARAM(IsTail, bool);

ov::ParameterVector generateInputParams(const std::vector<ov::PartialShape>& inputDynamicShapes,
                                        AttentionMask attentionMask) {
    ov::ParameterVector inputParams;
    const auto requiredParamNames =
            std::array<std::string_view, 6>{"query", "key", "value", "running_output", "running_max", "running_sum"};

    const auto inputType = ov::element::f32;
    for (auto [i, paramName] : vpux::enumerate(requiredParamNames)) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[i]));
        inputParams[i]->set_friendly_name(std::string{paramName});
    }

    const auto attentionMaskIdx = 6;
    if (attentionMask.value() != Mask::Absent) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[attentionMaskIdx]));
        inputParams.back()->set_friendly_name("attention_mask");
    }

    return inputParams;
}

template <typename Derived>
class FlashAttentionTileLayerTestBase : public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        std::mt19937 rng{SEED};

        const auto inputType = ov::element::f32;
        const auto& modelInputs = function->inputs();

        const auto hasAttentionMask = (attentionMask != Mask::Absent);
        const auto numRequiredInputs = 6;

        // Generate QKV
        for (int i = 0; i < 3; i++) {
            ov::test::utils::InputGenerateData data(-1, 2, 32, SEED);
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(inputType, targetInputStaticShapes[i], data);
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }

        // Generate initial running out/max/sum constants
        for (int i = 3; i < 6; i++) {
            const auto isRunningMaxTensor = (i == 4);
            const auto tensorValue = isRunningMaxTensor ? -std::numeric_limits<float>::infinity() : 0;

            auto tensor = ov::Tensor(inputType, targetInputStaticShapes[i]);
            auto data = tensor.data<float>();
            std::fill_n(data, tensor.get_size(), tensorValue);

            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }

        // Generate random sparse matrix
        if (hasAttentionMask) {
            const auto attentionMaskIdx = 6;

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            const auto sparsity = 0.3f;

            auto tensor = ov::Tensor(inputType, targetInputStaticShapes[attentionMaskIdx]);
            auto data = tensor.data<float>();
            const auto size = tensor.get_size();

            for (size_t i = 0; i < size; i++) {
                data[i] = dist(rng) < sparsity ? -std::numeric_limits<float>::infinity() : 0.0f;
            }

            inputs.insert({modelInputs[attentionMaskIdx].get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), 3u);
        ASSERT_EQ(actual.size(), 3u);

        const std::array<std::string_view, 3> names = {"Output", "Max", "Sum"};
        std::ostringstream failures;
        auto failureCount = 0;

        auto check = [&](size_t i, const auto* expPtr, const auto* actPtr) {
            const auto& shape = expected[i].get_shape();
            ASSERT_EQ(shape, actual[i].get_shape()) << "Shape mismatch for " << names[i];

            auto mismatchCount = 0;
            auto worstIdx = 0;
            auto maxAbs = 0.0;
            auto maxRel = 0.0;
            auto worstExp = 0.0;
            auto worstAct = 0.0;

            auto min = std::numeric_limits<float>::max();
            auto max = -std::numeric_limits<float>::infinity();

            const auto expectedSize = static_cast<int>(expected[i].get_size());
            for (auto j = 0; j < expectedSize; ++j) {
                auto e = static_cast<float>(expPtr[j]);
                auto a = static_cast<float>(actPtr[j]);
                auto absDiff = std::abs(e - a);
                auto relDiff = (e != 0.0) ? absDiff / std::abs(e) : absDiff;

                min = std::min(min, a);
                max = std::max(max, a);

                if (absDiff > abs_threshold && relDiff > rel_threshold) {
                    ++mismatchCount;
                    if (absDiff > maxAbs) {
                        maxAbs = absDiff;
                        maxRel = relDiff;
                        worstIdx = j;
                        worstExp = e;
                        worstAct = a;
                    }
                }
            }

            if (mismatchCount > 0) {
                ++failureCount;
                failures << "\n"
                         << names[i] << " [" << shape << "]:"
                         << "\n  Mismatches: " << mismatchCount << " / " << expected[i].get_size()
                         << "\n  Worst at flat index " << worstIdx << ":"
                         << " expected=" << worstExp << ", actual=" << worstAct << ", abs=" << maxAbs
                         << ", rel=" << maxRel;
            }
        };

        const auto isTail = this->isTail;

        check(0, expected[0].data<const float>(), actual[0].data<const float>());
        if (!isTail) {
            check(1, expected[1].data<const float>(), actual[1].data<const float>());
            check(2, expected[2].data<const float>(), actual[2].data<const float>());
        }

        ASSERT_EQ(failureCount, 0) << "Flash attention failed (abs=" << abs_threshold << ", rel=" << rel_threshold
                                   << "):" << failures.str();
    }

    void SetUp() override {
        VpuOv2LayerTest::SetUp();

        auto params = static_cast<Derived*>(this)->GetParam();

        this->attentionMask = std::get<AttentionMask>(params).value();
        this->isHead = std::get<IsHead>(params).value();
        this->isTail = std::get<IsTail>(params).value();

        const auto inputShapes = static_cast<Derived*>(this)->generateInputShapes(params);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, attentionMask);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto config = ov::intel_npu::op::FlashAttentionTile::Config();
        config.is_head = std::get<IsHead>(params).value();
        config.is_tail = std::get<IsTail>(params).value();

        auto flashAttentionTile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(inputs, config);
        flashAttentionTile->set_friendly_name("flash_attention_tile");

        ov::OutputVector outputs;
        for (const auto& output : flashAttentionTile->outputs()) {
            outputs.push_back(output);
        }

        function = std::make_shared<ov::Model>(outputs, inputParams, "FlashAttentionTile");
    }

public:
    Mask attentionMask = Mask::Absent;
    bool isHead = true;
    bool isTail = true;
};

using FlashAttentionTileParams =
        std::tuple<Heads, SourceSeqLen, TargetSeqLen, QKEmbeddingSize, VEmbeddingSize, AttentionMask, IsHead, IsTail>;

class FlashAttentionTileLayerTest :
        public FlashAttentionTileLayerTestBase<FlashAttentionTileLayerTest>,
        public ::testing::WithParamInterface<FlashAttentionTileParams> {
public:
    std::vector<InputShape> generateInputShapes(FlashAttentionTileParams params) {
        const auto& [heads, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, attentionMask, isHead,
                     isTail] = params;

        auto qShape = generateTestShape(heads.value(), targetSeqLen.value(), qkEmbeddingSize.value());
        auto kShape = generateTestShape(heads.value(), sourceSeqLen.value(), qkEmbeddingSize.value());
        auto vShape = generateTestShape(heads.value(), sourceSeqLen.value(), vEmbeddingSize.value());

        auto runningOutShape = generateTestShape(heads.value(), targetSeqLen.value(), vEmbeddingSize.value());
        auto runningMaxShape = generateTestShape(heads.value(), targetSeqLen.value());
        auto runningSumShape = generateTestShape(heads.value(), targetSeqLen.value());

        auto inputShapes =
                std::vector<InputShape>{qShape, kShape, vShape, runningOutShape, runningMaxShape, runningSumShape};

        if (attentionMask.value() == Mask::Full) {
            auto attentionMaskShape = generateTestShape(heads.value(), targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        } else if (attentionMask.value() == Mask::Broadcasted) {
            auto attentionMaskShape = generateTestShape(1, targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        }

        return inputShapes;
    }
};

}  // namespace

TEST_P(FlashAttentionTileLayerTest, NPU5010_HW) {
    abs_threshold = 0.001;
    rel_threshold = 0.001;
    run(Platform::NPU5010);
}

//
// FlashAttentionTileLayerTest
//

INSTANTIATE_TEST_SUITE_P(smoke, FlashAttentionTileLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),              //
                                            ::testing::Values(SourceSeqLen{160}),     //
                                            ::testing::Values(TargetSeqLen{64}),      //
                                            ::testing::Values(QKEmbeddingSize{128}),  //
                                            ::testing::Values(VEmbeddingSize{128}),   //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{
                                                    Mask::Full, Mask::Broadcasted, Mask::Absent}),  //
                                            ::testing::ValuesIn(std::vector<IsHead>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<IsTail>{true, false})   //
                                            ),                                                      //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_BigSequenceLength, FlashAttentionTileLayerTest,
                         ::testing::Combine(::testing::Values(Heads{32}),                                        //
                                            ::testing::Values(SourceSeqLen{1024}),                               //
                                            ::testing::Values(TargetSeqLen{1024}),                               //
                                            ::testing::Values(QKEmbeddingSize{128}),                             //
                                            ::testing::Values(VEmbeddingSize{128}),                              //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted}),  //
                                            ::testing::ValuesIn(std::vector<IsHead>{false}),                     //
                                            ::testing::ValuesIn(std::vector<IsTail>{true})                       //
                                            ),                                                                   //
                         PrintTestCaseName());

}  // namespace ov::test
