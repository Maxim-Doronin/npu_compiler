//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset1.hpp>
#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"

#include "common/quantization_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/depth_to_space.hpp"

// Subgraph:
//
//  [input]
//     |
//   {FQ}
//     |
//   (D2S)
//   /   \  ... -> (Expand fused into D2S before DPU.Add)
//  ( Add )
//     |
// [output]

namespace ov::test::subgraph {

using FuseD2sDynExpandParams = std::tuple<ov::test::InputShape,  // Input shapes
                                          std::size_t,           // Block size
                                          ov::element::Type,     // inType
                                          bool,                  // Enable FQ
                                          bool>;                 // Enable Preprocess

class FuseD2sExpandCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FuseD2sDynExpandParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseD2sDynExpandParams> obj) {
        const auto& [inShape, bs, inType, enableFQ, enablePreprocess] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShapeSize={" << inShape.first.to_string() << "}" << sep;
        result << "BlockSize=" << bs << sep;
        result << "inType=" << inType << sep;
        result << "EnableFQ=" << enableFQ << sep;
        result << "EnablePreprocess=" << enablePreprocess << sep;
        return result.str();
    }
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-d2s-to-transposed-conv-conversion=false "
                                                                       "enable-ops-as-dma=false "
                                                                       "enable-fuse-d2s-expand=true";
    }
    void SetUp() override {
        const auto& [inShape, blockSize, inType, enableFQ, enablePreprocess] = GetParam();

        init_input_shapes({inShape});

        ov::ParameterVector params;
        params.reserve(inputDynamicShapes.size());
        for (const auto& shape : inputDynamicShapes) {
            auto inParam = std::make_shared<ov::op::v0::Parameter>(inType, shape);
            params.push_back(inParam);
        }
        ov::Output<ov::Node> d2sInput = params[0]->output(0);
        if (enableFQ) {
            d2sInput = utils::makeFakeQuantize(d2sInput, ov::element::f16, 256,
                                               FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                               ->get_default_output();
        }
        const auto mode = ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
        auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(d2sInput, mode, blockSize);

        const auto add = std::make_shared<ov::opset6::Add>(d2s->output(0), d2s->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(add->output(0))};

        function = std::make_shared<ov::Model>(results, params, "D2sExpand");
        // Applying NHWC layout to input to avoid insertion of Permute before D2S
        // Partially done to avoid E#192054
        if (enablePreprocess) {
            auto preProc = ov::preprocess::PrePostProcessor(function);
            preProc.input().tensor().set_layout(ov::Layout("NHWC"));
            preProc.input().model().set_layout(ov::Layout("NCHW"));
            function = preProc.build();
        }
    }
};

class FuseD2sExpandCommon_HostCompile : public FuseD2sExpandCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-d2s-to-transposed-conv-conversion=false "
                                                                       "enable-ops-as-dma=false "
                                                                       "enable-fuse-d2s-expand=true "
                                                                       "adjust-input-shape=false "
                                                                       "adjust-convolution-shape=false "
                                                                       "optimize-slice-with-stride=false";
    }
};

TEST_P(FuseD2sExpandCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseD2sExpandCommon_HostCompile, NPU4000_HC) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "Host Pipeline does not support inference yet: C#164943";
    });
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(FuseD2sExpandCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseD2sExpandCommon_HostCompile, NPU5010_HC) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "Host Pipeline does not support inference yet: C#164943";
    });
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

// Important: use odd W/H dims to avoid 'inputShape1' reshaped into {1, 16, H/2, W}
//            and have the Expand C=8 to C=16 instead
// as a result fusion of D2S + Expand will be applied only with those shapes

std::vector<FuseD2sDynExpandParams> params1 = {{/* inputShape = */ generateTestShape(1, 8, 11, 11),
                                                /* blockSize = */ 2,
                                                /* inType = */ ov::element::f16,
                                                /* enableFQ = */ false,
                                                /* enablePreprocess = */ false}};

std::vector<FuseD2sDynExpandParams> params2 = {{/* inputShape = */ generateTestShape(1, 12, 3, 123),
                                                /* blockSize = */ 2,
                                                /* inType = */ ov::element::f16,
                                                /* enableFQ = */ true,
                                                /* enablePreprocess = */ false}};

std::vector<FuseD2sDynExpandParams> params3 = {{/* inputShape = */ generateTestShape(1, 12, 1079, 319),
                                                /* blockSize = */ 2,
                                                /* inType = */ ov::element::f16,
                                                /* enableFQ = */ false,
                                                /* enablePreprocess = */ false}};

std::vector<FuseD2sDynExpandParams> params4 = {{/* inputShape = */ generateTestShape(1, 12, 1079, 319),
                                                /* blockSize = */ 2,
                                                /* inType = */ ov::element::f16,
                                                /* enableFQ = */ true,
                                                /* enablePreprocess = */ false}};

std::vector<FuseD2sDynExpandParams> paramsDyn1 = {{/* inputShape = */ generateTestShape(1, 12, 1079, 319_Dyn),
                                                   /* blockSize = */ 2,
                                                   /* inType = */ ov::element::f16,
                                                   /* enableFQ = */ false,
                                                   /* enablePreprocess = */ true}};

std::vector<FuseD2sDynExpandParams> paramsDyn2 = {{/* inputShape = */ generateTestShape(1, 12, 1079, 319_Dyn),
                                                   /* blockSize = */ 2,
                                                   /* inType = */ ov::element::f16,
                                                   /* enableFQ = */ true,
                                                   /* enablePreprocess = */ true}};

std::vector<FuseD2sDynExpandParams> paramsDyn3 = {{/* inputShape = */ generateTestShape(1, 12, 1079_Dyn, 319_Dyn),
                                                   /* blockSize = */ 2,
                                                   /* inType = */ ov::element::f16,
                                                   /* enableFQ = */ false,
                                                   /* enablePreprocess = */ true}};

INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_f16, FuseD2sExpandCommon, ::testing::ValuesIn(params1),
                         FuseD2sExpandCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_u8q, FuseD2sExpandCommon, ::testing::ValuesIn(params2),
                         FuseD2sExpandCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_big_f16, FuseD2sExpandCommon, ::testing::ValuesIn(params3),
                         FuseD2sExpandCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_big_u8q, FuseD2sExpandCommon, ::testing::ValuesIn(params4),
                         FuseD2sExpandCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_Dyn_f16, FuseD2sExpandCommon_HostCompile, ::testing::ValuesIn(paramsDyn1),
                         FuseD2sExpandCommon_HostCompile::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_Dyn_u8q, FuseD2sExpandCommon_HostCompile, ::testing::ValuesIn(paramsDyn2),
                         FuseD2sExpandCommon_HostCompile::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_2Dyn_f16, FuseD2sExpandCommon_HostCompile, ::testing::ValuesIn(paramsDyn3),
                         FuseD2sExpandCommon_HostCompile::getTestCaseName);

}  // namespace
