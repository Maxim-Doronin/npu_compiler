//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test::behavior {

class WsSameBufferConstantsSubGraphTest : public VpuOv2LayerTest {
    void setupSpecialEnvironment() {
        const auto filePrefix = ov::test::utils::generateTestFilePrefix();
        _xmlPath = filePrefix + "same_buf_constants_test" + ".xml";
        _binPath = filePrefix + "same_buf_constants_test" + ".bin";

        // Note: both PLUGIN and DRIVER are OK here, but PLUGIN is generally
        // preferred for weights separation.
        configuration["NPU_COMPILER_TYPE"] = "PLUGIN";
        configuration["ENABLE_WEIGHTLESS"] = "YES";
        configuration["WEIGHTS_PATH"] = _binPath;
    }

    void TearDown() override {
        std::remove(_xmlPath.c_str());
        std::remove(_binPath.c_str());
    }

    void SetUp() override {
        setupSpecialEnvironment();

        inType = ov::element::f32;

        const ov::Shape inputShape0{1, 2, 3};
        const ov::Shape inputShape1{1, 2, 3};

        init_input_shapes(static_shapes_to_test_representation({inputShape0, inputShape1}));

        const auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto input2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));

        // Note: two separate constant nodes with the same data - this is to
        // make use of the model compression.
        const auto const1 = ov::op::v0::Constant::create(ov::element::f32, inputShape0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        const auto multiply1 = std::make_shared<ov::op::v1::Multiply>(input1, const1);

        const auto const2 = ov::op::v0::Constant::create(ov::element::f32, inputShape0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        const auto multiply2 = std::make_shared<ov::op::v1::Multiply>(input2, const2);

        const auto add = std::make_shared<ov::op::v1::Add>(multiply1, multiply2);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
        const auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2},
                                                       "WsSameBufferConstantsSubGraphTest");

        // Note: this test requires model serialization and de-serialization,
        // with compression enabled. so that the OV constants end up pointing to
        // the same buffer.
        ov::pass::Serialize(_xmlPath, _binPath).run_on_model(model);
        function = core->read_model(_xmlPath, _binPath);
    }

private:
    std::string _xmlPath{};
    std::string _binPath{};
};

TEST_F(WsSameBufferConstantsSubGraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(WsSameBufferConstantsSubGraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(WsSameBufferConstantsSubGraphTest, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test::behavior
