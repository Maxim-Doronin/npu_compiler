//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/env.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/softmax.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace utils = ov::test::utils;

class ProfilingTempReportEnv {
public:
    ProfilingTempReportEnv(const char* type = "JSON") {
        auto filename = std::tmpnam(tempName);
        EXPECT_FALSE(filename == nullptr);
        EXPECT_TRUE(vpux::env::getEnvVar("NPU_PRINT_PROFILING") == std::nullopt);
        vpux::env::setEnvVar("NPU_PRINT_PROFILING", type);
        vpux::env::setEnvVar("NPU_PROFILING_OUTPUT_FILE", filename);
    }
    void cleanup() {
        std::remove(tempName);
        vpux::env::unsetEnvVar("NPU_PRINT_PROFILING");
        vpux::env::unsetEnvVar("NPU_PROFILING_OUTPUT_FILE");
    }
    std::string readAsString() {
        std::ifstream input(tempName);
        input.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        return std::string(std::istreambuf_iterator<std::string::value_type>(input), {});
    }

private:
    ProfilingTempReportEnv(const ProfilingTempReportEnv&) = delete;
    void operator=(const ProfilingTempReportEnv&) = delete;

    char tempName[L_tmpnam] = "";
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, std::optional<T> opt) {
    if (opt.has_value()) {
        return stream << *opt;
    } else {
        return stream << "<nullopt>";
    }
}

class ProfilingSubgraphTestBase : public ov::test::VpuOv2LayerTest {
protected:
    void SetUp() override {
        createSubgraphFunction();
    }

    void createSubgraphFunction() {
        const auto type = ov::element::f32;

        ov::Shape inputShape = {1, 3, 128, 128};
        auto param = std::make_shared<ov::op::v0::Parameter>(type, inputShape);

        const std::vector<float> exponent({1.0f});
        const auto power_const = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{1}, exponent);
        auto pow = std::make_shared<ov::op::v1::Power>(param, power_const);
        auto add = std::make_shared<ov::op::v1::Add>(param, pow->output(0));
        auto softmax = std::make_shared<ov::op::v1::Softmax>(add, /*axis*/ 1);
        auto result = std::make_shared<ov::op::v0::Result>(softmax->output(0));

        ov::ParameterVector params{param};
        ov::ResultVector results{result};
        function = std::make_shared<ov::Model>(results, params, "ProfSubgraph");

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));
    }
};

typedef int VoidParam;

class ProfilingSubgraphSanityTest : public ProfilingSubgraphTestBase, public testing::WithParamInterface<VoidParam> {
protected:
    void runTest() {
        run(utils::getTestDeviceId());
    }
};

using ProfilingTestParams = std::tuple<std::string>;

class ProfilingSubgraphTest :
        public ProfilingSubgraphTestBase,
        public testing::WithParamInterface<ProfilingTestParams> {
protected:
    ProfilingSubgraphTest(): log(vpux::Logger::global().nest("prof", 0)) {
    }

    void runTest() {
        run(utils::getTestDeviceId());
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ProfilingSubgraphTest::ParamType>& info) {
        const auto& [compilerType] = info.param;
        return compilerType;
    }

private:
    void SetUp() override {
        const auto& [compilerType] = GetParam();
        configuration.emplace(ov::enable_profiling.name(), true);
        configuration.emplace(ov::intel_npu::compiler_type.name(), compilerType);

        ProfilingSubgraphTestBase::SetUp();

        // Extract layer names from the model
        for (const auto& layer : function->get_ops()) {
            if (layer->get_type_name() == std::string("Parameter")) {
                continue;
            }
            layerNames.insert(layer->get_friendly_name());
        }
    }

    virtual void infer() override {
        ov::test::VpuOv2LayerTest::infer();
        checkProfilingOutput();
    }

    void checkProfilingOutput() {
        ProfilingTempReportEnv tempReport;
        auto profData = inferRequest.get_profiling_info();
        auto jsonReport = tempReport.readAsString();
        tempReport.cleanup();

        ASSERT_TRUE(profData.size() > 0);

        for (const auto& profInfo : profData) {
            checkLayer(profInfo, jsonReport);
        }
    }

    void checkLayer(ov::ProfilingInfo profInfo, const std::string& jsonReport) {
        std::set<std::string> layerExecTypes = {"DPU", "Shave", "DMA"};

        auto inTheRange = [](long long val, long long min, long long max) {
            return val >= min && val <= max;
        };
        auto inAllowedExecTypes = [&](std::string execType) {
            return layerExecTypes.find(execType) != layerExecTypes.end();
        };

        const auto& layerName = profInfo.node_name;
        const auto& layerType = profInfo.node_type;
        const auto cpuTime = profInfo.cpu_time.count();
        const auto realTime = profInfo.real_time.count();
        log.info("Layer {0} '{1}' ({2}) cpu: {3} us, real: {4} us", profInfo.node_type, layerName, profInfo.exec_type,
                 cpuTime, realTime);

        ASSERT_PRED1(inAllowedExecTypes, profInfo.exec_type);
        ASSERT_TRUE(layerNames.count(layerName) == 1 || layerType == "Parameter") << "Unexpected layer:" << layerName;
        ASSERT_TRUE(jsonReport.find(layerName) != std::string::npos)
                << "Could not find the expected layer name: " << layerName << " in the json report.";
        // real time can be smaller than the cpu time depending on number of tiles
        // use course range here to detect corrupted results
        ASSERT_PRED3(inTheRange, cpuTime, 2, 100000) << "CPU time " << cpuTime << "us is out of range.";
        ASSERT_PRED3(inTheRange, realTime, 2, 100000) << "real time " << realTime << "us is out of range.";
    }

    std::set<std::string> layerNames;
    vpux::Logger log;
};

// Profiling disabled sanity test-case

TEST_P(ProfilingSubgraphSanityTest, ProfilingDisabledTest) {
    runTest();
}

INSTANTIATE_TEST_SUITE_P(precommit_BehaviorTest_ProfilingDisabledTest, ProfilingSubgraphSanityTest,
                         testing::ValuesIn(std::vector<VoidParam>{0}),
                         (utils::appendPlatformTypeTestName<ProfilingSubgraphSanityTest>));

// Profiling enabled test cases

TEST_P(ProfilingSubgraphTest, ProfilingTest) {
    runTest();
}

INSTANTIATE_TEST_SUITE_P(precommit_BehaviorTest_ProfilingTest, ProfilingSubgraphTest,
                         testing::Values("PLUGIN", "DRIVER"),
                         (utils::appendPlatformTypeTestName<ProfilingSubgraphTest>));
