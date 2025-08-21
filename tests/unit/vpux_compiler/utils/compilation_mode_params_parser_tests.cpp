//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"
#include "vpux/compiler/compilation_options.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace vpux;

class CompilationModeParamsParserTest : public MLIR_UnitBase {};

TEST_F(CompilationModeParamsParserTest, EmptyOptions) {
    std::string params = "";
    auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                        /*logLevel=*/LogLevel::None);
    ASSERT_TRUE(result != nullptr);
}

TEST_F(CompilationModeParamsParserTest, PublicOptions) {
    {
        std::string params = "optimization-level=100";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
        EXPECT_EQ(result->optimizationLevel.getValue(), 100);
    }
    {
        std::string params = "optimization-level=200 performance-hint-override=randomEntry";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
        EXPECT_EQ(result->optimizationLevel.getValue(), 200);
        EXPECT_EQ(result->performanceHintOverride.getValue(), "randomEntry");
    }
}

TEST_F(CompilationModeParamsParserTest, PrivateOptions) {
    {
        std::string params = "schedule-trace-file-name=randomFileName";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
        EXPECT_NE(result->scheduleTraceFile.getValue(), "randomFileName");
    }
    {
        std::string params = "schedule-trace-file-name=randomFileName weights-sparsity-threshold=1234567890";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
        EXPECT_NE(result->scheduleTraceFile.getValue(), "randomFileName");
        EXPECT_NE(result->weightsSparsityThreshold.getValue(), 1234567890);
    }
}

TEST_F(CompilationModeParamsParserTest, MixedOptions) {
    {
        std::string params = "optimization-level=100 schedule-trace-file-name=randomFileName";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
        EXPECT_EQ(result->optimizationLevel.getValue(), 100);
        EXPECT_NE(result->scheduleTraceFile.getValue(), "randomFileName");
    }
    {
        std::string params = "optimization-level=200 schedule-trace-file-name=randomFileName "
                             "performance-hint-override=randomEntry weights-sparsity-threshold=1234567890";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
        EXPECT_EQ(result->optimizationLevel.getValue(), 200);
        EXPECT_NE(result->scheduleTraceFile.getValue(), "randomFileName");
        EXPECT_EQ(result->performanceHintOverride.getValue(), "randomEntry");
        EXPECT_NE(result->weightsSparsityThreshold.getValue(), 1234567890);
    }
}

TEST_F(CompilationModeParamsParserTest, InvalidOptions) {
    {
        // Note: correct option would be `optimization-level`
        std::string params = "my-optimization-level=100";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result == nullptr);
    }
    {
        // Note: correct option would be `optimization-level`
        std::string params = "performance-hint-override=randomEntry my-optimization-level=200";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result == nullptr);
    }
}

TEST_F(CompilationModeParamsParserTest, ValuesWithSpaces) {
    {
        std::string params = "function-outlining=\"repeating-blocks='min-ops-in-block=2 max-num-iterations=10'\"";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
    }
    {
        std::string params = "debatcher-settings={debatching-inlining-method=naive max-batch-number-disable-limit=-1}";
        auto result = parseOnlyPublic<DefaultHWOptions40XX>(params, config::ArchKind::NPU40XX, /*warnForPrivate=*/false,
                                                            /*logLevel=*/LogLevel::None);
        ASSERT_TRUE(result != nullptr);
    }
}
