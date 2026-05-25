//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ci_compliant_name.hpp"

#include <gtest/gtest.h>

using ov::test::utils::isCICompliantTestName;

TEST(CICompliantNameCheck, SuiteWithLayerTestIsCompliant) {
    EXPECT_TRUE(isCICompliantTestName("EltwiseLayerTest", "smoke_test/0"));
}

TEST(CICompliantNameCheck, SuiteWithTestKindSubgraphIsCompliant) {
    EXPECT_TRUE(isCICompliantTestName("DecomposeLayerTestKindSubgraph", "test/0"));
}

TEST(CICompliantNameCheck, TestNameWithTestKindSubgraphIsCompliant) {
    EXPECT_TRUE(isCICompliantTestName("SomeSuite", "NPU4000_TestKindSubgraph/0"));
}

TEST(CICompliantNameCheck, SuiteWithBehaviorTestIsCompliant) {
    EXPECT_TRUE(isCICompliantTestName("DynamicStridesBehaviorTest", "All4DTilingPermutations/0"));
}

TEST(CICompliantNameCheck, TestNameWithBehaviorTestIsCompliant) {
    EXPECT_TRUE(isCICompliantTestName("ElfConfigTests", "smoke_BehaviorTest_ELF/0"));
}

TEST(CICompliantNameCheck, PlainNamesAreNotCompliant) {
    EXPECT_FALSE(isCICompliantTestName("MyTestSuite", "myTestCase"));
}

TEST(CICompliantNameCheck, EmptyNamesAreNotCompliant) {
    EXPECT_FALSE(isCICompliantTestName("", ""));
}

TEST(CICompliantNameCheck, CaseSensitiveMismatchIsNotCompliant) {
    EXPECT_FALSE(isCICompliantTestName("EltwiseLayertest", "smoke/0"));
    EXPECT_FALSE(isCICompliantTestName("NPU4000_Testkindsubgraph", "test/0"));
    EXPECT_FALSE(isCICompliantTestName("Behaviortest", "smoke/0"));
}

TEST(CICompliantNameCheck, PartialMatchIsNotCompliant) {
    EXPECT_FALSE(isCICompliantTestName("LayerSuite", "LayerCase"));
    EXPECT_FALSE(isCICompliantTestName("SubgraphTest", "Subgraph/0"));
    EXPECT_FALSE(isCICompliantTestName("TestKindSuite", "TestKindCase"));
}
