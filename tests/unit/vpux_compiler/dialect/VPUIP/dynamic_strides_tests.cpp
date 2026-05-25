//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dynamic_strides_utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

namespace {

struct DmaCanonicalizationTestParams {
    MemStrides inStrides;
    Bit inElemSize;
    MemStrides outStrides;
    Bit outElemSize;
    bool expectedCompatible;
};

using DynamicStridesCompatibilityTest = ::testing::TestWithParam<DmaCanonicalizationTestParams>;

TEST_P(DynamicStridesCompatibilityTest, BasicTest) {
    auto param = GetParam();
    auto compatible =
            VPUIP::areStridesCompatible(param.inStrides, param.inElemSize, param.outStrides, param.outElemSize);

    EXPECT_EQ(compatible, param.expectedCompatible);
}

INSTANTIATE_TEST_SUITE_P(
        DynamicStridesCompatibilityTestSuite, DynamicStridesCompatibilityTest,
        testing::Values(
                DmaCanonicalizationTestParams{
                        {Bit(12), Bit(4), Bit(1)}, Bit(1), {Bit(12), Bit(12), Bit(4), Bit(4), Bit(1)}, Bit(1), true},
                DmaCanonicalizationTestParams{
                        {Bit(12), Bit(4), Bit(4), Bit(1)}, Bit(1), {Bit(12), Bit(4), Bit(1)}, Bit(1), true},
                DmaCanonicalizationTestParams{
                        {Bit(12), Bit(3), Bit(1)}, Bit(1), {Bit(12), Bit(4), Bit(1)}, Bit(1), false},
                DmaCanonicalizationTestParams{
                        {Bit(24), Bit(8), Bit(2)}, Bit(2), {Bit(12), Bit(4), Bit(1)}, Bit(1), true},
                DmaCanonicalizationTestParams{
                        {Bit(24), Bit(24), Bit(8), Bit(2), Bit(2)}, Bit(2), {Bit(4), Bit(1)}, Bit(1), true}));
}  // namespace
