//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/compat_string_check.hpp"

#include <gtest/gtest.h>

using namespace vpux::compat;

TEST(CompatStringTest, BasicString) {
    auto req = parseCompatibilityString("npu=5010;t=3;elf=2.0.0;mi=11.7.0");
    EXPECT_EQ(req.platformId, 5010);
    EXPECT_EQ(req.numTiles, 3);
}

TEST(CompatStringTest, MinimalString) {
    auto req = parseCompatibilityString("npu=5020;t=1");
    EXPECT_EQ(req.platformId, 5020);
    EXPECT_EQ(req.numTiles, 1);
}

TEST(CompatStringTest, MissingTiles) {
    EXPECT_THROW(parseCompatibilityString("npu=5020"), std::runtime_error);
}

TEST(CompatStringTest, IllegalAttribute) {
    EXPECT_THROW(parseCompatibilityString("npu=5010;t=2;unknown=1"), std::runtime_error);
}

TEST(CompatStringTest, InvalidNumber) {
    EXPECT_THROW(parseCompatibilityString("npu=5010;t=2A"), std::runtime_error);
}
