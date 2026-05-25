//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

namespace ov::test::utils {

inline bool isCICompliantTestName(std::string_view suiteName, std::string_view testName) {
    if (suiteName.find("LayerTest") != std::string::npos) {
        return true;
    }
    if ((suiteName.find("TestKindSubgraph") != std::string::npos) ||
        (testName.find("TestKindSubgraph") != std::string::npos)) {
        return true;
    }
    if ((suiteName.find("BehaviorTest") != std::string::npos) || (testName.find("BehaviorTest") != std::string::npos)) {
        return true;
    }
    return false;
}

}  // namespace ov::test::utils
