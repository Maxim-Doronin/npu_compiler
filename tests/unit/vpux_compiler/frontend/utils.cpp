//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend/utils.hpp"

namespace test_utils {
bool isStrEqualSpaceAgnostic(const std::string& lhs, const std::string& rhs) {
    auto isspace = [](char ch) {
        return std::isspace(static_cast<unsigned char>(ch));
    };
    auto seekWhile = [](auto& beginIt, const auto& endIt, auto cond) {
        while (cond(*beginIt) && beginIt != endIt) {
            ++beginIt;
            continue;
        }
    };

    auto lhsIt = lhs.begin();
    auto rhsIt = rhs.begin();
    for (; lhsIt != lhs.end() && rhsIt != rhs.end();) {
        seekWhile(lhsIt, lhs.end(), isspace);
        seekWhile(rhsIt, rhs.end(), isspace);
        if (lhsIt == lhs.end() || rhsIt == rhs.end()) {
            return rhsIt == rhs.end() && lhsIt == lhs.end();
        }
        if (*lhsIt != *rhsIt) {
            return false;
        }
        ++lhsIt;
        ++rhsIt;
    }
    seekWhile(lhsIt, lhs.end(), isspace);
    seekWhile(rhsIt, rhs.end(), isspace);
    return lhsIt == lhs.end() && rhsIt == rhs.end();
}
}  // namespace test_utils
