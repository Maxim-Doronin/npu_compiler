//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

//
// Default location for performance debug files to be stored
//
constexpr StringLiteral perfDebugFilesRoot = "perf_debug_dumps";
constexpr StringLiteral perfDebugDefaultIRFilter =
        "MultiClusterStrategyAssignment|MergeVfSubgraphs|OptimizeConvertDMAOp|FeasibleAllocation";
constexpr StringLiteral perfDebugDefaultLogFilter =
        "DeveloperConfig|MultiClusterStrategyAssignment|MergeVfSubgraphs|OptimizeConvertDMAOp|FeasibleAllocation";
constexpr StringRef dotDebugDefaultPasses[] = {"multi-cluster-strategy-assignment", "merge-vertical-fusion-subgraphs",
                                               "optimize-convert-dma-op"};

void parseEnv(StringRef envVarName, std::string& var);
void parseEnv(StringRef envVarName, bool& var);

bool isPerfDebugMode();
constexpr bool isDeveloperBuild() {
#ifdef VPUX_DEVELOPER_BUILD
    return true;
#else
    return false;
#endif
}

}  // namespace vpux
