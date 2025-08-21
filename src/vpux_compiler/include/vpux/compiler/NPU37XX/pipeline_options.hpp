//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux {
//
// DefaultHWOptions37XX
//

struct DefaultHWOptions37XX final :
        public IE::arch37xx::DefaultHWOptions,
        VPU::arch37xx::DefaultHWOptions,
        VPUIP::arch37xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions37XX> {
    DefaultHWOptions37XX() = default;
    DefaultHWOptions37XX(config::ArchKind arch): PublicOptions(arch) {
    }

    static std::unique_ptr<DefaultHWOptions37XX> createFromString(StringRef options, config::ArchKind arch) {
        auto result = std::make_unique<DefaultHWOptions37XX>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }
};

}  // namespace vpux
