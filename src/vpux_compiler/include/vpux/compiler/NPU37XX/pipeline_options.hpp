//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux {

//
// ReferenceSWOptions37XX
//

struct ReferenceSWOptions37XX final :
        public PublicOptions,
        public ReferenceSWOptions<ReferenceSWOptions37XX>,
        public vpux::BatchCompileOptionsAdapter {
    ReferenceSWOptions37XX(): vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
    }
    ReferenceSWOptions37XX(VPU::ArchKind arch)
            : PublicOptions(arch), vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
    }

    static std::unique_ptr<ReferenceSWOptions37XX> createFromString(StringRef options, VPU::ArchKind arch) {
        auto result = std::make_unique<ReferenceSWOptions37XX>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }

    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(false)};
    BoolOption enableDecomposeGRUSequence{*this, "decompose-gru-sequence",
                                          llvm::cl::desc("Enable decompose-gru-sequence pass"), llvm::cl::init(false)};
};

//
// DefaultHWOptions37XX
//

struct DefaultHWOptions37XX final :
        public IE::arch37xx::DefaultHWOptions,
        VPU::arch37xx::DefaultHWOptions,
        VPUIP::arch37xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions37XX> {
    DefaultHWOptions37XX() = default;
    DefaultHWOptions37XX(VPU::ArchKind arch): PublicOptions(arch) {
    }

    static std::unique_ptr<DefaultHWOptions37XX> createFromString(StringRef options, VPU::ArchKind arch) {
        auto result = std::make_unique<DefaultHWOptions37XX>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }
};

}  // namespace vpux
