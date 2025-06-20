//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux {

//
// ReferenceSWOptions40XX
//

struct ReferenceSWOptions40XX final :
        public PublicOptions,
        public ReferenceSWOptions<ReferenceSWOptions40XX>,
        public vpux::BatchCompileOptionsAdapter {
    ReferenceSWOptions40XX(): vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
    }
    ReferenceSWOptions40XX(VPU::ArchKind arch)
            : PublicOptions(arch), vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
    }

    static std::unique_ptr<ReferenceSWOptions40XX> createFromString(StringRef options, VPU::ArchKind arch) {
        auto result = std::make_unique<ReferenceSWOptions40XX>(arch);
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

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions40XX& options,
                                  Logger log = Logger::global());

//
// DefaultHWOptions40XX
//

struct DefaultHWOptions40XX final :
        public IE::arch40xx::DefaultHWOptions,
        VPU::arch40xx::DefaultHWOptions,
        VPUIP::arch40xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions40XX> {
    DefaultHWOptions40XX() = default;
    DefaultHWOptions40XX(VPU::ArchKind arch): PublicOptions(arch) {
    }

    static std::unique_ptr<DefaultHWOptions40XX> createFromString(StringRef options, VPU::ArchKind arch) {
        auto result = std::make_unique<DefaultHWOptions40XX>(arch);
        if (mlir::failed(result->parseFromString(options))) {
            return nullptr;
        }
        return result;
    }
};

//
// BackendCompilationOptions40XX
//

struct BackendCompilationOptions40XX final : public BackendCompilationOptionsBase<BackendCompilationOptions40XX> {};

//
// BackendCompilationOptions40XX
//

void setupParamsAccordingToOptimizationLevel(int optimizationLevel, DefaultHWOptions40XX& compilationOptions,
                                             bool useWlm);
void setupPWLMParams(DefaultHWOptions40XX& compilationOptions);

}  // namespace vpux
