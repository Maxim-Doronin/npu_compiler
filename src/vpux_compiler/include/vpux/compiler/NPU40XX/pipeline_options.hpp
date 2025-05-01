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
        public ReferenceSWOptions<ReferenceSWOptions40XX>,
        public vpux::BatchCompileOptionsAdapter {
    ReferenceSWOptions40XX(): vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
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
    // Due to multiple inheritance, 'DefaultHWOptions40XX' has multiple definitions of 'createFromString' method
    // here we assume that we are interested in a "final" method that includes parameters from all parent classes
    using mlir::PassPipelineOptions<DefaultHWOptions40XX>::createFromString;
};

//
// BackendCompilationOptions40XX
//

struct BackendCompilationOptions40XX final : public BackendCompilationOptionsBase<BackendCompilationOptions40XX> {};

//
// BackendCompilationOptions40XX
//

void setupPWLMCompilationParams(int optimizationLevel, DefaultHWOptions40XX& compilationOptions, bool useWlm);

}  // namespace vpux
