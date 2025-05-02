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
        public ReferenceSWOptions<ReferenceSWOptions37XX>,
        public vpux::BatchCompileOptionsAdapter {
    ReferenceSWOptions37XX(): vpux::BatchCompileOptionsAdapter(static_cast<mlir::detail::PassOptions&>(*this)) {
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
    // Due to multiple inheritance, 'DefaultHWOptions37XX' has multiple definitions of 'createFromString' method
    // here we assume that we are interested in a "final" method that includes parameters from all parent classes
    using mlir::PassPipelineOptions<DefaultHWOptions37XX>::createFromString;
};

}  // namespace vpux
