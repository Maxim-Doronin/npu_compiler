//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class KernelParamsRewriter : public VPUASMSymbolizationPattern<VPUMI40XX::KernelParamsOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI40XX::KernelParamsOp op, SymbolMapper& mappper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(VPUMI40XX::KernelParamsOp op, size_t) override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
