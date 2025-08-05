//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class MappedInferenceRewriter : public VPUASMSymbolizationPattern<VPUMI40XX::MappedInferenceOp> {
public:
    using Base::Base;

    MappedInferenceRewriter(mlir::func::FuncOp netFunc, SymbolizationTypeConverter& typeConverter, SymbolMapper& mapper,
                            SectionMapper& sectionMap, mlir::MLIRContext* ctx, Logger log, bool disableDmaSwFifo)
            : VPUASMSymbolizationPattern<VPUMI40XX::MappedInferenceOp>(netFunc, typeConverter, mapper, sectionMap, ctx,
                                                                       log),
              _disableDmaSwFifo(disableDmaSwFifo) {
    }
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI40XX::MappedInferenceOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(VPUMI40XX::MappedInferenceOp, size_t) override;

private:
    bool _disableDmaSwFifo;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
