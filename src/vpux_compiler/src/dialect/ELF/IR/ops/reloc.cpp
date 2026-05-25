//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"

#include <vpux_elf/writer.hpp>

#include <mlir/IR/Builders.h>
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"

using namespace vpux;

void vpux::ELF::RelocOp::serialize(elf::writer::Relocation* relocation, vpux::ELF::SymbolMapType& symbolMap,
                                   vpux::ELF::DmaSymbolMapType& dmaSymbolMap) {
    auto symbolRef = ELF::lookupNearestSymbolFrom(getOperation(), getSourceSymbolAttr());

    auto symbolMapEntry = symbolMap.find(symbolRef);
    auto dmaSymbolMapEntry = dmaSymbolMap.find(symbolRef);
    if (symbolMapEntry != symbolMap.end()) {
        auto symbolEntry = symbolMapEntry->second;
        relocation->setSymbol(symbolEntry);
    } else if (dmaSymbolMapEntry != dmaSymbolMap.end()) {
        auto dmaSymbolEntry = dmaSymbolMapEntry->second;
        relocation->setSpecialSymbol(dmaSymbolEntry->getIndex());
    } else {
        VPUX_THROW("Unable to locate symbol entry for relocation");
    }

    auto relocType = getRelocationType();
    auto relocAddend = getAddend();

    relocation->setType(static_cast<elf::Elf_Word>(relocType));
    relocation->setOffset(getOffset());
    relocation->setAddend(relocAddend);
}

void vpux::ELF::RelocOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, int64_t offset,
                               ::mlir::SymbolRefAttr sourceSymbol, vpux::ELF::RelocationType relocationType,
                               int64_t addend, llvm::StringRef description) {
    build(odsBuilder, odsState, offset, sourceSymbol, relocationType, addend, odsBuilder.getStringAttr(description));
}

void vpux::ELF::RelocOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, int64_t offset,
                               ::mlir::SymbolRefAttr sourceSymbol, vpux::ELF::RelocationType relocationType,
                               int64_t addend) {
    build(odsBuilder, odsState, offset, sourceSymbol, relocationType, addend, "");
}
