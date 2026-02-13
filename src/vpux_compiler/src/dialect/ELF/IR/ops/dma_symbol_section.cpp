//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

using namespace vpux;

void ELF::DmaSymbolSectionOp::serialize(elf::Writer& writer, ELF::SectionMapType& sectionMap, ELF::SymbolMapType&,
                                        ELF::DmaSymbolMapType& dmaSymbolMap, ELF::SymbolReferenceMap&) {
    const auto name = getSymName().str();
    auto section = writer.addDmaSymbolSection(name);

    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));

    auto& operations = getBody()->getOperations();
    for (auto& op : operations) {
        auto dmaSymbol = section->addDmaSymbolEntry();
        auto dmaSymOp = llvm::dyn_cast<ELF::DmaSymbolOp>(op);

        VPUX_THROW_UNLESS(dmaSymOp, "DMA symbol table section op is expected to contain only DmaSymbolOps. Got {0}",
                          op);
        dmaSymOp.serialize(dmaSymbol);
        dmaSymbolMap[dmaSymOp.getOperation()] = dmaSymbol;
    }

    // since we only currently issue Symbols with STB_LOCAL binding, we just set the info to the number of symbolOps
    // in the block

    section->setInfo(static_cast<uint32_t>(operations.size()));

    sectionMap[getOperation()] = section;
}

void vpux::ELF::DmaSymbolSectionOp::preserialize(elf::Writer&, vpux::ELF::SectionMapType&,
                                                 vpux::ELF::SymbolReferenceMap&) {
    // don't implement and go into elf::Writer internal state first
    // as there are symbol entries to be updated first
    // E#136375
}
