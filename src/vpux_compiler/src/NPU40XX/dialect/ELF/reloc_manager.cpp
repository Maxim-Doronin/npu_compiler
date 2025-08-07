//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/reloc_manager.hpp"
#include <cstdint>
#include "vpux/compiler/NPU40XX/dialect/ELF/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/relocation_functions.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;
using namespace ELF;

namespace {
// Relocation specific methods and values
const std::map<ELF::RelocationType, RelocFunc> relocationMap = {
        {ELF::RelocationType::R_VPU_64_BIT_OR_B21_B26_UNSET, VPU_64_BIT_OR_B21_B26_UNSET_Relocation},
        {ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET, VPU_32_BIT_OR_B21_B26_UNSET_Relocation},
        {ELF::RelocationType::R_VPU_LO_21_RSHIFT_4, VPU_LO_21_BIT_RSHIFT_4_Relocation},
        {ELF::RelocationType::R_VPU_LO_21, VPU_LO_21_BIT_Relocation},
        {ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_16, VPU_16_BIT_LSB_21_RSHIFT_5_LSHIFT_16_Relocation},
        {ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5, VPU_16_BIT_LSB_21_RSHIFT_5_Relocation},
        {ELF::RelocationType::R_VPU_16_LSB_21_RSHIFT_5_LSHIFT_CUSTOM,
         VPU_16_BIT_LSB_21_RSHIFT_5_LSHIFT_CUSTOM_Relocation},
        {ELF::RelocationType::R_VPU_LO_21_SUM, VPU_LO_21_BIT_SUM_Relocation},
        {ELF::RelocationType::R_VPU_64, VPU_64_BIT_Relocation},
        {ELF::RelocationType::R_VPU_32, VPU_32_BIT_Relocation},
        {ELF::RelocationType::R_VPU_LO_21_MULTICAST_BASE, VPU_LO_21_BIT_MULTICAST_BASE_Relocation},
        {ELF::RelocationType::R_VPU_CMX_LOCAL_RSHIFT_5, VPU_CMX_LOCAL_RSHIFT_5_Relocation},
        {ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET_LOW_16, VPU_32_BIT_OR_B21_B26_UNSET_LOW_16_Relocation},
        {ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET_HIGH_16, VPU_32_BIT_OR_B21_B26_UNSET_HIGH_16_Relocation},
        {ELF::RelocationType::R_VPU_32_OR_LO_19_LSB_21_RSHIFT_2, VPU_32_OR_LO_19_LSB_21_RSHIFT_2_Relocation}};

}  // namespace

ELF::CreateRelocationSectionOp ELF::RelocManager::getRelocationSection(ELF::ElfSectionInterface targetSection,
                                                                       ELF::CreateSymbolTableSectionOp symbolTable) {
    auto key = std::make_pair(targetSection.getOperation(), symbolTable);

    auto relocSectionIt = relocMap_.find(key);

    if (relocSectionIt != relocMap_.end()) {
        return relocSectionIt->getSecond();
    }

    auto targetSectionSymbolIface = mlir::cast<mlir::SymbolOpInterface>(targetSection.getOperation());
    auto symtabSymbolIface = mlir::cast<mlir::SymbolOpInterface>(symbolTable.getOperation());

    mlir::StringAttr nameAttr =
            mlir::StringAttr::get(builder_.getContext(), llvm::Twine("rela.") + targetSectionSymbolIface.getName() +
                                                                 "." + symtabSymbolIface.getName());

    auto targetSectionRef = mlir::FlatSymbolRefAttr::get(targetSectionSymbolIface.getNameAttr());
    auto symTabRef = mlir::FlatSymbolRefAttr::get(symtabSymbolIface.getNameAttr());

    auto symTabFlags = symbolTable.getSecFlags();

    auto relaSectionFlags = symTabFlags;
    auto isJITRelaSection = (static_cast<uint32_t>(relaSectionFlags & ELF::SectionFlagsAttr::VPU_SHF_USERINPUT) ||
                             static_cast<uint32_t>(relaSectionFlags & ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT) ||
                             static_cast<uint32_t>(relaSectionFlags & ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT));

    VPUX_THROW_WHEN(isJITRelaSection && !ELF::bitEnumContainsAll(relaSectionFlags, ELF::SectionFlagsAttr::VPU_SHF_JIT),
                    "Reloc Section for JIT symbols must have VPU_SHF_JIT Flag");

    auto flags = ELF::SectionFlagsAttrAttr::get(builder_.getContext(), relaSectionFlags);
    auto newRelocSection = builder_.create<ELF::CreateRelocationSectionOp>(symbolTable.getLoc(), nameAttr,
                                                                           targetSectionRef, symTabRef, flags);

    relocMap_[key] = newRelocSection;

    return newRelocSection;
}

ELF::SymbolOp ELF::RelocManager::getSymbolOfBinOpOrEncapsulatingSection(mlir::Operation* binOp) {
    auto sectionOp = mlir::isa<ELF::ElfSectionInterface>(binOp) ? mlir::cast<ELF::ElfSectionInterface>(binOp)
                                                                : binOp->getParentOfType<ELF::ElfSectionInterface>();

    auto symbolMapIt = symbolMap_.find(sectionOp.getOperation());

    if (symbolMapIt != symbolMap_.end()) {
        return symbolMapIt->getSecond();
    }

    VPUX_THROW("No ELF Symbol found for the provided operation");
}

void ELF::RelocManager::createRelocations(mlir::Operation* op, ELF::RelocationInfo& relocInfo) {
    auto sourceOp = symRefMap_.lookupSymbol(relocInfo.source);

    ELF::SymbolOp sourceSym = getSymbolOfBinOpOrEncapsulatingSection(sourceOp);

    // we can only modify ops for which the binary format is known
    auto targetOp = mlir::dyn_cast<VPURegMapped::NPURegDescriptorOpInterface>(op);
    if (sourceSym.getValue().has_value() && targetOp) {
        auto relocFunc = relocationMap.find(relocInfo.relocType);
        VPUX_THROW_UNLESS(relocFunc != relocationMap.end(), "Relocation type {0} not known!",
                          stringifyRelocationType(relocInfo.relocType));

        // reloc offset at this point is expressed only inside of the operation specific descriptor
        auto descriptor = targetOp.getDescriptorStorage();
        VPUX_THROW_UNLESS(relocInfo.offset < descriptor.size(), "Offset is outside of descriptor!");

        relocFunc->second(reinterpret_cast<void*>(descriptor.begin() + relocInfo.offset), sourceSym.getValue().value(),
                          relocInfo.addend);

    } else {
        ELF::CreateSymbolTableSectionOp symTab =
                mlir::dyn_cast<ELF::CreateSymbolTableSectionOp>(sourceSym->getParentOp());
        auto symForReloc = ELF::composeSectionObjectSymRef(symTab, sourceSym.getOperation());
        ELF::CreateRelocationSectionOp relocSection = getRelocationSection(relocInfo.targetSection, symTab);
        auto relocBuilder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        auto offset = relocInfo.offset;

        // here we set the actual offset from the beginning of the final ELF file
        if (relocInfo.isOffsetRelative) {
            auto baseBinarySizeOp = mlir::cast<ELF::BinarySizeOpInterface>(op);
            offset += baseBinarySizeOp.getMemoryOffset();
        }

        relocBuilder.create<ELF::RelocOp>(relocSection.getLoc(), offset, symForReloc, relocInfo.relocType,
                                          relocInfo.addend, relocInfo.description);
    }
}

void ELF::RelocManager::createRelocations(mlir::Operation* op, std::vector<ELF::RelocationInfo>& relocInfo) {
    for (auto& reloc : relocInfo) {
        createRelocations(op, reloc);
    }
}

void ELF::RelocManager::createRelocations(ELF::RelocatableOpInterface relocatableOp) {
    auto relocsInfo = relocatableOp.getRelocationInfo(symRefMap_);
    createRelocations(relocatableOp.getOperation(), relocsInfo);
}

void ELF::RelocManager::constructSymbolMap(ELF::MainOp elfMain) {
    auto symbolTables = elfMain.getOps<ELF::CreateSymbolTableSectionOp>();

    for (auto symbolTable : symbolTables) {
        auto elfSymbols = symbolTable.getOps<ELF::SymbolOp>();
        for (auto elfSymbol : elfSymbols) {
            auto reference = symRefMap_.lookupSymbol(elfSymbol.getReference());
            symbolMap_[reference] = elfSymbol;
        }
    }
}
