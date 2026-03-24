//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/reloc_manager.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDRELOCATIONSFORDYNAMICSTRIDESDMAS
#define GEN_PASS_DEF_ADDRELOCATIONSFORDYNAMICSTRIDESDMAS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class AddRelocationsForDynamicStridesDmas :
        public ELF::impl::AddRelocationsForDynamicStridesDmasBase<AddRelocationsForDynamicStridesDmas> {
public:
    explicit AddRelocationsForDynamicStridesDmas(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    enum IoType { Input, Output };
    ELF::DmaSymbolOp createDynamicStridesDmaSymbol(ELF::RelocatableOpWithDynamicStridesInterface relocatableOp,
                                                   ELF::DmaSymbolSectionOp targetSection,
                                                   VPUASM::DeclareBufferOp declareOp, IoType ioType);
};

ELF::DmaSymbolOp AddRelocationsForDynamicStridesDmas::createDynamicStridesDmaSymbol(
        ELF::RelocatableOpWithDynamicStridesInterface relocatableOp, ELF::DmaSymbolSectionOp targetSection,
        VPUASM::DeclareBufferOp declareOp, IoType ioType) {
    llvm::SmallVector<int64_t, elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> tensorStridesInElements;
    llvm::SmallVector<int64_t, elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> tensorShapes;
    llvm::SmallVector<int64_t, elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> dmaStridesInElements;
    llvm::SmallVector<int64_t, elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS> dmaShapes;
    int64_t dmaSize = 0;
    auto elemSizeByte = Byte(1);

    auto memrefType = mlir::cast<NDTypeInterface>(declareOp.getBufferType().getMemref());
    elemSizeByte = Byte(memrefType.getElemTypeSize());

    // DMA symbol defines one more shape and stride then the DMA descriptor since DMA descriptor
    // flattens one dimension into dst/src_width field. That's why below code updates final dimension
    // to stride 0 and shape 1.
    for (auto strideIdx : vpux::irange(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS - 1)) {
        auto strideValueInElements =
                relocatableOp.getStrideValue(strideIdx, ioType == IoType::Input).count() / elemSizeByte.count();
        tensorStridesInElements.push_back(strideValueInElements);
        dmaStridesInElements.push_back(strideValueInElements);
    }
    tensorStridesInElements.push_back(0);
    dmaStridesInElements.push_back(0);

    for (auto shapeIdx : vpux::irange(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS - 1)) {
        auto shapeValue = relocatableOp.getShapeValue(shapeIdx, ioType == IoType::Input);
        dmaShapes.push_back(shapeValue);
        tensorShapes.push_back(shapeValue);
    }
    dmaShapes.push_back(1);
    tensorShapes.push_back(1);

    auto ioIndex = declareOp.getBufferType().getLocation().getSectionIndex();

    SmallVector<int64_t> tileOffsets{0, 0, 0, 0, 0, 0};
    if (auto offsetsAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(declareOp->getAttr(vpux::viewOffsetsAttrName))) {
        auto dimsOrder = mlir::cast<NDTypeInterface>(declareOp.getBufferType().getMemref()).getDimsOrder();
        auto permutedStrides = dimsOrder.toMemoryOrder(Shape(parseIntArrayAttr<int64_t>(offsetsAttr)));
        for (size_t idx = 0; idx < tileOffsets.size() && idx < permutedStrides.size(); idx++) {
            tileOffsets[idx] = permutedStrides[MemDim(permutedStrides.size() - idx - 1)];
        }
    }

    dmaSize = relocatableOp.getWidthValue(ioType == IoType::Input);

    auto dmaSymbolSectionBuilder = mlir::OpBuilder::atBlockEnd(targetSection.getBlock());
    auto ctx = declareOp->getContext();
    auto symbolOp = mlir::cast<mlir::SymbolOpInterface>(relocatableOp.getOperation());
    return dmaSymbolSectionBuilder.create<ELF::DmaSymbolOp>(
            relocatableOp->getLoc(), symbolOp.getNameAttr(), ioIndex, getIntArrayAttr(ctx, tensorShapes),
            getIntArrayAttr(ctx, tensorStridesInElements), getIntArrayAttr(ctx, tileOffsets),
            getIntArrayAttr(ctx, dmaShapes), getIntArrayAttr(ctx, dmaStridesInElements), dmaSize);
}

void AddRelocationsForDynamicStridesDmas::safeRunOnFunc() {
    auto funcOp = getOperation();

    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto elfBuilder = mlir::OpBuilder::atBlockBegin(elfMain.getBody());
    auto dmaInputSymbolSection = elfBuilder.create<ELF::DmaSymbolSectionOp>(
            elfBuilder.getUnknownLoc(), "dmaInputSymbolsSection",
            ELF::SectionFlagsAttr::VPU_SHF_JIT | ELF::SectionFlagsAttr::VPU_SHF_USERINPUT);
    auto dmaOutputSymbolSection = elfBuilder.create<ELF::DmaSymbolSectionOp>(
            elfBuilder.getUnknownLoc(), "dmaOutputSymbolsSection",
            ELF::SectionFlagsAttr::VPU_SHF_JIT | ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT);

    ELF::RelocManager relocManager(elfMain);
    for (auto targetSection : elfMain.getOps<ELF::ElfSectionInterface>()) {
        // TODO:  E#195185 consider to add interface dedicated for sections that are intended to hold program data, aka
        // should be relocateable?
        if (!mlir::isa<ELF::DataSectionOp, ELF::LogicalSectionOp>(targetSection)) {
            continue;
        }

        auto block = targetSection.getBlock();
        for (auto relocatableOp : block->getOps<ELF::RelocatableOpWithDynamicStridesInterface>()) {
            std::vector<ELF::RelocationInfo> relocs;
            if (relocatableOp->getAttr(vpux::stridedInputAttrName)) {
                auto declareOp =
                        mlir::cast<VPUASM::DeclareBufferOp>(elfMain.lookupSymbol(relocatableOp.getInputSymbol()));
                auto inputSymbol =
                        createDynamicStridesDmaSymbol(relocatableOp, dmaInputSymbolSection, declareOp, IoType::Input);
                auto newSymRef = mlir::SymbolRefAttr::get(&getContext(), inputSymbol.getSymName());
                auto inputSymbolName =
                        mlir::SymbolRefAttr::get(&getContext(), dmaInputSymbolSection.getSymName(), {newSymRef});
                relocs.emplace_back(inputSymbolName, targetSection, 0, ELF::RelocationType::R_VPU_DMA_DESCRIPTOR_INPUT,
                                    0, "DMA descriptor relocation for strided input");
            }

            if (relocatableOp->getAttr(vpux::stridedOutputAttrName)) {
                auto declareOp =
                        mlir::cast<VPUASM::DeclareBufferOp>(elfMain.lookupSymbol(relocatableOp.getOutputSymbol()));
                auto outputSymbol =
                        createDynamicStridesDmaSymbol(relocatableOp, dmaOutputSymbolSection, declareOp, IoType::Output);
                auto newSymRef = mlir::SymbolRefAttr::get(&getContext(), outputSymbol.getSymName());
                auto outputSymbolName =
                        mlir::SymbolRefAttr::get(&getContext(), dmaOutputSymbolSection.getSymName(), {newSymRef});
                relocs.emplace_back(outputSymbolName, targetSection, 0,
                                    ELF::RelocationType::R_VPU_DMA_DESCRIPTOR_OUTPUT, 0,
                                    "DMA descriptor relocation for strided output");
            }

            relocManager.createDynamicStridesDMARelocations(relocatableOp, relocs);
        }
    }

    if (dmaInputSymbolSection.getRegion().getOps<ELF::DmaSymbolOp>().empty()) {
        dmaInputSymbolSection.erase();
    }
    if (dmaOutputSymbolSection.getRegion().getOps<ELF::DmaSymbolOp>().empty()) {
        dmaOutputSymbolSection.erase();
    }
}

}  // namespace

//
// createRemoveEmptyELFSectionsPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddRelocationsForDynamicStridesDMAsPass(Logger log) {
    return std::make_unique<AddRelocationsForDynamicStridesDmas>(log);
}
