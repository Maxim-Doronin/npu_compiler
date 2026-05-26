//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// ShaveStackFrameBuffOp
//

size_t vpux::VPUASM::ShaveStackFrameBuffOp::getBinarySizeCached(ELF::SymbolReferenceMap&, config::ArchKind) {
    return getStackSize();
}

size_t vpux::VPUASM::ShaveStackFrameBuffOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_SHAVE_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::ShaveStackFrameBuffOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::ShaveStackFrameBuffOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "stackBuffer"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC | ELF::SectionFlagsAttr::SHF_WRITE,
                                 ELF::SectionTypeAttr::SHT_NOBITS);
}

bool vpux::VPUASM::ShaveStackFrameBuffOp::hasMemoryFootprint() {
    return false;
}

vpux::VPURT::BufferSection VPUASM::ShaveStackFrameBuffOp::getMemorySection() {
    return VPURT::BufferSection::DDR;
}
