//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

//
// StackFramesOp
//

void vpux::VPUASM::StackFramesOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    const auto& addresses = getProperties().addresses;
    binDataSection.appendData(reinterpret_cast<const uint8_t*>(addresses.data()),
                              getBinarySize(config::ArchKind::UNKNOWN));
}

size_t vpux::VPUASM::StackFramesOp::getBinarySize(config::ArchKind) {
    return getAddresses().size() * sizeof(uint32_t);
}

size_t vpux::VPUASM::StackFramesOp::getAlignmentRequirements(config::ArchKind) {
    return sizeof(uint32_t);
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::StackFramesOp::getPredefinedMemoryAccessors() {
    return (ELF::SectionFlagsAttr::SHF_EXECINSTR);
}

std::optional<ELF::SectionSignature> vpux::VPUASM::StackFramesOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "stackFrames"),
                                 ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::StackFramesOp::hasMemoryFootprint() {
    return true;
}

void vpux::VPUASM::StackFramesOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                        SmallVector<uint32_t>&& addresses) {
    auto& props = state.getOrAddProperties<Properties>();
    props.sym_name = symName;
    props.addresses = std::move(addresses);
}
