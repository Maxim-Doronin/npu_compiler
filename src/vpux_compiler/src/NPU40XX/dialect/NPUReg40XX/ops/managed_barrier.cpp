//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

//
// ManagedBarrierOp
//

void NPUReg40XX::ManagedBarrierOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto barrierDescriptor = getProperties().getDescriptor();

    VPUX_THROW_UNLESS(sizeof(nn_public::VpuTaskBarrierMap) == barrierDescriptor.size(),
                      "HW VpuTaskBarrierMap size {0} != regMapped representation size {1}.",
                      sizeof(nn_public::VpuTaskBarrierMap), barrierDescriptor.size());

    auto serializedBarrierDescriptor = barrierDescriptor.getStorage();
    binDataSection.appendData(serializedBarrierDescriptor.data(), getBinarySize(config::ArchKind::NPU40XX));
}

size_t NPUReg40XX::ManagedBarrierOp::getBinarySize(config::ArchKind) {
    return sizeof(nn_public::VpuTaskBarrierMap);
}

size_t vpux::NPUReg40XX::ManagedBarrierOp::getAlignmentRequirements(config::ArchKind) {
    return alignof(nn_public::VpuTaskBarrierMap);
}

void vpux::NPUReg40XX::ManagedBarrierOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                               vpux::NPUReg40XX::Descriptors::VpuTaskBarrierMap&& descriptor) {
    auto& props = state.getOrAddProperties<Properties>();

    props.sym_name = symName;
    props.descriptor = std::move(descriptor);
}
