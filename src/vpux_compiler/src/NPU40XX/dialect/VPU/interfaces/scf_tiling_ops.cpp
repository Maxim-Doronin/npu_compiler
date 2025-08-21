//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_viewlike_interfaces.hpp"

void vpux::VPU::arch40xx::registerSCFTilingOpsInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEEltwiseOp::attachInterface<vpux::VPU::SCFTilingEltwiseModelOp>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<vpux::VPU::SCFTilingPoolingModelOp<VPU::NCEAveragePoolOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<vpux::VPU::SCFTilingPoolingModelOp<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEConvolutionOp::attachInterface<vpux::VPU::SCFConvOpModel>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<vpux::VPU::SCFTilingDepthConvModelOp>(*ctx);

        VPU::LayoutCastOp::attachInterface<vpux::VPU::SCFLayoutCastTilingModelOp>(*ctx);
    });
}
