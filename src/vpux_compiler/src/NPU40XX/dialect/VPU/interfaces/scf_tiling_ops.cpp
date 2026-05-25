//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_viewlike_interfaces.hpp"

void vpux::VPU::arch40xx::registerSCFTilingOpsInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEEltwiseOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<vpux::VPU::SCFAvgPoolOpModel>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<vpux::VPU::SCFMaxPoolOpModel>(*ctx);
        VPU::NCEConvolutionOp::attachInterface<vpux::VPU::SCFConvOpModel>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<vpux::VPU::SCFCompressConvOpModel>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<vpux::VPU::SCFTilingDepthConvModelOp>(*ctx);
        VPU::NCEPermuteOp::attachInterface<vpux::VPU::SCFTilingPermuteModelOp>(*ctx);
        VPU::NCEReduceOp::attachInterface<vpux::VPU::SCFNCEReduceModelOp>(*ctx);

        VPU::DepthToSpaceOp::attachInterface<vpux::VPU::SCFDepthToSpaceModelOp>(*ctx);
        VPU::InterpolateOp::attachInterface<vpux::VPU::SCFInterpolateModelOp>(*ctx);
        VPU::ConvertOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ConvertOp>>(*ctx);
        VPU::YuvToRgbOp::attachInterface<vpux::VPU::SCFYuvToRgbModelOp>(*ctx);

        VPU::LayoutCastOp::attachInterface<vpux::VPU::SCFGenericViewLikeTilingModelOp<VPU::LayoutCastOp>>(*ctx);
        VPU::PermuteCastOp::attachInterface<vpux::VPU::SCFPermuteCastTilingModelOp>(*ctx);
        VPU::SliceOp::attachInterface<vpux::VPU::SCFSliceTilingModelOp>(*ctx);
        VPU::QuantizeCastOp::attachInterface<vpux::VPU::SCFGenericViewLikeTilingModelOp<VPU::QuantizeCastOp>>(*ctx);

        VPU::ReduceLogicalOrOp::attachInterface<vpux::VPU::SCFReduceLogicalOrModelOp>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<vpux::VPU::SCFReduceLogicalAndModelOp>(*ctx);
        VPU::ReduceMeanOp::attachInterface<vpux::VPU::SCFReduceMeanModelOp>(*ctx);
        VPU::ReduceSumOp::attachInterface<vpux::VPU::SCFReduceSumModelOp>(*ctx);
        VPU::ReduceL2Op::attachInterface<vpux::VPU::SCFReduceL2ModelOp>(*ctx);
        VPU::ReduceL1Op::attachInterface<vpux::VPU::SCFReduceL1ModelOp>(*ctx);
        VPU::ReduceSquareOp::attachInterface<vpux::VPU::SCFReduceSquareModelOp>(*ctx);
        VPU::ReduceMinOp::attachInterface<vpux::VPU::SCFReduceMinModelOp>(*ctx);
        VPU::ReduceMaxOp::attachInterface<vpux::VPU::SCFReduceMaxModelOp>(*ctx);
        VPU::ReduceProdOp::attachInterface<vpux::VPU::SCFReduceProdModelOp>(*ctx);
        VPU::LSTMGatesOp::attachInterface<vpux::VPU::SCFLSTMGatesModelOp>(*ctx);
        VPU::TopKOp::attachInterface<vpux::VPU::SCFTopKModelOp>(*ctx);
    });
}
