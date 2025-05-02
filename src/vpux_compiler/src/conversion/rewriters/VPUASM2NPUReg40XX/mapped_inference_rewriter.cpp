//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/mapped_inference_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult MappedInferenceRewriter::matchAndRewrite(VPUASM::MappedInferenceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(origOp.getDmaCount());

    mlir::SmallVector<int64_t> dmaCountDDR;
    mlir::SmallVector<int64_t> dmaCountCMX;
    dmaCountDDR.reserve(dmaCount.size());
    dmaCountCMX.reserve(dmaCount.size());

    for (size_t dmaTileIndex = 0; dmaTileIndex < dmaCount.size(); dmaTileIndex++) {
        VPUX_THROW_UNLESS(dmaCount[dmaTileIndex].size() == 2, "Unsupported number of DMA types - '{0}'",
                          dmaCount[dmaTileIndex].size());

        dmaCountDDR.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]);
        dmaCountCMX.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]);
    }

    const auto dmaCountDDRAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountDDR));
    const auto dmaCountCMXAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountCMX));

    rewriter.create<NPUReg40XX::MappedInferenceOp>(origOp->getLoc(),                           //
                                                   origOp.getSymNameAttr(),                    //
                                                   origOp.getDmaCountAttr(),                   //
                                                   dmaCountDDRAttr,                            //
                                                   dmaCountCMXAttr,                            //
                                                   origOp.getInvariantCountAttr(),             //
                                                   origOp.getVariantCountAttr(),               //
                                                   origOp.getActKernelRangesCountAttr(),       //
                                                   origOp.getActKernelInvocationsCountAttr(),  //
                                                   origOp.getMediaCountAttr(),                 //
                                                   origOp.getBarrierCountAttr(),               //
                                                   origOp.getMappedInferenceVersionAttr(),     //
                                                   origOp.getActShaveRtAttr(),                 //
                                                   origOp.getActShaveStacksAttr(),             //
                                                   origOp.getDmaHwpBaseAttr(),                 //
                                                   origOp.getHwpWorkpointCfgAttr(),            //
                                                   origOp.getManagedMappedInferenceAttr(),
                                                   origOp.getDmaTasksAttr(),              //
                                                   origOp.getInvariantTasksAttr(),        //
                                                   origOp.getVariantTasksAttr(),          //
                                                   origOp.getActKernelRangesAttr(),       //
                                                   origOp.getActKernelInvocationsAttr(),  //
                                                   origOp.getMediaTasksAttr(),            //
                                                   origOp.getBarrierTasksAttr());         //
    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
