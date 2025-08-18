//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace VPUIP {

//
// LUTConverterBase
//

class LUTConverterBase : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    LUTConverterBase(mlir::MLIRContext* ctx, Logger log, mlir::func::FuncOp netFunc)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log), _netFunc(netFunc) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceClusterTask,
                                        mlir::PatternRewriter& rewriter) const final;

protected:
    virtual mlir::Value createLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                               mlir::PatternRewriter& rewriter) const = 0;
    mlir::Value createCopyDestination(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value LUTConst,
                                      mlir::PatternRewriter& rewriter) const;
    VPU::DistributionInfoAttr createDistributionInfoAttr(VPUIP::DistributedBufferType inputDistribType,
                                                         VPUIP::NCEClusterTaskOp nceClusterTask) const;
    VPUIP::DistributedBufferType createDistributedBufferType(VPU::DistributionInfoAttr distributedInfo,
                                                             VPUIP::NCEClusterTaskOp nceClusterTask,
                                                             mlir::Value LUTConst) const;
    virtual void replaceWithConstInput(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value lutNceInput,
                                       mlir::PatternRewriter& rewriter) const = 0;

    Logger _log;
    mutable mlir::func::FuncOp _netFunc;
};

}  // namespace VPUIP
}  // namespace vpux
