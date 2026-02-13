//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFFUSELASTVIEWLIKEOP
#define GEN_PASS_DEF_SCFFUSELASTVIEWLIKEOP
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// SCFFuseLastViewLikeOpPass
//

class SCFFuseLastViewLikeOpPass final : public VPU::impl::SCFFuseLastViewLikeOpBase<SCFFuseLastViewLikeOpPass> {
public:
    explicit SCFFuseLastViewLikeOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

using OffsetSizePair = std::pair<SmallVector<int64_t>, SmallVector<int64_t>>;

OffsetSizePair updateOffsetAndSize(VPU::PermuteCastOp permuteCastOp, ArrayRef<int64_t> origOffsets,
                                   ArrayRef<int64_t> origSizes) {
    auto permuteCastInType = mlir::cast<NDTypeInterface>(permuteCastOp.getInput().getType());
    auto permuteCastOutType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());

    auto permuteArr = [&](ArrayRef<int64_t> arr) {
        const auto arrInMemOrder = permuteCastInType.getDimsOrder().toMemoryOrder(Shape(arr));
        const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, permuteCastOp.getMemPerm());
        return permuteCastOutType.getDimsOrder().toLogicalOrder(arrPermutedInMemOrder).raw();
    };

    const auto newOffsets = permuteArr(origOffsets);
    const auto newSizes = permuteArr(origSizes);

    return {newOffsets, newSizes};
}

void moveOpInsideForLoop(VPU::ViewLikeOpInterface viewLikeOp, mlir::tensor::InsertSliceOp insertSliceOp) {
    mlir::OpBuilder bodyBuilder(insertSliceOp.getOperation());

    mlir::IRMapping mapper;
    mapper.map(viewLikeOp->getOperand(0), insertSliceOp.getSource());
    auto newViewLikeOp = bodyBuilder.clone(*viewLikeOp.getOperation(), mapper);
    vpux::inferReturnTypes(newViewLikeOp, vpux::InferShapedTypeMode::ALL);

    auto viewLikeOutType = mlir::cast<NDTypeInterface>(viewLikeOp->getResult(0).getType());
    insertSliceOp.setOperand(0, newViewLikeOp->getResult(0));
    insertSliceOp.getDestMutable().get().setType(viewLikeOutType);
    insertSliceOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(viewLikeOutType));

    if (auto permuteCastOp = mlir::dyn_cast<VPU::PermuteCastOp>(viewLikeOp.getOperation())) {
        auto [newOffsets, newSizes] =
                updateOffsetAndSize(permuteCastOp, insertSliceOp.getStaticOffsets(), insertSliceOp.getStaticSizes());
        insertSliceOp.setStaticOffsets(newOffsets);
        insertSliceOp.setStaticSizes(newSizes);
    } else if (auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(viewLikeOp.getOperation())) {
        insertSliceOp.setStaticSizes(viewLikeOutType.getShape().raw());
    } else {
        VPUX_THROW("Unsupported operator is being moved inside for loop");
    }
}

bool traverseNestedForLoops(mlir::scf::ForOp forOp, VPU::ViewLikeOpInterface viewLikeOp, Logger log) {
    auto viewOpCastOutType = mlir::cast<NDTypeInterface>(viewLikeOp->getResult(0).getType());
    auto body = forOp.getBody();

    auto yieldOps = body->getOps<mlir::scf::YieldOp>();
    assert(!yieldOps.empty() && "no scf.yield op in for loop");
    assert(std::distance(yieldOps.begin(), yieldOps.end()) == 1 && "only one scf.yield op allowed per loop");

    auto yieldOp = *yieldOps.begin();

    mlir::Operation* producer = yieldOp->getOperand(0).getDefiningOp();

    bool movedViewOp = false;
    if (auto insertSliceOp = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(producer)) {
        log.trace("Found InsertSliceOp at {0}", insertSliceOp->getLoc());
        moveOpInsideForLoop(viewLikeOp, insertSliceOp);
        movedViewOp = true;
    }

    if (auto innerForOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(producer)) {
        log.trace("Found inner scf for loop at {0}", innerForOp->getLoc());
        movedViewOp = traverseNestedForLoops(innerForOp, viewLikeOp, log);
    }

    if (movedViewOp) {
        log.trace("Permute got moved, adapt return type for scf.for at {0}", forOp->getLoc());
        auto iterArg = forOp.getInitArgsMutable().begin()->get();
        iterArg.setType(viewOpCastOutType);
        forOp.getResult(0).setType(viewOpCastOutType);
    }
    return movedViewOp;
}

void SCFFuseLastViewLikeOpPass::safeRunOnFunc() {
    auto func = getOperation();

    SmallVector<VPU::ViewLikeOpInterface> viewLikeOpsToErase;

    func->walk<mlir::WalkOrder::PreOrder>([&](VPU::ViewLikeOpInterface viewLikeOp) {
        mlir::Operation* op = viewLikeOp.getOperation();
        if (!mlir::isa<VPU::PermuteCastOp>(op) && !mlir::isa<VPU::SliceOp>(op)) {
            _log.trace("Skipping non-PermuteCast and non-Slice view-like op at '{0}'", viewLikeOp->getLoc());
            return;
        }
        const auto& nestedLog = _log.nest();
        _log.trace("Got '{0}' at '{1}'", viewLikeOp->getName(), viewLikeOp->getLoc());

        mlir::Operation* user = *viewLikeOp->getResult(0).getUsers().begin();
        if (!mlir::isa<mlir::func::ReturnOp>(user)) {
            nestedLog.trace("Not used by a return operation");
            return;
        }

        mlir::Operation* producer = op->getOperand(0).getDefiningOp();
        if (!mlir::isa<mlir::scf::ForOp>(producer)) {
            nestedLog.trace("Not produced by a scf::ForOp");
            return;
        }

        auto forOp = mlir::cast<mlir::scf::ForOp>(producer);
        if (!forOp->getResult(0).hasOneUse()) {
            nestedLog.trace("scf::ForOp output has more than one use.");
            return;
        }

        const auto movedViewOp = traverseNestedForLoops(forOp, viewLikeOp, nestedLog);

        if (!movedViewOp) {
            nestedLog.trace("Could not move view op into parent scf.for.");
            return;
        }

        viewLikeOp->replaceAllUsesWith(forOp.getResults());

        if (viewLikeOp->getUses().empty()) {
            viewLikeOpsToErase.push_back(viewLikeOp);
        }
        nestedLog.trace("Successfully fused last view op into scf::ForOp.");
    });

    for (auto viewLikeOp : llvm::make_early_inc_range(viewLikeOpsToErase)) {
        viewLikeOp->erase();
    }
}

}  // namespace

//
// createSCFFuseLastViewLikeOpPass
//

std::unique_ptr<mlir::Pass> VPU::createSCFFuseLastViewLikeOpPass(Logger log) {
    return std::make_unique<SCFFuseLastViewLikeOpPass>(log);
}
