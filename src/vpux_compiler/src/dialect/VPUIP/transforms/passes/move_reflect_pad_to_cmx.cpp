//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVEREFLECTPADTOCMX
#define GEN_PASS_DEF_MOVEREFLECTPADTOCMX
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

constexpr double MIN_PERCENT_COPIED_MEM = 25;
namespace {

bool fitsIntoCMX(mlir::Value rootInput, VPUIP::ConcatViewOp concatView) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(rootInput.getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(concatView.getOutputBuff().getType());
    auto requiredCMX = inputType.getTotalAllocSize() + outputType.getTotalAllocSize();
    return requiredCMX < VPU::getTotalCMXSize(concatView);
}

void propagateReturnType(mlir::Operation* op, mlir::OpBuilder& builder) {
    for (auto user : op->getUsers()) {
        if (mlir::isa<VPUIP::ConcatViewOp>(user)) {
            return;
        }
        if (mlir::isa<VPUIP::PermuteCastOp>(user)) {
            builder.setInsertionPoint(user);
            auto permuteCastUser = mlir::cast<VPUIP::PermuteCastOp>(user);
            auto resultType = mlir::cast<NDTypeInterface>(permuteCastUser.getResult().getType());
            auto newResultType = resultType.changeMemSpace(
                    IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
            auto newPermuteCast = builder.create<VPUIP::PermuteCastOp>(
                    permuteCastUser.getLoc(), newResultType, permuteCastUser.getSource(),
                    permuteCastUser.getDstOrderAttr(), permuteCastUser.getMemPermAttr());
            permuteCastUser->getResult(0).replaceAllUsesWith(newPermuteCast);
            if (permuteCastUser.use_empty()) {
                permuteCastUser.erase();
            }
            propagateReturnType(newPermuteCast, builder);
        } else {
            inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);
            propagateReturnType(user, builder);
        }
    }
}

bool checkForSameRoot(llvm::SmallVector<mlir::Operation*>& ops) {
    if (ops.empty())
        return true;

    auto firstSubViewOp = *ops.begin();
    mlir::Value firstRoot = mlir::cast<VPUIP::SubViewOp>(firstSubViewOp).getSource();

    return llvm::all_of(ops, [&](mlir::Operation* op) {
        return mlir::cast<VPUIP::SubViewOp>(op).getSource() == firstRoot;
    });
}

mlir::memref::AllocOp createAllocInCmx(mlir::OpBuilder& builder, mlir::Value value) {
    auto allocType = mlir::cast<NDTypeInterface>(value.getType());
    auto newOutputType = allocType.changeMemSpace(
            IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
    auto outputMemRefType = mlir::cast<mlir::MemRefType>(newOutputType);
    return builder.create<mlir::memref::AllocOp>(value.getLoc(), outputMemRefType);
}

/*
 Possible patterns:
 1)
                                        Copy (CMX->DDR)
                   /                          |                   \
        (      SubView           )            |               (    SubView            )
        (         |              )   (copying the Pad input)  (       |               )
        (    Copy (DDR->DDR)     )            |               (   Copy (DDR->DDR)     )
  N x   (     (out: alloc)       )            |               (    (out: alloc)       )    x M
        (         |              )            |               (       |               )
        (    Copy (DDR->DDR)     )     Copy (DDR->DDR)        (   Copy (DDR->DDR)     )
        (  (out: SubView of out  )   (out: SubView of out     ( (out: SubView of out  )
        (  buff of ConcatVIewOp) )    of ConcatVIewOp)        ( buff of ConcatVIewOp) )
                            \                 |                 /
                               \              |              /
                                        ConcatViewOp
                                       (alloc in DDR)

 2)
                                        Copy (CMX->DDR)
                     /                        |                  \
                   /                (copying the Pad input)        \
        (      SubView           )            |               (    SubView            )
        (         |              )      Copy (DDR->CMX)       (       |               )
        (    Copy (DDR->DDR)     )            |               (   Copy (DDR->DDR)     )
  N x   (     (out: alloc)       )       NCE MaxPool          (    (out: alloc)       )    x M
        (         |              )            |               (       |               )
        (     PermuteCast        )      Copy (CMX->DDR)       (    PermuteCast        )
        (         |              )            |               (       |               )
        (    Copy (DDR->DDR)     )     Copy (DDR->DDR)        (   Copy (DDR->DDR)     )
        (  (out: SubView of out  )   (out: SubView of out     ( (out: SubView of out  )
        (  buff of ConcatVIewOp) )    of ConcatVIewOp)        ( buff of ConcatVIewOp) )
                            \                 |                 /
                               \              |              /
                                        ConcatViewOp
                                       (alloc in DDR)
*/

bool matchReflectPadPatterns(const Logger& log, VPUIP::ConcatViewOp& origOp,
                             mlir::DenseSet<mlir::memref::AllocOp>& allocsToBeMovedToCMX,
                             SmallVector<mlir::Operation*>& inputOpsWithSameSource, VPUIP::CopyOp& inputCopy) {
    auto origOpOutBuff = origOp.getOutputBuff();
    auto origOpInputs = origOp.getInputs();
    auto origOpInput = *origOpInputs.begin();

    auto inputShape = to_small_vector(vpux::getShape(origOpInput));
    auto outputShape = to_small_vector(vpux::getShape(origOpOutBuff));

    auto dimToPad = -1;
    for (size_t idx = 0; idx < inputShape.size(); idx++) {
        if (inputShape[idx] != outputShape[idx]) {
            if (dimToPad != -1) {
                return false;
            }
            dimToPad = idx;
        }
    }
    if (dimToPad == -1) {
        return false;
    }

    auto nestedLog = log.nest();
    nestedLog.trace("Padding is done only on one dimension and with 1.");

    // check origOp output to be in DDR
    auto origOpOutBuffType = mlir::cast<vpux::NDTypeInterface>(origOpOutBuff.getType());
    if (origOpOutBuffType.getMemoryKind() != VPU::MemoryKind::DDR) {
        return false;
    }
    nestedLog.trace("ConcatView operation's output is in DDR.");

    SmallVector<mlir::Operation*> outputOpsWithSameSource;
    mlir::Value inputSource;
    // check origOp input patterns
    for (auto origInput : origOpInputs) {
        // inputs must be Copies DDR->DDR
        auto copy = mlir::dyn_cast_or_null<VPUIP::CopyOp>(origInput.getDefiningOp());
        if (copy == nullptr) {
            nestedLog.trace("Pattern match failed: ConcatView operation has non-copy inputs.");
            return false;
        }
        if (!isCopyFromDDR(copy) || !isCopyToDDR(copy)) {
            return false;
        }

        // output should be a SubView of a DDR alloc
        if (mlir::isa<mlir::BlockArgument>(copy.getOutputBuff())) {
            nestedLog.trace("Pattern match failed: ConcatView operation has block arg inputs.");
            return false;
        }
        auto copyOutput = copy.getOutputBuff().getDefiningOp();
        outputOpsWithSameSource.push_back(copyOutput);
        if (!mlir::isa<VPUIP::SubViewOp>(copyOutput)) {
            return false;
        }

        auto copyOutputSubview = mlir::cast<VPUIP::SubViewOp>(copyOutput);
        auto copyOutputSubviewSource = copyOutputSubview.getSource().getDefiningOp();
        if (!mlir::isa<mlir::memref::AllocOp>(copyOutputSubviewSource)) {
            return false;
        }
        auto copyOutputAllocOp = mlir::cast<mlir::memref::AllocOp>(copyOutputSubviewSource);
        allocsToBeMovedToCMX.insert(copyOutputAllocOp);

        // input is either Permute or Copy DDR -> DDR or Copy CMX -> DDR
        if (mlir::isa<mlir::BlockArgument>(copy.getInput())) {
            return false;
        }
        auto copyInput = copy.getInput().getDefiningOp();
        auto copyInputOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(copyInput);
        if (auto permuteCastOp = mlir::dyn_cast_or_null<VPUIP::PermuteCastOp>(copyInput)) {
            if (mlir::isa<mlir::BlockArgument>(permuteCastOp.getSource())) {
                return false;
            }
            copyInputOp = permuteCastOp.getSource().getDefiningOp<VPUIP::CopyOp>();
        }

        if (copyInputOp == nullptr) {
            return false;
        }

        // all copies should be CMX/DDR -> DDR
        if (!isCopyToDDR(copyInputOp)) {
            return false;
        }

        // cpying the input of the Pad (should happen only once)
        if (!isCopyFromDDR(copyInputOp)) {
            if (inputCopy) {
                nestedLog.trace("Pattern match failed: There must be only one Reflect Pad input copy.");
                return false;
            }
            nestedLog.trace("Found Reflect Pad input copy.");
            inputCopy = copy;

            // the ConcatView pattern should be moved to CMX only when the we are padding with more then 25% of the
            // input data, otherwise it is more efficient to keep the data in DDR (based on measurements performed on
            // full models containing Reflect Pad ops and on subgraph tests).
            auto padInputShape = to_small_vector(vpux::getShape(inputCopy.getOutputBuff()));
            if (((outputShape[dimToPad] - padInputShape[dimToPad]) / static_cast<double>(padInputShape[dimToPad])) *
                        100 <
                MIN_PERCENT_COPIED_MEM) {
                nestedLog.trace("Pattern match failed: At least 25% of memory should be copied.");
                return false;
            }
            continue;
        }

        // make sure the ConcatView comes from a Reflect Pad Op
        auto origInputShape = getShape(origInput).raw();
        if (origInputShape[dimToPad] != 1) {
            return false;
        }

        auto output = copyInputOp.getOutputBuff().getDefiningOp();
        if (!mlir::isa<mlir::memref::AllocOp>(output)) {
            return false;
        }
        auto outputAlloc = mlir::cast<mlir::memref::AllocOp>(output);
        allocsToBeMovedToCMX.insert(outputAlloc);

        if (mlir::isa<mlir::BlockArgument>(copyInputOp.getInput())) {
            return false;
        }
        auto input = copyInputOp.getInput().getDefiningOp();
        if (!mlir::isa<VPUIP::SubViewOp>(input)) {
            return false;
        }
        inputOpsWithSameSource.push_back(input);
        auto inputSubView = mlir::cast<VPUIP::SubViewOp>(input);
        inputSource = inputSubView.getSource();

        if (mlir::isa<mlir::BlockArgument>(inputSource)) {
            return false;
        }

        if (!fitsIntoCMX(inputSource, origOp)) {
            nestedLog.trace("Pattern match failed: ConcatView operation doesn't fit in CMX.");
            return false;
        }

        if (!mlir::isa<VPUIP::CopyOp>(inputSource.getDefiningOp())) {
            return false;
        }
    }

    if (!inputCopy) {
        nestedLog.trace("Pattern match failed: Did not find copy for Reflect Pad input.");
        return false;
    }

    return checkForSameRoot(inputOpsWithSameSource) && checkForSameRoot(outputOpsWithSameSource);
}

//
// MoveReflectPadToCMXPass
//

class MoveReflectPadToCMXPass final : public VPUIP::impl::MoveReflectPadToCMXBase<MoveReflectPadToCMXPass> {
public:
    explicit MoveReflectPadToCMXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveReflectPadToCMXPass::safeRunOnFunc() {
    auto func = getOperation();
    func.walk([&](VPUIP::ConcatViewOp origOp) {
        _log.trace("Found ConcatView operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

        mlir::OpBuilder builder(origOp);

        mlir::DenseSet<mlir::memref::AllocOp> allocsToBeMovedToCMX;
        SmallVector<mlir::Operation*> inputOpsWithSameSource;
        VPUIP::CopyOp inputCopy;
        auto matchedAnyPattern =
                matchReflectPadPatterns(_log, origOp, allocsToBeMovedToCMX, inputOpsWithSameSource, inputCopy);
        if (!matchedAnyPattern) {
            _log.nest().trace("Found wrong pattern.", origOp->getName(), origOp->getLoc());
            return;
        }
        _log.nest().trace("ConcatView operation '{0}' at '{1}' will be moved to CMX", origOp->getName(),
                          origOp->getLoc());

        // move allocs to CMX
        for (auto& alloc : allocsToBeMovedToCMX) {
            builder.setInsertionPoint(alloc);

            auto newCMXAlloc = createAllocInCmx(builder, alloc);
            alloc->getResult(0).replaceAllUsesWith(newCMXAlloc);

            if (alloc.use_empty()) {
                alloc.erase();
            }

            for (auto& use : newCMXAlloc->getUses()) {
                inferReturnTypes(use.getOwner(), vpux::InferShapedTypeMode::ALL);
            }
        }

        // add CopyOps DDR -> CMX and infer return types
        auto handleCopy = [&](mlir::Operation* sourceOp) {
            builder.setInsertionPointAfter(sourceOp);
            auto opResult = sourceOp->getResult(0);
            auto newCmxInput = createAllocInCmx(builder, opResult);
            auto copyDDRToCMX = builder.create<VPUIP::CopyOp>(sourceOp->getLoc(), opResult, newCmxInput);
            opResult.replaceUsesWithIf(copyDDRToCMX, [&](mlir::OpOperand& operand) {
                return llvm::is_contained(inputOpsWithSameSource, operand.getOwner());
            });
            return copyDDRToCMX;
        };

        auto opBeforeCmxToDdrCopy = inputOpsWithSameSource.front();
        auto inputRoot = opBeforeCmxToDdrCopy->getOperand(0);
        auto cmxToDdrCopy = inputRoot.getDefiningOp();
        auto inputCopyInOp = inputCopy.getInput().getDefiningOp();
        if (inputCopyInOp == cmxToDdrCopy) {
            inputOpsWithSameSource.push_back(inputCopy);
        } else {
            handleCopy(inputCopyInOp);
            inferReturnTypes(inputCopy, vpux::InferShapedTypeMode::ALL);
        }

        auto copyDDRToCMX = handleCopy(cmxToDdrCopy);
        propagateReturnType(copyDDRToCMX, builder);

        // move origOp output back to DDR
        builder.setInsertionPointAfter(origOp);
        const auto origOpOutBuff = origOp.getOutputBuff();
        auto origOpOutBuffType = mlir::cast<NDTypeInterface>(origOpOutBuff.getType());
        auto newOutputType = origOpOutBuffType.changeMemSpace(VPU::MemoryKind::DDR);
        auto outputMemRefType = mlir::cast<mlir::MemRefType>(newOutputType);
        auto outputAllocInDdr = builder.create<mlir::memref::AllocOp>(origOp.getLoc(), outputMemRefType);
        auto copyOutputToDdr = builder.create<VPUIP::CopyOp>(origOp.getLoc(), origOp, outputAllocInDdr);
        origOp->getResult(0).replaceAllUsesExcept(copyOutputToDdr, copyOutputToDdr);
    });
}
}  // namespace

//
// createMoveReflectPadToCMXPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMoveReflectPadToCMXPass(Logger log) {
    return std::make_unique<MoveReflectPadToCMXPass>(log);
}
