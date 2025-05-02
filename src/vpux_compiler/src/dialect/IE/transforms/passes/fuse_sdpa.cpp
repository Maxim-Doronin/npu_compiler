//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESDPA
#define GEN_PASS_DEF_FUSESDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseSDPAPass
//

class FuseSDPAPass final : public IE::impl::FuseSDPABase<FuseSDPAPass> {
public:
    explicit FuseSDPAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

// Supported SDPA Configurations:
// InputQ    {N, C, 1, W}
// InputK    {N, C, H, W}
// InputV    {N, C, W, H}
// InputMask {1, 1, 1, W}

// General Pattern with Multiply (To be enabled in E#160851)
// InputQ --------------> MatMul ---> Add ---> Softmax ---> MatMul ---> Output
//                           ^         ^                       ^
//                           |         |                       |
// InputK ----> Multiply -----         |                       |
//                                     |                       |
// InputMask ---------------------------                       |
//                                                             |
// InputV ------------------------------------------------------

// General Pattern with Divide (To be enabled in E#160851)
// InputQ --------------> MatMul ---> Divide---> Add ---> Softmax ---> MatMul ---> Output
//                           ^                    ^                       ^
//                           |                    |                       |
// InputK --------------------                    |                       |
//                                                |                       |
// InputMask --------------------------------------                       |
//                                                                        |
// InputV -----------------------------------------------------------------

// FullyConnected Pattern with Multiply
// InputQ ---------> Reshape -----> FullyConnected -> Reshape -> Add -> Softmax -> Reshape -> FullyConnected -> Reshape
//                                             ^                   ^                                ^
//                                             |                   |                                |
// InputK -> Multiply -> Transpose -> Reshape -                    |                                |
//                                                                 |                                |
// InputMask -------------------------------------------------------                                |
//                                                                                                  |
// InputV ---> Reshape ------------------------------------------------------------------------------

// FullyConnected Pattern with Divide
// InputQ -> Reshape -> FullyConnected -> Reshape -> Div/Mul -> Add -> Softmax -> Reshape -> FullyConnected -> Reshape
//                            ^                                  ^                                ^
//                            |                                  |                                |
// InputK -> Reshape ----------                                  |                                |
//                                                               |                                |
// InputMask -----------------------------------------------------                                |
//                                                                                                |
// InputV ---> Reshape ---------------------------------------------------------------------------

// AdaptiveStripping Pattern with Multiply
// InputQ-> Reshape-> FullyConnected-> Reshape-> Multiply-> Add-> Softmax-> ReLU-> Reshape-> FullyConnected-> Reshape
//                            ^                               ^                                        ^
//                            |                               |                                        |
// InputK -> Reshape ----------                               |                                        |
//                                                            |                                        |
// InputMask --------------------------------------------------                                        |
//                                                                                                     |
// InputV ---> Reshape ---------------------------------------------------------------------------------

bool isMatMulAsFC(mlir::Value output, mlir::Value& input1, mlir::Value& input2) {
    auto reshapeOutOp = output.getDefiningOp<IE::ReshapeOp>();
    if (!reshapeOutOp) {
        return false;
    }
    auto fcOp = reshapeOutOp->getOperand(0).getDefiningOp<IE::FullyConnectedOp>();
    if (!fcOp) {
        return false;
    }
    auto reshapeInput1Op = fcOp->getOperand(0).getDefiningOp<IE::ReshapeOp>();
    if (!reshapeInput1Op) {
        return false;
    }
    auto reshapeInput2Op = fcOp->getOperand(1).getDefiningOp<IE::ReshapeOp>();
    if (!reshapeInput2Op) {
        return false;
    }
    input1 = reshapeInput1Op->getOperand(0);
    input2 = reshapeInput2Op->getOperand(0);
    return true;
}

bool isAdaptiveStripping(mlir::Value output, mlir::Value& input1, mlir::Value& input2, mlir::Value& inputMask) {
    auto reluOp = output.getDefiningOp<IE::ReLUOp>();
    if (!reluOp) {
        return false;
    }
    auto softmaxOp = reluOp->getOperand(0).getDefiningOp<IE::SoftMaxOp>();
    if (!softmaxOp) {
        return false;
    }
    auto addOp = softmaxOp->getOperand(0).getDefiningOp<IE::AddOp>();
    if (!addOp) {
        return false;
    }
    auto multiplyOp = addOp->getOperand(0).getDefiningOp<IE::MultiplyOp>();
    if (!multiplyOp) {
        return false;
    }
    if (!isMatMulAsFC(multiplyOp->getOperand(0), input1, input2)) {
        return false;
    }
    inputMask = addOp->getOperand(1);
    return true;
}

void createSDPA(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV, mlir::Value mask,
                IE::TransposeOp transposeK = nullptr) {
    auto builder = mlir::OpBuilder(op);
    if (transposeK) {
        auto transposedKOp = builder.create<IE::TransposeOp>(appendLoc(op->getLoc(), "_transposed_k"), inputK, nullptr,
                                                             transposeK.getOrderValueAttr());
        auto sdpaOp = builder.create<IE::SDPAOp>(appendLoc(op->getLoc(), "_sdpa"), inputQ, transposedKOp, inputV, mask);
        op->replaceAllUsesWith(sdpaOp);
    } else {
        auto sdpaOp = builder.create<IE::SDPAOp>(appendLoc(op->getLoc(), "_sdpa"), inputQ, inputK, inputV, mask);
        op->replaceAllUsesWith(sdpaOp);
    }
}

mlir::Operation* getScaleOp(mlir::Operation* op) {
    auto divideOp = mlir::dyn_cast_or_null<IE::DivideOp>(op);
    if (divideOp != nullptr && divideOp->hasOneUse()) {
        return divideOp;
    }

    auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(op);
    if (multiplyOp != nullptr && multiplyOp->hasOneUse()) {
        return multiplyOp;
    }

    return nullptr;
}

bool isLegalSDPA(mlir::Value inputQ) {
    auto tensorType = mlir::cast<NDTypeInterface>(inputQ.getType());
    const auto rank = tensorType.getRank();
    if (rank < 2 || rank > 4) {
        return false;
    }
    auto shape = tensorType.getShape().raw();
    const int supportedH = 1;
    const int supportedW = 64;
    if (shape[rank - 2] != supportedH || shape[rank - 1] != supportedW) {
        return false;
    }
    return true;
}

void FuseSDPAPass::safeRunOnFunc() {
    auto func = getOperation();
    const auto arch = VPU::getArch(func);
    // Force to fuse only on 40XX for now
    if (arch != VPU::ArchKind::NPU40XX) {
        return;
    }
    func->walk([&](IE::ReshapeOp reshape0Op) {
        auto fc0Op = reshape0Op.getOperand(0).getDefiningOp<IE::FullyConnectedOp>();
        if (!fc0Op) {
            return;
        }
        auto reshape1Op = fc0Op->getOperand(0).getDefiningOp<IE::ReshapeOp>();
        if (!reshape1Op) {
            return;
        }
        auto reshapeVOp = fc0Op->getOperand(1).getDefiningOp<IE::ReshapeOp>();
        if (!reshapeVOp) {
            return;
        }
        mlir::Value inputV = reshapeVOp->getOperand(0);
        mlir::Value inputK = nullptr;
        mlir::Value inputQ = nullptr;
        mlir::Value mask = nullptr;
        IE::TransposeOp transposeK = nullptr;

        auto softmaxOp = reshape1Op->getOperand(0).getDefiningOp<IE::SoftMaxOp>();
        if (!softmaxOp) {
            if (isAdaptiveStripping(reshape1Op->getOperand(0), inputQ, inputK, mask)) {
                if (!isLegalSDPA(inputQ)) {
                    return;
                }
                createSDPA(reshape0Op, inputQ, inputK, inputV, mask);
            }
            return;
        }
        auto addOp = softmaxOp->getOperand(0).getDefiningOp<IE::AddOp>();
        if (!addOp) {
            return;
        }
        mask = addOp->getOperand(1);
        auto scaleOp = getScaleOp(addOp->getOperand(0).getDefiningOp());
        if (!scaleOp) {
            if (!isMatMulAsFC(addOp->getOperand(0), inputQ, inputK)) {
                return;
            }
            transposeK = inputK.getDefiningOp<IE::TransposeOp>();
            if (!transposeK) {
                return;
            }
            auto multiplyOp = transposeK->getOperand(0).getDefiningOp<IE::MultiplyOp>();
            if (!multiplyOp) {
                return;
            }
            inputK = multiplyOp->getOperand(0);
        } else {
            if (!isMatMulAsFC(scaleOp->getOperand(0), inputQ, inputK)) {
                return;
            }
        }

        // For performance increase for certain networks, we limit the supported dimensions to {N, C, 1, 64} for inputQ
        // Follow next ticket for updates on supported dimensions:
        // E#160851
        if (!isLegalSDPA(inputQ)) {
            return;
        }
        createSDPA(reshape0Op, inputQ, inputK, inputV, mask, transposeK);
    });
}

}  // namespace

//
// createFuseSDPAPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSDPAPass(Logger log) {
    return std::make_unique<FuseSDPAPass>(log);
}
