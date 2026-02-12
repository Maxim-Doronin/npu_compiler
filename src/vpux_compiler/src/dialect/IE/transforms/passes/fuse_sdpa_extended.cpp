//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/type/float16.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESDPAEXTENDED
#define GEN_PASS_DEF_FUSESDPAEXTENDED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseSDPAExtendedPass
//

class FuseSDPAExtendedPass final : public IE::impl::FuseSDPAExtendedBase<FuseSDPAExtendedPass> {
public:
    explicit FuseSDPAExtendedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MM-SM-MM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void createMatMulSoftmaxMatMul(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV) {
    auto builder = mlir::OpBuilder(op);
    auto sdpaOp = builder.create<IE::SDPAExtendedOp>(appendLoc(op->getLoc(), "sdpa_extended"), inputQ, inputK, inputV,
                                                     nullptr, nullptr, nullptr, nullptr);
    op->replaceAllUsesWith(sdpaOp);
}

bool isLegalFusibleMmSmMm(mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV) {
    auto tensorTypeQ = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto tensorTypeK = mlir::cast<NDTypeInterface>(inputK.getType());
    auto tensorTypeV = mlir::cast<NDTypeInterface>(inputV.getType());
    const auto rankQ = tensorTypeQ.getRank();
    const auto rankK = tensorTypeK.getRank();
    const auto rankV = tensorTypeV.getRank();
    auto shapeQ = tensorTypeQ.getShape().raw();
    auto shapeK = tensorTypeK.getShape().raw();
    if (rankQ < 2 || rankQ > 4) {
        return false;
    }
    if (rankK < 2 || rankK > 4) {
        return false;
    }
    if (rankV < 2 || rankV > 4) {
        return false;
    }

    auto batchSize = shapeQ[0] * shapeQ[1];
    auto L = shapeQ[rankQ - 2];
    auto S = shapeK[rankK - 2];

    // experimental restriction based on current shave implementation
    if ((batchSize < 24) || (L < 128) || (S < 128) || (L > 256) || (S > 256)) {
        return false;
    }

    return true;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SDPA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void createSDPA(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV, mlir::Value mask,
                mlir::Value scale, mlir::Value bias, IE::TransposeOp transposeK = nullptr) {
    auto builder = mlir::OpBuilder(op);
    mlir::Value sdpaKInput = inputK;
    if (transposeK) {
        auto transposedKOp = builder.create<IE::TransposeOp>(appendLoc(op->getLoc(), "transposed_k"), inputK, nullptr,
                                                             transposeK.getOrderValueAttr());
        sdpaKInput = transposedKOp.getOutput();
    }
    if (scale == nullptr) {
        // Get eDim from inputQ type (assuming inputQ is a ranked tensor)
        auto inputQType = mlir::cast<NDTypeInterface>(inputQ.getType());
        VPUX_THROW_UNLESS(inputQType != nullptr, "inputQ must be a RankedTensorType");
        auto eDim = inputQType.getShape().back();
        float scaleValue = 1.0f / std::sqrt(static_cast<float>(eDim));
        const auto scaleTensorType = mlir::RankedTensorType::get({1, 1, 1, 1}, builder.getF32Type());
        const auto loc = appendLoc(op->getLoc(), "scale");
        scale = Const::createConst(builder, loc, scaleTensorType, llvm::ArrayRef<float>{scaleValue});
    }

    auto sdpaOp = builder.create<IE::SDPAExtendedOp>(appendLoc(op->getLoc(), "sdpa_extended_full"), inputQ, sdpaKInput,
                                                     inputV, mask, scale, bias, nullptr);
    op->replaceAllUsesWith(sdpaOp);
}

bool isLegalSDPA(mlir::Value inputQ) {
    auto tensorType = mlir::cast<NDTypeInterface>(inputQ.getType());
    const auto rank = tensorType.getRank();
    if (rank < 2 || rank > 4) {
        return false;
    }
    return true;
}

//
// safeRunOnFunc
//
void FuseSDPAExtendedPass::safeRunOnFunc() {
    auto func = getOperation();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MM-SM-MM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Detect MatMul - (Reshapes (+ ReLU)) - Softmax - (Reshapes (+ ReLU)) - MatMul
    func->walk([&](IE::MatMulOp matMul2Op) {
        auto skipOptionalOpsIfPresent = [](mlir::Operation* op) -> mlir::Operation* {
            // No Reshapes
            if (!mlir::isa_and_nonnull<IE::ReshapeOp>(op)) {
                return op;
            }
            auto nextOp = op->getOperand(0).getDefiningOp();
            if (!mlir::isa_and_nonnull<IE::ReshapeOp>(nextOp)) {
                if (!mlir::isa_and_nonnull<IE::ReLUOp>(nextOp)) {
                    return nextOp;
                }
                auto nextNextOp = nextOp->getOperand(0).getDefiningOp();
                if (!mlir::isa_and_nonnull<IE::ReshapeOp>(nextNextOp)) {
                    return nextNextOp;
                }
                return nextNextOp->getOperand(0).getDefiningOp();
            }
            if (!op->hasOneUse() || !nextOp->hasOneUse()) {
                return nullptr;
            }
            // Double Reshapes
            return nextOp->getOperand(0).getDefiningOp();
        };
        auto softmaxOp = mlir::dyn_cast_or_null<IE::SoftMaxOp>(
                skipOptionalOpsIfPresent(matMul2Op.getOperand(0).getDefiningOp()));
        if (!softmaxOp) {
            return;
        }
        auto matMul1Op = mlir::dyn_cast_or_null<IE::MatMulOp>(
                skipOptionalOpsIfPresent(softmaxOp->getOperand(0).getDefiningOp()));
        if (!matMul1Op) {
            return;
        }

        // Create SDPAExtended Operator
        mlir::Value inputQ = matMul1Op->getOperand(0);
        mlir::Value inputK = matMul1Op->getOperand(1);
        mlir::Value inputV = matMul2Op.getOperand(1);
        if ((!isLegalFusibleMmSmMm(inputQ, inputK, inputV))) {
            return;
        }
        createMatMulSoftmaxMatMul(matMul2Op, inputQ, inputK, inputV);
    });

    // Detect [Reshape - FullyConnected - Reshape] - Softmax - [Reshape - FullyConnected - Reshape]
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
        mlir::Value inputK = mlir::Value();
        mlir::Value inputQ = mlir::Value();
        auto softmaxOp = reshape1Op->getOperand(0).getDefiningOp<IE::SoftMaxOp>();
        if (!softmaxOp) {
            return;
        }
        if (!isMatMulAsFC(softmaxOp->getOperand(0), inputQ, inputK)) {
            return;
        }

        if ((!isLegalFusibleMmSmMm(inputQ, inputK, inputV))) {
            return;
        }
        createMatMulSoftmaxMatMul(reshape0Op, inputQ, inputK, inputV);
    });

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SDPA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Real SDPA op convert to SDPAExtended
    func->walk([&](IE::SDPAOp sdpaOp) {
        auto inputKType = mlir::cast<NDTypeInterface>(sdpaOp.getInputK().getType());
        auto inputVType = mlir::cast<NDTypeInterface>(sdpaOp.getInputV().getType());
        const auto rank = inputKType.getRank();

        const auto inputKShape = inputKType.getShape().raw();
        const auto inputVShape = inputVType.getShape().raw();
        auto inputV = sdpaOp.getInputV();
        if (inputKShape[rank - 2] != inputVShape[rank - 2]) {
            return;
        }
        auto builder = mlir::OpBuilder(sdpaOp);
        SmallVector<uint32_t> permuteNdOrder = {};
        for (int i = 0; i < rank - 2; i++) {
            permuteNdOrder.push_back(i);
        }
        permuteNdOrder.push_back(rank - 1);
        permuteNdOrder.push_back(rank - 2);
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permuteNdOrder, builder.getContext()));
        auto transposedVOp =
                builder.create<IE::TransposeOp>(appendLoc(sdpaOp->getLoc(), "transpose_v"), inputV, nullptr, orderAttr);

        if (!isLegalSDPA(sdpaOp.getInputQ())) {
            return;
        }

        createSDPA(sdpaOp, sdpaOp.getInputQ(), sdpaOp.getInputK(), transposedVOp.getOutput(), sdpaOp.getInputMask(),
                   sdpaOp.getInputScale(), nullptr, nullptr);
    });
}

}  // namespace

//
// createFuseSDPAExtendedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSDPAExtendedPass(Logger log) {
    return std::make_unique<FuseSDPAExtendedPass>(log);
}
