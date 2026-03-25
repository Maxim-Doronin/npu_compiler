//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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

//
// Skip through layout operations (Reshape, AffineReshape, Transpose) that have single use
//
mlir::Operation* skipLayoutAndReshapeOps(mlir::Value value) {
    auto op = value.getDefiningOp();
    while (mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::TransposeOp>(op)) {
        if (!op->hasOneUse()) {
            return op;
        }

        // Check if this operation swaps the last 2 dimensions
        auto inputType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
        auto outputType = mlir::cast<NDTypeInterface>(op->getResult(0).getType());
        const auto inputShape = inputType.getShape();
        const auto outputShape = outputType.getShape();
        const auto inputRank = inputShape.size();
        const auto outputRank = outputShape.size();

        if (inputRank >= 2 && outputRank == inputRank) {
            // For Transpose, check if it swaps last 2 dimensions
            if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(op)) {
                auto order = transposeOp.getOrderValue();
                if (order.has_value()) {
                    auto permutation = order.value();
                    if (permutation.getDimPosition(inputRank - 2) == inputRank - 1 &&
                        permutation.getDimPosition(inputRank - 1) == inputRank - 2) {
                        return op;
                    }
                }
            }

            // For Reshape/AffineReshape, check if last 2 dimensions are swapped
            if (mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp>(op)) {
                const auto inputLast2Product = inputShape[Dim(inputRank - 2)] * inputShape[Dim(inputRank - 1)];
                const auto outputLast2Product = outputShape[Dim(outputRank - 2)] * outputShape[Dim(outputRank - 1)];

                if (inputLast2Product != outputLast2Product) {
                    return op;
                }

                if (inputShape[Dim(inputRank - 2)] == outputShape[Dim(outputRank - 1)] &&
                    inputShape[Dim(inputRank - 1)] == outputShape[Dim(outputRank - 2)] &&
                    inputShape[Dim(inputRank - 2)] != inputShape[Dim(inputRank - 1)]) {
                    return op;
                }
            }
        }

        op = op->getOperand(0).getDefiningOp();
    }
    return op;
}

//
// Optimize SDPA inputs by concentrating Batch over Channels
//
std::tuple<mlir::Value, mlir::Value, mlir::Value, mlir::Value, bool, SmallVector<int64_t>> optimizeSDPAInputs(
        mlir::OpBuilder& builder, mlir::Location loc, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV,
        mlir::Value mask) {
    const auto ctx = builder.getContext();
    auto queryType = mlir::cast<NDTypeInterface>(inputQ.getType());
    const auto rank = queryType.getRank();
    const auto queryShape = queryType.getShape();

    if (rank != 4) {
        return {inputQ, inputK, inputV, mask, false, SmallVector<int64_t>{}};
    }

    mlir::Value optimizedQ = inputQ;
    mlir::Value optimizedK = inputK;
    mlir::Value optimizedV = inputV;
    mlir::Value optimizedMask = mask;
    bool needsReshapeBack = false;
    SmallVector<int64_t> origOutputShape;

    const auto batch = queryShape.raw()[0];
    if (batch != 1) {
        auto getNCProduct = [](mlir::Value input) -> std::optional<int64_t> {
            if (!input) {
                return std::nullopt;
            }
            const auto inputType = mlir::cast<NDTypeInterface>(input.getType());
            const auto inputShape = inputType.getShape();
            const auto inputRank = inputShape.size();
            if (inputRank < 3 || inputRank > 4) {
                return std::nullopt;
            }
            const auto batch = (inputRank == 4) ? inputShape.raw()[0] : 1;
            return batch * inputShape.raw()[1];
        };

        const auto qNC = getNCProduct(inputQ);
        const auto kNC = getNCProduct(inputK);
        const auto vNC = getNCProduct(inputV);

        // Check if all N*C products are broadcastable
        bool compatibleShapes = true;
        if (qNC.has_value() && kNC.has_value() && !vpux::isBroadcastable(qNC.value(), kNC.value())) {
            compatibleShapes = false;
        }
        if (compatibleShapes && qNC.has_value() && vNC.has_value() &&
            !vpux::isBroadcastable(qNC.value(), vNC.value())) {
            compatibleShapes = false;
        }
        if (compatibleShapes && mask) {
            const auto maskNC = getNCProduct(mask);
            if (qNC.has_value() && maskNC.has_value() && !vpux::isBroadcastable(qNC.value(), maskNC.value())) {
                compatibleShapes = false;
            }
        }

        if (compatibleShapes) {
            auto reshapeBatchToChannels = [&](mlir::Value input, StringRef name) -> mlir::Value {
                if (!input) {
                    return input;
                }

                const auto inputType = mlir::cast<NDTypeInterface>(input.getType());
                const auto inputShape = inputType.getShape();
                const auto inputRank = inputShape.size();

                if (inputRank != 4) {
                    return input;
                }

                const auto inputBatch = inputShape.raw()[0];
                if (inputBatch != batch) {
                    return input;
                }

                const auto N = inputShape.raw()[0];
                const auto C = inputShape.raw()[1];
                const auto seqLen = inputShape.raw()[2];
                const auto headDim = inputShape.raw()[3];

                SmallVector<int64_t> newShape = {1, N * C, seqLen, headDim};
                auto reshapeOp =
                        builder.create<IE::ReshapeOp>(appendLoc(loc, name), input, getIntArrayAttr(ctx, newShape));
                return reshapeOp.getOutput();
            };

            optimizedQ = reshapeBatchToChannels(inputQ, "reshape_q_batch_to_channels");
            optimizedK = reshapeBatchToChannels(inputK, "reshape_k_batch_to_channels");
            optimizedV = reshapeBatchToChannels(inputV, "reshape_v_batch_to_channels");
            optimizedMask = reshapeBatchToChannels(mask, "reshape_mask_batch_to_channels");

            if (optimizedQ != inputQ || optimizedK != inputK || optimizedV != inputV || optimizedMask != mask) {
                needsReshapeBack = true;

                // Compute broadcasted output shape from Q, K, V
                const auto keyType = mlir::cast<NDTypeInterface>(inputK.getType());
                const auto valueType = mlir::cast<NDTypeInterface>(inputV.getType());
                const auto keyShape = keyType.getShape();
                const auto valueShape = valueType.getShape();

                const auto maxBatch = std::max({queryShape.raw()[0], keyShape.raw()[0], valueShape.raw()[0]});
                const auto maxChannels = std::max({queryShape.raw()[1], keyShape.raw()[1], valueShape.raw()[1]});
                const auto seqLen = queryShape.raw()[2];
                const auto headDim = valueShape.raw()[2];

                origOutputShape = {maxBatch, maxChannels, seqLen, headDim};
            }
        }
    }

    return {optimizedQ, optimizedK, optimizedV, optimizedMask, needsReshapeBack, origOutputShape};
}

void createSDPAExtendedFromSDPA(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV,
                                mlir::Value mask, mlir::Value scale) {
    auto builder = mlir::OpBuilder(op);
    const auto ctx = builder.getContext();
    const auto loc = op->getLoc();

    if (scale == nullptr) {
        auto inputQType = mlir::cast<NDTypeInterface>(inputQ.getType());
        auto eDim = inputQType.getShape().back();
        float scaleValue = 1.0f / std::sqrt(static_cast<float>(eDim));
        SmallVector<int64_t> scaleShape(inputQType.getRank(), 1);
        const auto scaleTensorType = mlir::RankedTensorType::get(scaleShape, builder.getF32Type());
        const auto scaleLoc = appendLoc(loc, "scale");
        scale = Const::createConst(builder, scaleLoc, scaleTensorType, llvm::ArrayRef<float>{scaleValue});
    }

    auto [finalQ, finalK, finalV, finalMask, needsReshapeBack, origOutputShape] =
            optimizeSDPAInputs(builder, loc, inputQ, inputK, inputV, mask);

    auto sdpaOp = builder.create<IE::SDPAExtendedOp>(appendLoc(loc, "sdpa_extended_full"), finalQ, finalK, finalV,
                                                     finalMask, scale, nullptr, nullptr);

    if (needsReshapeBack) {
        auto reshapeBackOp = builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_output_channels_to_batch"),
                                                           sdpaOp.getOutput(), getIntArrayAttr(ctx, origOutputShape));
        op->replaceAllUsesWith(reshapeBackOp);
    } else {
        op->replaceAllUsesWith(sdpaOp);
    }

    op->erase();
}

void createSDPAExtended(mlir::Operation* op, mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV,
                        mlir::Value mask, mlir::Value scale, mlir::Value bias) {
    auto builder = mlir::OpBuilder(op);
    const auto ctx = builder.getContext();
    const auto loc = op->getLoc();

    auto [finalQ, finalK, finalV, finalMask, needsReshapeBack, origOutputShape] =
            optimizeSDPAInputs(builder, loc, inputQ, inputK, inputV, mask);

    auto sdpaOp = builder.create<IE::SDPAExtendedOp>(appendLoc(loc, "sdpa_extended_pattern"), finalQ, finalK, finalV,
                                                     finalMask, scale, bias, nullptr);

    if (needsReshapeBack) {
        auto reshapeBackOp = builder.create<IE::ReshapeOp>(appendLoc(loc, "reshape_output_channels_to_batch"),
                                                           sdpaOp.getOutput(), getIntArrayAttr(ctx, origOutputShape));
        op->replaceAllUsesWith(reshapeBackOp);
    } else {
        op->replaceAllUsesWith(sdpaOp);
    }

    op->erase();
}

//
// safeRunOnFunc
//
void FuseSDPAExtendedPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](IE::SDPAOp sdpaOp) {
        auto inputKType = mlir::cast<NDTypeInterface>(sdpaOp.getInputK().getType());
        auto inputVType = mlir::cast<NDTypeInterface>(sdpaOp.getInputV().getType());
        const auto rank = inputKType.getRank();
        if (rank < 2 || rank > 4) {
            return;
        }

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

        createSDPAExtendedFromSDPA(sdpaOp, sdpaOp.getInputQ(), sdpaOp.getInputK(), transposedVOp.getOutput(),
                                   sdpaOp.getInputMask(), sdpaOp.getInputScale());
    });

    // Custom Attention pattern detection and fusion into SDPAExtended
    func->walk([&](IE::MatMulOp matMulVOp) {
        mlir::Operation* currentOp = skipLayoutAndReshapeOps(matMulVOp.getInput1());

        auto softmaxOp = mlir::dyn_cast_or_null<IE::SoftMaxOp>(currentOp);
        if (softmaxOp == nullptr) {
            return;
        }

        currentOp = skipLayoutAndReshapeOps(softmaxOp.getInput());

        mlir::Value attentionMask = nullptr;
        mlir::Value bias = nullptr;

        if (auto firstAddOp = mlir::dyn_cast_or_null<IE::AddOp>(currentOp); firstAddOp != nullptr) {
            currentOp = skipLayoutAndReshapeOps(firstAddOp.getInput1());

            if (auto secondAddOp = mlir::dyn_cast_or_null<IE::AddOp>(currentOp)) {
                bias = firstAddOp.getInput2();
                attentionMask = secondAddOp.getInput2();
                currentOp = skipLayoutAndReshapeOps(secondAddOp.getInput1());
            } else {
                attentionMask = firstAddOp.getInput2();
            }
        }

        mlir::Value scale = nullptr;
        if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(currentOp)) {
            scale = multiplyOp.getInput2();
            currentOp = skipLayoutAndReshapeOps(multiplyOp.getInput1());
        }

        auto qkMatMulOp = mlir::dyn_cast_or_null<IE::MatMulOp>(currentOp);
        if (qkMatMulOp == nullptr) {
            return;
        }

        mlir::Value inputQ = qkMatMulOp.getInput1();
        mlir::Value inputK = qkMatMulOp.getInput2();

        // Check all input ranks before any graph modifications.
        auto hasInvalidRank = [](mlir::Value val) -> bool {
            if (val == nullptr) {
                return false;
            }
            const auto r = mlir::cast<NDTypeInterface>(val.getType()).getRank();
            return r < 2 || r > 4;
        };
        if (hasInvalidRank(inputQ) || hasInvalidRank(inputK) || hasInvalidRank(matMulVOp.getInput2()) ||
            hasInvalidRank(attentionMask) || hasInvalidRank(scale) || hasInvalidRank(bias)) {
            return;
        }

        auto extractScaleAndReapplyTransforms = [&](mlir::Value input) -> std::pair<mlir::Value, mlir::Value> {
            SmallVector<mlir::Operation*> layoutOps;
            mlir::Value current = input;
            while (auto defOp = current.getDefiningOp()) {
                if (mlir::isa<IE::TransposeOp, IE::ReshapeOp, IE::AffineReshapeOp>(defOp)) {
                    if (!defOp->hasOneUse()) {
                        break;
                    }
                    layoutOps.push_back(defOp);
                    current = defOp->getOperand(0);
                } else {
                    break;
                }
            }

            mlir::Value scale = nullptr;
            mlir::Value baseInput = input;

            if (auto multiplyOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(current.getDefiningOp())) {
                if (multiplyOp.getOutput().hasOneUse()) {
                    // Single use: extract scale, reroute and delete multiply
                    scale = multiplyOp.getInput2();
                    if (hasInvalidRank(scale)) {
                        return {input, nullptr};
                    }

                    if (!layoutOps.empty()) {
                        // Reconnect the first layout op to bypass the multiply
                        layoutOps.back()->setOperand(0, multiplyOp.getInput1());
                    } else {
                        // No layout ops, update baseInput directly
                        baseInput = multiplyOp.getInput1();
                    }
                }
            }

            return {baseInput, scale};
        };

        // If scale is not found in the direct path, check if it's applied to Q or K with transformations
        if (!scale) {
            auto [qBase, qScale] = extractScaleAndReapplyTransforms(inputQ);
            if (qScale) {
                scale = qScale;
                inputQ = qBase;
            } else {
                auto [kBase, kScale] = extractScaleAndReapplyTransforms(inputK);
                if (kScale) {
                    scale = kScale;
                    inputK = kBase;
                }
            }
        }

        mlir::Value inputV = matMulVOp.getInput2();

        auto builder = mlir::OpBuilder(matMulVOp);
        const auto ctx = builder.getContext();

        auto addTransposeIfNeeded = [&](mlir::Value& input, bool needsTranspose, StringRef name) {
            if (!needsTranspose) {
                return;
            }

            auto inputType = mlir::cast<NDTypeInterface>(input.getType());
            const auto rank = inputType.getRank();
            SmallVector<uint32_t> permuteOrder;
            for (int i = 0; i < rank - 2; i++) {
                permuteOrder.push_back(i);
            }
            permuteOrder.push_back(rank - 1);
            permuteOrder.push_back(rank - 2);

            auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permuteOrder, ctx));
            auto transposeOp = builder.create<IE::TransposeOp>(
                    appendLoc(matMulVOp.getLoc(), formatv("transpose_{0}_for_sdpa", name)), input, nullptr, orderAttr);

            input = transposeOp.getOutput();
        };

        auto inputQType = mlir::cast<NDTypeInterface>(inputQ.getType());
        auto inputKType = mlir::cast<NDTypeInterface>(inputK.getType());
        auto inputVType = mlir::cast<NDTypeInterface>(inputV.getType());

        if (inputKType.getRank() >= 2 && inputVType.getRank() >= 2 && inputKType.getRank() == inputVType.getRank() &&
            inputQType.getRank() == inputKType.getRank()) {
            const auto rank = inputKType.getRank();
            const auto inputQShape = inputQType.getShape().raw();
            auto inputKShape = inputKType.getShape().raw();
            const auto inputVShape = inputVType.getShape().raw();

            // Check if K needs transpose: Q[..., tSL, e], K[..., sSL, e]
            bool kNeedsTranspose = inputKShape[rank - 1] != inputQShape[rank - 1];
            addTransposeIfNeeded(inputK, kNeedsTranspose, "k");

            // Update K shape after potential transpose
            if (kNeedsTranspose) {
                inputKShape = mlir::cast<NDTypeInterface>(inputK.getType()).getShape().raw();
            }

            // Check if V needs transpose: K[..., sSL, e], V[..., eV, sSL]
            bool vNeedsTranspose = inputKShape[rank - 2] != inputVShape[rank - 1];
            addTransposeIfNeeded(inputV, vNeedsTranspose, "v");
        }

        createSDPAExtended(matMulVOp, inputQ, inputK, inputV, attentionMask, scale, bias);
    });
}

}  // namespace

//
// createFuseSDPAExtendedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseSDPAExtendedPass(Logger log) {
    return std::make_unique<FuseSDPAExtendedPass>(log);
}
