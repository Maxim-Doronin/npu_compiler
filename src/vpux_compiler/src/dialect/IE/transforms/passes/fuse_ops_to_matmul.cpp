// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/Utils/IndexingUtils.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOPSTOMATMUL
#define GEN_PASS_DEF_FUSEOPSTOMATMUL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Minimum M*N*K product to justify the MatMul rewrite.
constexpr int64_t MIN_MATMUL_ELEMENTS = 4096;

// Returns true when a batched MatMul with the given dimensions can be lowered to VPU.NCE.MatMul on the current target.
static bool isBroadcastMultiplyReduceSumBeneficialAsNCEMatMul(IE::MultiplyOp mulOp, int64_t batchSize, int64_t M,
                                                              int64_t N, int64_t K, bool enableGroupedMatMul) {
    if (!enableGroupedMatMul) {
        return false;
    }

    if (M * N * K < MIN_MATMUL_ELEMENTS) {
        return false;
    }

    const auto moduleOp = mulOp->getParentOfType<mlir::ModuleOp>();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(mulOp.getOutput().getType());

    const Shape input5DShape{batchSize, static_cast<int64_t>(1), K, M, static_cast<int64_t>(1)};
    const Shape filter5DShape{batchSize, N, K, static_cast<int64_t>(1), static_cast<int64_t>(1)};
    const Shape output5DShape{batchSize, static_cast<int64_t>(1), N, M, static_cast<int64_t>(1)};

    const auto emptyLogCb = [](const formatv_object_base&) {};
    return VPU::isNCEMatMulSupported(outputType.changeShape(input5DShape), outputType.changeShape(filter5DShape),
                                     outputType.changeShape(output5DShape), moduleOp, emptyLogCb,
                                     /*checkLayout=*/false, /*checkChannelAlignment=*/false);
}

class BroadcastMultiplyReduceSumToMatMulRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    BroadcastMultiplyReduceSumToMatMulRewriter(mlir::MLIRContext* ctx, Logger log, bool enableGroupedMatMul)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log), _enableGroupedMatMul(enableGroupedMatMul) {
        setDebugName("BroadcastMultiplyReduceSumToMatMulRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult BroadcastMultiplyReduceSumToMatMulRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum at '{1}'", getDebugName(), origOp->getLoc());

    // 1. The input to ReduceSum must be a single-use broadcast Multiply
    auto mulOp = origOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (mulOp == nullptr || !mulOp->hasOneUse()) {
        return mlir::failure();
    }

    // 2. Check the Multiply has NUMPY broadcast
    auto broadcastAttr = mulOp.getAutoBroadcast();
    if (broadcastAttr != IE::AutoBroadcastType::NUMPY) {
        return mlir::failure();
    }

    // 2b. Guard element type: IE.MatMul only supports F16/F32/F64/SI32/quant.
    //     IE.Multiply additionally allows SI64/UI8/UI16/UI32/UI64, which IE.MatMul rejects.
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(mulOp.getOutput().getType()).getElementType();
    const bool isSupportedByMatMul = elemType.isF16() || elemType.isF32() || elemType.isF64() ||
                                     elemType.isInteger(32) || mlir::isa<mlir::quant::QuantizedType>(elemType);
    if (!isSupportedByMatMul) {
        _log.trace("[{0}] Element type '{1}' not supported by IE.MatMul, skipping", getDebugName(), elemType);
        return mlir::failure();
    }

    // 3. Get shapes and identify broadcast pattern: exactly two dims differ (one is 1 in each input)
    const auto mulOutShape = getShape(mulOp.getOutput());
    const auto lhsShape = getShape(mulOp.getInput1());
    const auto rhsShape = getShape(mulOp.getInput2());
    const auto rank = mulOutShape.size();

    if (rank < 3 || lhsShape.size() != rank || rhsShape.size() != rank) {
        return mlir::failure();
    }

    // Find which dims have broadcast-from-1 on LHS vs RHS
    // dim where LHS has 1 but RHS has >1
    int64_t lhsBroadcastDim = -1;
    // dim where RHS has 1 but LHS has >1
    int64_t rhsBroadcastDim = -1;

    for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
        const bool lhsIs1 = (lhsShape[Dim(i)] == 1 && mulOutShape[Dim(i)] > 1);
        const bool rhsIs1 = (rhsShape[Dim(i)] == 1 && mulOutShape[Dim(i)] > 1);
        if (lhsIs1 && !rhsIs1) {
            if (lhsBroadcastDim != -1) {
                _log.trace("multiple broadcast dims on LHS");
                return mlir::failure();
            }
            lhsBroadcastDim = i;
        } else if (rhsIs1 && !lhsIs1) {
            if (rhsBroadcastDim != -1) {
                _log.trace("multiple broadcast dims on RHS");
                return mlir::failure();
            }
            rhsBroadcastDim = i;
        }
    }

    if (lhsBroadcastDim == -1 || rhsBroadcastDim == -1) {
        _log.trace("not a dual-axis broadcast (outer product)");
        return mlir::failure();
    }

    // 4. The ReduceSum must reduce over exactly one contraction (K) dimension that is
    //    neither of the two broadcast dims (those are M and N).
    const auto axes = IE::extractAxes(origOp->getLoc(), origOp);
    if (axes.size() != 1) {
        return mlir::failure();
    }

    // Normalize the axis: convert negative values to positive and validate bounds.
    int64_t reduceAxis = axes.front();
    if (reduceAxis < 0) {
        reduceAxis += static_cast<int64_t>(rank);
    }
    if (reduceAxis < 0 || reduceAxis >= static_cast<int64_t>(rank)) {
        return mlir::failure();
    }
    if (reduceAxis == lhsBroadcastDim || reduceAxis == rhsBroadcastDim) {
        return mlir::failure();
    }

    // Both inputs must have the same size on the K (contraction) dimension.
    if (lhsShape[Dim(reduceAxis)] != rhsShape[Dim(reduceAxis)]) {
        return mlir::failure();
    }

    // 4b. When reduceAxis is not the last dim, compute the permutation that moves K to last.
    //     All downstream steps use the "effective" (post-permutation) dim indices and shape.
    const bool needKToLastTranspose = (reduceAxis != static_cast<int64_t>(rank) - 1);
    SmallVector<int64_t> kToLastPerm;
    for (size_t i = 0; i < rank; ++i) {
        if (static_cast<int64_t>(i) != reduceAxis) {
            kToLastPerm.push_back(static_cast<int64_t>(i));
        }
    }
    kToLastPerm.push_back(static_cast<int64_t>(reduceAxis));

    // Map original dim index d → its new position after kToLastPerm.
    const auto remapDim = [&](int64_t d) -> int64_t {
        for (size_t i = 0; i < kToLastPerm.size(); ++i) {
            if (kToLastPerm[i] == d) {
                return static_cast<int64_t>(i);
            }
        }
        VPUX_THROW("remapDim: dim {0} not found in kToLastPerm", d);
    };

    const int64_t effectiveLhsBroadcastDim = needKToLastTranspose ? remapDim(lhsBroadcastDim) : lhsBroadcastDim;
    const int64_t effectiveRhsBroadcastDim = needKToLastTranspose ? remapDim(rhsBroadcastDim) : rhsBroadcastDim;
    const int64_t effectiveReduceAxis = static_cast<int64_t>(rank) - 1;

    // Virtual output shape after kToLastPerm (used for M/N/K/batchSize derivation).
    const auto effectiveOutShapeVec = mlir::applyPermutation(mulOutShape.raw(), kToLastPerm);

    // Compute the batch size using effective dims.
    int64_t batchSize = 1;
    SmallVector<int64_t> batchDims;
    for (size_t i = 0; i < rank; ++i) {
        if (static_cast<int64_t>(i) != effectiveLhsBroadcastDim &&
            static_cast<int64_t>(i) != effectiveRhsBroadcastDim && static_cast<int64_t>(i) != effectiveReduceAxis) {
            batchDims.push_back(static_cast<int64_t>(i));
            batchSize *= effectiveOutShapeVec[i];
        }
    }

    const auto M = effectiveOutShapeVec[effectiveRhsBroadcastDim];
    const auto N = effectiveOutShapeVec[effectiveLhsBroadcastDim];
    const auto K = effectiveOutShapeVec[effectiveReduceAxis];

    _log.trace("[{0}] Detected outer product + reduce: batch={1}, M={2}, N={3}, K={4}", getDebugName(), batchSize, M, N,
               K);

    // 5. Guard: only convert when the resulting IE.MatMul can be lowered to VPU.NCE.MatMul.
    if (!isBroadcastMultiplyReduceSumBeneficialAsNCEMatMul(mulOp, batchSize, M, N, K, _enableGroupedMatMul)) {
        _log.trace("[{0}] NCE.MatMul not supported for batchSize={1}, skipping conversion", getDebugName(), batchSize);
        return mlir::failure();
    }

    // 6. Build the MatMul
    const auto origLoc = origOp->getLoc();
    const auto ctx = rewriter.getContext();

    // 6a. If reduceAxis is not the last dim, pre-transpose both inputs to move K to last.
    //     The broadcast dim (size=1) in each input is also moved, which is harmless.
    mlir::Value lhsInput = mulOp.getInput1();
    mlir::Value rhsInput = mulOp.getInput2();
    if (needKToLastTranspose) {
        SmallVector<unsigned> kToLastPermUnsigned;
        kToLastPermUnsigned.reserve(kToLastPerm.size());
        for (const auto dim : kToLastPerm) {
            kToLastPermUnsigned.push_back(checked_cast<unsigned>(dim));
        }
        const auto kToLastPermAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(kToLastPermUnsigned, ctx));
        lhsInput = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "lhs_k_to_last"), lhsInput, nullptr,
                                                    kToLastPermAttr)
                           .getOutput();
        rhsInput = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "rhs_k_to_last"), rhsInput, nullptr,
                                                    kToLastPermAttr)
                           .getOutput();
    }

    // Check whether the input dims are already in the order needed for a direct reshape.
    // A Transpose is needed when this order differs from the identity [0,1,...,rank-1].
    const auto buildPerm = [&](int64_t mOrNDim, int64_t broadcastDim) -> SmallVector<unsigned> {
        SmallVector<unsigned> perm;
        for (const auto d : batchDims) {
            perm.push_back(checked_cast<unsigned>(d));
        }
        perm.push_back(checked_cast<unsigned>(mOrNDim));
        perm.push_back(checked_cast<unsigned>(broadcastDim));
        perm.push_back(checked_cast<unsigned>(effectiveReduceAxis));
        return perm;
    };

    // Build a transposed+reshaped 3D tensor [batchSize, mOrN, K] from the given input.
    const auto prepareInput = [&](mlir::Value input, int64_t mOrNDim, int64_t broadcastDim, int64_t mOrN,
                                  StringRef tag) -> mlir::Value {
        mlir::Value val = input;
        const auto perm = buildPerm(mOrNDim, broadcastDim);
        const auto permMap = mlir::AffineMap::getPermutationMap(perm, ctx);
        const auto inMemShape = MemShape(getShape(val).raw());
        if (!isTrivialPermute(inMemShape, permMap)) {
            val = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, (tag + "_transpose").str()), val, nullptr,
                                                   mlir::AffineMapAttr::get(permMap))
                          .getOutput();
        }
        // Reshape to [batchSize, mOrN, K], squeezing the broadcast-1 dim.
        const auto reshapeShape = SmallVector<int64_t>{batchSize, mOrN, K};
        return rewriter
                .create<IE::ReshapeOp>(appendLoc(origLoc, (tag + "_reshape").str()), val,
                                       getIntArrayAttr(ctx, reshapeShape))
                .getOutput();
    };

    // LHS: [batch, M, K]
    auto lhsReshaped = prepareInput(lhsInput, effectiveRhsBroadcastDim, effectiveLhsBroadcastDim, M, "matmul_lhs");
    // RHS: [batch, N, K]
    auto rhsReshaped = prepareInput(rhsInput, effectiveLhsBroadcastDim, effectiveRhsBroadcastDim, N, "matmul_rhs");

    // MatMul: [batch, M, K] @ [batch, N, K]^T -> [batch, M, N]
    auto matmulOp = rewriter.create<IE::MatMulOp>(appendLoc(origLoc, "broadcast_mul_reduce_as_matmul"), lhsReshaped,
                                                  rhsReshaped,
                                                  /*transpose_a=*/false, /*transpose_b=*/true);

    // 7. Reshape + optional Transpose output to match ReduceSum output shape.
    // The MatMul output [batchSize, M, N] is expanded back to (rank-1) dims in effective space,
    // then an inverse Transpose restores the original dim order (excluding K which was reduced).
    mlir::Value outValue = matmulOp.getOutput();
    const auto outPerm = buildPerm(effectiveRhsBroadcastDim, effectiveLhsBroadcastDim);
    const SmallVector<int64_t> outPermI64(outPerm.begin(), outPerm.end());
    const bool anyTransposed = !mlir::isIdentityPermutation(outPermI64);
    if (anyTransposed) {
        // Permuted shape: [batchDim sizes in batchDims order, M, N]
        SmallVector<int64_t> permutedShape;
        for (auto d : batchDims) {
            permutedShape.push_back(effectiveOutShapeVec[d]);
        }
        permutedShape.push_back(M);
        permutedShape.push_back(N);
        outValue = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "matmul_out_expand"), outValue,
                                                  getIntArrayAttr(ctx, permutedShape))
                           .getOutput();

        SmallVector<unsigned> currentOrder;
        currentOrder.reserve(batchDims.size() + 2);
        for (const auto d : batchDims) {
            currentOrder.push_back(checked_cast<unsigned>(d));
        }
        currentOrder.push_back(checked_cast<unsigned>(effectiveRhsBroadcastDim));
        currentOrder.push_back(checked_cast<unsigned>(effectiveLhsBroadcastDim));

        const auto currentOrderMap = mlir::AffineMap::getPermutationMap(currentOrder, ctx);
        const auto invPermMap = mlir::inversePermutation(currentOrderMap);

        if (!invPermMap.isIdentity()) {
            const auto invPermAttr = mlir::AffineMapAttr::get(invPermMap);
            outValue = rewriter.create<IE::TransposeOp>(appendLoc(origLoc, "matmul_out_transpose"), outValue, nullptr,
                                                        invPermAttr)
                               .getOutput();
        }
    }

    // Final reshape to the exact ReduceSum output shape (restores keep_dims size-1 if needed).
    const auto reducedShape = getShape(origOp.getOutput());
    auto outReshaped = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "matmul_out_reshape"), outValue,
                                                      getIntArrayAttr(ctx, reducedShape));

    rewriter.replaceOp(origOp, outReshaped.getOutput());

    _log.trace("[{0}] Replaced broadcast Multiply + ReduceSum with MatMul at '{1}'", getDebugName(), origLoc);
    return mlir::success();
}

//
// FuseOpsToMatMulPass
//

class FuseOpsToMatMulPass final : public IE::impl::FuseOpsToMatMulBase<FuseOpsToMatMulPass> {
public:
    explicit FuseOpsToMatMulPass(const bool enableGroupedMatMul, Logger log)
            : _enableGroupedMatMul(enableGroupedMatMul) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableGroupedMatMul;
};

mlir::LogicalResult FuseOpsToMatMulPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableGroupedMatMul.hasValue()) {
        _enableGroupedMatMul = enableGroupedMatMul;
    }
    return mlir::success();
}

void FuseOpsToMatMulPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BroadcastMultiplyReduceSumToMatMulRewriter>(&ctx, _log, _enableGroupedMatMul);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseOpsToMatMulPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOpsToMatMulPass(const bool enableGroupedMatMul, Logger log) {
    return std::make_unique<FuseOpsToMatMulPass>(enableGroupedMatMul, log);
}
