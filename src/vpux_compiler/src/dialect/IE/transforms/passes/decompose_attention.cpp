//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transpose_op_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEATTENTION
#define GEN_PASS_DEF_DECOMPOSEATTENTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Legal Attention configurations: {qHeadSize, tSL, sSL}
struct AttentionConfig {
    int64_t qHeadSize;
    int64_t tSL;
    int64_t sSL;
};

static const SmallVector<AttentionConfig> LEGAL_ATTENTION_CONFIGS = {
        // {qHeadSize, tSL, sSL}
        {192, 225, 225},  {12, 3600, 3600}, {8, 300, 300},  {16, 577, 577}, {12, 577, 577},
        {10, 1024, 1024}, {10, 1024, 77},   {20, 256, 256}, {20, 256, 77},  {6, 3072, 3072},
        {6, 151, 151},    {12, 512, 512},   {16, 256, 256}, {6, 2752, 2752}};

bool isLegalAttention(IE::AttentionOp op) {
    auto inputQ = op.getInputQ();
    auto inputK = op.getInputK();

    auto tensorTypeQ = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto tensorTypeK = mlir::cast<NDTypeInterface>(inputK.getType());

    const auto rankQ = tensorTypeQ.getRank();
    const auto rankK = tensorTypeK.getRank();

    if (rankQ < 3 || rankQ > 4) {
        return false;
    }
    if (rankK < 3 || rankK > 4) {
        return false;
    }

    // Boolean mask (i8 signless) is not yet supported in legal SDPA configs
    auto inputMask = op.getInputMask();
    if (inputMask) {
        auto tensorTypeMask = mlir::cast<NDTypeInterface>(inputMask.getType());
        auto maskElemType = tensorTypeMask.getElementType();
        if (maskElemType.isSignlessInteger(8)) {
            return false;
        }
    }

    auto shapeQ = tensorTypeQ.getShape().raw();
    auto shapeK = tensorTypeK.getShape().raw();

    const auto qHeadSize = (rankQ == 3) ? shapeQ[0] : shapeQ[0] * shapeQ[1];
    const auto tSL = shapeQ[rankQ - 2];
    const auto sSL = shapeK[rankK - 2];

    if (op.getInputSink() != nullptr && tSL == 1024) {
        return true;
    }

    auto matches = [&](const AttentionConfig& config) {
        return (config.qHeadSize == qHeadSize) && (config.sSL == sSL) && (config.tSL == tSL);
    };

    if (llvm::any_of(LEGAL_ATTENTION_CONFIGS, matches)) {
        return true;
    }

    const auto kvHeadSize = (rankK == 3) ? shapeK[0] : shapeK[0] * shapeK[1];
    if (qHeadSize > kvHeadSize && kvHeadSize == 1) {
        return VPU::AttentionOp::isSupported(op);
    }

    return false;
}

enum class ScalePlacement { NONE, OnQuery, OnKey, OnResult };

ScalePlacement determineScalePlacement(int64_t L, int64_t S, int64_t E) {
    const int64_t costOnQuery = L * E;
    const int64_t costOnKey = S * E;
    const int64_t costOnResult = L * S;

    if (costOnQuery <= costOnKey && costOnQuery <= costOnResult) {
        return ScalePlacement::OnQuery;
    } else if (costOnKey <= costOnResult) {
        return ScalePlacement::OnKey;
    } else {
        return ScalePlacement::OnResult;
    }
}

//
// Broadcast inputs for MQA Configurations
// [N, 1, H, W] -> [N, targetHeads, H, W]
//
mlir::Value broadcastInputToHeads(mlir::OpBuilder& builder, mlir::Location loc, mlir::MLIRContext* ctx,
                                  mlir::Value input, int64_t targetHeads, StringRef suffix) {
    mlir::Value inputToBroadcast = input;

    if (auto transposeOp = input.getDefiningOp<IE::TransposeOp>()) {
        if (IE::isWHSwappingTranspose(transposeOp)) {
            inputToBroadcast = transposeOp.getInput();
        }
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(inputToBroadcast.getType());
    auto inputShape = inputType.getShape().raw();
    const auto rank = inputShape.size();
    SmallVector<int64_t> broadcastShape;
    if (rank == 4) {
        broadcastShape = {inputShape[0], targetHeads, inputShape[2], inputShape[3]};
    } else {
        broadcastShape = {targetHeads, inputShape[1], inputShape[2]};
    }

    // Create shape constant
    auto shapeStorageType =
            mlir::RankedTensorType::get({static_cast<int64_t>(broadcastShape.size())}, getSInt64Type(ctx));
    auto shapeConst = Const::createConst(builder, appendLoc(loc, suffix), shapeStorageType, ArrayRef(broadcastShape),
                                         [&](Const::ContentSetup& setup) {
                                             return setup.castElemType(getSInt32Type(ctx));
                                         });

    // Create broadcast operation
    return builder
            .create<IE::BroadcastOp>(appendLoc(loc, suffix), inputToBroadcast, shapeConst, nullptr,
                                     IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::BIDIRECTIONAL))
            .getOutput();
}

//
// Collect preprocessing operations that uniquely feed into Value
//
SmallVector<mlir::Operation*> collectUniquePreprocessingOps(mlir::Value value) {
    SmallVector<mlir::Operation*> opsToClone;

    mlir::Value current = value;
    while (auto definingOp = current.getDefiningOp()) {
        // Stop if we hit a constant
        if (mlir::isa<Const::DeclareOp>(definingOp)) {
            break;
        }

        // Check if this operation's result is used only once (by the current chain)
        if (!definingOp->getResult(0).hasOneUse()) {
            break;
        }

        // Add this operation to the list
        opsToClone.push_back(definingOp);
        current = definingOp->getOperand(0);
    }

    // Reverse to get execution order (input to output)
    std::reverse(opsToClone.begin(), opsToClone.end());

    return opsToClone;
}

//
// Clone preprocessing operations
//
mlir::Value clonePreprocessingOps(mlir::OpBuilder& builder, ArrayRef<mlir::Operation*> ops, mlir::Value initialInput) {
    if (ops.empty()) {
        return initialInput;
    }

    mlir::IRMapping mapping;
    // Map the initial input (the source before preprocessing chain)
    if (!ops.empty() && ops[0]->getNumOperands() > 0) {
        mapping.map(ops[0]->getOperand(0), initialInput);
    }

    mlir::Value currentOutput = initialInput;
    for (auto* op : ops) {
        auto* clonedOp = builder.clone(*op, mapping);
        currentOutput = clonedOp->getResult(0);
    }

    return currentOutput;
}

//
// Create a valid MatMul operator with automatic transpose_b handling.
// outWasTransposedB (optional): set to true when a WH-swapping Transpose on B
//   was peeled from the input. Callers should skip that Transpose when collecting
//   V preprocessing ops (use B's pre-transpose value as the collection root).
// outSquareRebuildTranspose (optional): when the matrix is square and a new
//   equivalent WH-swapping Transpose was re-created inside this function, the
//   caller receives that TransposeOp. Step 7 can reuse its orderAttr directly
//   instead of reconstructing the permutation map.
//
mlir::Value createMatMul(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value a, mlir::Value b, StringRef suffix,
                         bool* outWasTransposedB = nullptr, IE::TransposeOp* outSquareRebuildTranspose = nullptr) {
    bool wasTransposedB = false;
    if (auto transposeOp = b.getDefiningOp<IE::TransposeOp>()) {
        if (IE::isWHSwappingTranspose(transposeOp)) {
            b = transposeOp.getInput();
            wasTransposedB = true;
        }
    }

    const auto aType = mlir::cast<vpux::NDTypeInterface>(a.getType());
    const auto bType = mlir::cast<vpux::NDTypeInterface>(b.getType());

    const auto aShape = aType.getShape().raw();
    const auto bShape = bType.getShape().raw();
    VPUX_THROW_UNLESS(aShape.size() == bShape.size(), "MatMul inputs must have the same rank, got {0} and {1}",
                      aShape.size(), bShape.size());

    const auto rank = aShape.size();
    const auto widthA = aShape[rank - 1];
    const auto heightB = bShape[rank - 2];
    const auto widthB = bShape[rank - 1];

    if (widthB == heightB && wasTransposedB) {
        SmallVector<unsigned> transposeOrder;
        for (unsigned i = 0; i < rank - 2; i++) {
            transposeOrder.push_back(i);
        }
        transposeOrder.push_back(rank - 1);
        transposeOrder.push_back(rank - 2);

        auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(transposeOrder, builder.getContext()));
        auto rebuildTranspose = builder.create<IE::TransposeOp>(appendLoc(loc, formatv("{0}_transpose_b", suffix)), b,
                                                                nullptr, orderAttr);
        b = rebuildTranspose.getOutput();
        if (outSquareRebuildTranspose != nullptr) {
            *outSquareRebuildTranspose = rebuildTranspose;
        }
    }

    if (outWasTransposedB != nullptr) {
        *outWasTransposedB = wasTransposedB;
    }

    bool transposeB = (widthA != heightB) || (widthB == heightB);
    auto matmul = builder.create<IE::MatMulOp>(appendLoc(loc, suffix), a, b,
                                               /*transpose_a=*/false, /*transpose_b=*/transposeB);

    return matmul.getOutput();
}

void decomposeAttention(IE::AttentionOp origOp, Logger log) {
    log.trace("Got AttentionOp for decomposition - '{0}'", origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    const auto ctx = origOp.getContext();
    const auto loc = origOp.getLoc();

    auto query = origOp.getInputQ();
    auto key = origOp.getInputK();
    auto value = origOp.getInputV();
    auto attentionMask = origOp.getInputMask();
    auto scale = origOp.getInputScale();
    auto bias = origOp.getInputBias();

    const auto queryType = mlir::cast<vpux::NDTypeInterface>(query.getType());
    const auto queryShape = queryType.getShape();
    const auto rank = queryShape.size();

    const auto keyType = mlir::cast<vpux::NDTypeInterface>(key.getType());
    const auto keyShape = keyType.getShape();

    const auto valueType = mlir::cast<vpux::NDTypeInterface>(value.getType());
    const auto valueShape = valueType.getShape();

    const auto L = queryShape.raw()[rank - 2];
    const auto E = queryShape.raw()[rank - 1];
    const auto S = keyShape.raw()[rank - 2];

    // Detect MQA pattern: Q has multiple heads, K/V have 1 head
    const auto qHeads = (rank == 4) ? queryShape.raw()[1] : queryShape.raw()[0];
    const auto kHeads = (rank == 4) ? keyShape.raw()[1] : keyShape.raw()[0];
    const auto vHeads = (rank == 4) ? valueShape.raw()[1] : valueShape.raw()[0];
    const bool isMQA = (kHeads == 1 && qHeads > 1) || (vHeads == 1 && qHeads > 1);

    const auto scalePlacement = (scale != nullptr) ? determineScalePlacement(L, S, E) : ScalePlacement::NONE;

    // Use mlir::Value for query/key/value since they may be reassigned
    mlir::Value queryVal = query;
    mlir::Value keyVal = key;
    mlir::Value valueVal = value;

    // Handle MQA: Broadcast K and V from 1 head to qHeads
    if (isMQA) {
        // Broadcast K: [N, 1, S, E] -> [N, qHeads, S, E]
        if (kHeads == 1 && qHeads > 1) {
            keyVal = broadcastInputToHeads(builder, loc, ctx, keyVal, qHeads, "k_broadcast");
            log.trace("Broadcast K from 1 head to {0} heads", qHeads);
        }

        // Broadcast V: [N, 1, S, E] -> [N, qHeads, S, E]
        if (vHeads == 1 && qHeads > 1) {
            valueVal = broadcastInputToHeads(builder, loc, ctx, valueVal, qHeads, "v_broadcast");
            log.trace("Broadcast V from 1 head to {0} heads", qHeads);
        }
    }

    if (scalePlacement == ScalePlacement::OnQuery) {
        queryVal =
                builder.createOrFold<IE::MultiplyOp>(appendLoc(loc, "scale_query"), queryVal, scale,
                                                     IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied scale on Query (L={0}, S={1}, E={2})", L, S, E);
    } else if (scalePlacement == ScalePlacement::OnKey) {
        keyVal = builder.createOrFold<IE::MultiplyOp>(appendLoc(loc, "scale_key"), keyVal, scale,
                                                      IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied scale on Key (L={0}, S={1}, E={2})", L, S, E);
    }

    // Step 1: Compute Q * K^T using MatMul
    mlir::Value attentionScores = createMatMul(builder, loc, queryVal, keyVal, "qk_matmul");

    // Step 2: Apply scale on result if that's the optimal placement
    if (scalePlacement == ScalePlacement::OnResult) {
        attentionScores =
                builder.createOrFold<IE::MultiplyOp>(appendLoc(loc, "scale_multiply"), attentionScores, scale,
                                                     IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied scale on result (L={0}, S={1}, E={2})", L, S, E);
    }

    // Step 3: Add attention mask if provided
    if (attentionMask) {
        auto maskType = mlir::cast<vpux::NDTypeInterface>(attentionMask.getType());
        auto maskElemType = maskType.getElementType();

        if (maskElemType.isSignlessInteger(8)) {
            // Get the target element type from attention scores
            auto scoresType = mlir::cast<vpux::NDTypeInterface>(attentionScores.getType());
            auto targetElemType = mlir::RankedTensorType::get({}, scoresType.getElementType());

            // Create constant for "keep" position
            auto keepValueConst = Const::createConst(builder, appendLoc(loc, "mask_keep_value"), targetElemType,
                                                     llvm::ArrayRef<float>{0.0f});

            // Create constant for "mask" position
            auto maskValueConst = Const::createConst(builder, appendLoc(loc, "mask_mask_value"), targetElemType,
                                                     llvm::ArrayRef<float>{-std::numeric_limits<float>::infinity()});

            // If mask != 0 (true), use keepValue (0.0), else use maskValue (-INF)
            attentionMask = builder.create<IE::SelectOp>(
                                           appendLoc(loc, "mask_select"), attentionMask, keepValueConst, maskValueConst,
                                           IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY))
                                    .getOutput();
        }

        attentionScores =
                builder.createOrFold<IE::AddOp>(appendLoc(loc, "mask_add"), attentionScores, attentionMask,
                                                IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied attention mask");
    }

    // Step 4: Add bias if provided
    if (bias) {
        attentionScores =
                builder.createOrFold<IE::AddOp>(appendLoc(loc, "bias_add"), attentionScores, bias,
                                                IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
        log.trace("Applied bias");
    }

    // Step 5: Apply Softmax on the last dimension
    const auto softmaxAxisAttr = getIntAttr(ctx, static_cast<int64_t>(rank) - 1);
    auto softmaxOp =
            builder.create<IE::SoftMaxOp>(appendLoc(loc, "softmax"), attentionScores, softmaxAxisAttr, nullptr);
    log.trace("Applied softmax");

    // Step 6: Compute attention_output = softmax_output * V using MatMul.
    // Track whether createMatMul consumed a WH-swapping Transpose on V
    // (vTransposePeeled) and whether a square-matrix Transpose was rebuilt
    // internally (vSquareRebuildTranspose). Both guide Step 7.
    bool vTransposePeeled = false;
    IE::TransposeOp vSquareRebuildTranspose;
    mlir::Value outputMatMul = createMatMul(builder, loc, softmaxOp.getOutput(), valueVal, "output_matmul",
                                            &vTransposePeeled, &vSquareRebuildTranspose);

    origOp.getOutput().replaceAllUsesWith(outputMatMul);
    origOp.erase();

    // Step 7: Move V preprocessing operations right before the MatMul that uses V.
    //
    // When vTransposePeeled=true the WH-swapping Transpose on V was consumed by
    // createMatMul. Collect preprocessing ops from the pre-transpose value to
    // avoid re-cloning the Transpose with an incompatible shape.
    //
    // When vSquareRebuildTranspose is set, createMatMul additionally re-created
    // an equivalent WH-swapping Transpose on the square rawV. After cloning the
    // preprocessing chain we re-apply a Transpose — reusing the rebuilt op's
    // orderAttr — so the MatMul (transposeB=true) still computes softmax × rawV.
    mlir::Value vForPreprocessing = valueVal;
    if (vTransposePeeled) {
        if (auto transposeOp = valueVal.getDefiningOp<IE::TransposeOp>()) {
            if (IE::isWHSwappingTranspose(transposeOp)) {
                vForPreprocessing = transposeOp.getInput();
            }
        }
    }
    auto preprocessingOps = collectUniquePreprocessingOps(vForPreprocessing);
    if (!preprocessingOps.empty()) {
        auto matmulOp = outputMatMul.getDefiningOp<IE::MatMulOp>();
        if (matmulOp) {
            builder.setInsertionPoint(matmulOp);
            mlir::Value originalInput = preprocessingOps[0]->getOperand(0);
            mlir::Value newVInput = clonePreprocessingOps(builder, preprocessingOps, originalInput);
            // In the square-transpose-rebuilt case the MatMul relies on
            // transposeB=true. Clone the rebuilt Transpose with its input remapped
            // to newVInput so the computation remains softmax × newVInput.
            if (vSquareRebuildTranspose) {
                mlir::IRMapping mapping;
                mapping.map(vSquareRebuildTranspose.getInput(), newVInput);
                newVInput = mlir::cast<IE::TransposeOp>(builder.clone(*vSquareRebuildTranspose.getOperation(), mapping))
                                    .getOutput();
            }
            matmulOp.getInput2Mutable().assign(newVInput);

            // Delete old preprocessing operations that are no longer used
            for (auto it = preprocessingOps.rbegin(); it != preprocessingOps.rend(); ++it) {
                auto* op = *it;
                if (op->use_empty()) {
                    op->erase();
                }
            }
        }
    }

    log.trace("Successfully decomposed Attention operation");
}

//
// DecomposeAttentionPass
//

class DecomposeAttentionPass final : public IE::impl::DecomposeAttentionBase<DecomposeAttentionPass> {
public:
    explicit DecomposeAttentionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeAttentionPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](IE::AttentionOp origOp) {
        if (isLegalAttention(origOp)) {
            return;
        }
        decomposeAttention(origOp, _log);
    });
}

}  // namespace

//
// createDecomposeAttentionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeAttentionPass(Logger log) {
    return std::make_unique<DecomposeAttentionPass>(log);
}
