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
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSESDPAEXTENDED
#define GEN_PASS_DEF_DECOMPOSESDPAEXTENDED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Legal SDPA configurations: {headSize, tSL, sSL}
struct SDPAConfig {
    int64_t headSize;
    int64_t tSL;
    int64_t sSL;
};

static const SmallVector<SDPAConfig> LEGAL_SDPA_CONFIGS = {
        // {headSize, tSL, sSL}
        {192, 225, 225}, {12, 3600, 3600}, {8, 300, 300},   {16, 577, 577}, {10, 1024, 1024}, {10, 1024, 77},
        {20, 256, 256},  {20, 256, 77},    {6, 3072, 3072}, {6, 151, 151},  {12, 512, 512}};

bool isLegalSDPAExtended(IE::SDPAExtendedOp op) {
    auto inputQ = op.getInputQ();
    auto inputK = op.getInputK();

    auto tensorTypeQ = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto tensorTypeK = mlir::cast<NDTypeInterface>(inputK.getType());

    const auto rankQ = tensorTypeQ.getRank();
    const auto rankK = tensorTypeK.getRank();

    if (rankQ < 2 || rankQ > 4) {
        return false;
    }
    if (rankK < 2 || rankK > 4) {
        return false;
    }

    auto shapeQ = tensorTypeQ.getShape().raw();
    auto shapeK = tensorTypeK.getShape().raw();

    const auto headSize = (rankQ == 3) ? shapeQ[0] : shapeQ[0] * shapeQ[1];
    const auto tSL = shapeQ[rankQ - 2];
    const auto sSL = shapeK[rankK - 2];

    auto matches = [&](const SDPAConfig& config) {
        return (config.headSize == headSize) && (config.sSL == sSL) && (config.tSL == tSL);
    };

    return llvm::any_of(LEGAL_SDPA_CONFIGS, matches);
}

bool isTransposedOnLastTwoDims(mlir::Value value) {
    auto transposeOp = value.getDefiningOp<IE::TransposeOp>();
    if (!transposeOp) {
        return false;
    }

    auto orderAttr = transposeOp.getOrderValueAttr();
    if (!orderAttr) {
        return false;
    }

    auto orderMap = orderAttr.getValue();
    const auto rank = orderMap.getNumDims();
    if (rank < 2) {
        return false;
    }

    // Check if the last two dimensions are swapped
    // Expected pattern for transpose on last two dims: [..., rank-1, rank-2]
    SmallVector<unsigned> expectedOrder;
    for (unsigned i = 0; i < rank - 2; i++) {
        expectedOrder.push_back(i);
    }
    expectedOrder.push_back(rank - 1);
    expectedOrder.push_back(rank - 2);

    for (unsigned i = 0; i < rank; i++) {
        if (orderMap.getDimPosition(i) != expectedOrder[i]) {
            return false;
        }
    }

    return true;
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
// Create a valid MatMul operator with automatic transpose_b handling
//
mlir::Value createMatMul(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value a, mlir::Value b, StringRef suffix) {
    bool wasTransposedB = false;
    if (auto transposeOp = b.getDefiningOp<IE::TransposeOp>()) {
        if (isTransposedOnLastTwoDims(b)) {
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
        auto transposeOp = builder.create<IE::TransposeOp>(appendLoc(loc, formatv("{0}_transpose_b", suffix)), b,
                                                           nullptr, orderAttr);
        b = transposeOp.getOutput();
    }

    bool transposeB = (widthA != heightB) || (widthB == heightB);
    auto matmul = builder.create<IE::MatMulOp>(appendLoc(loc, suffix), a, b,
                                               /*transpose_a=*/false, /*transpose_b=*/transposeB);

    return matmul.getOutput();
}

void decomposeSDPAExtended(IE::SDPAExtendedOp origOp, Logger log) {
    log.trace("Got SDPAExtendedOp for decomposition - '{0}'", origOp->getLoc());

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

    const auto L = queryShape.raw()[rank - 2];
    const auto E = queryShape.raw()[rank - 1];
    const auto S = keyShape.raw()[rank - 2];

    const auto scalePlacement = (scale != nullptr) ? determineScalePlacement(L, S, E) : ScalePlacement::NONE;

    // Use mlir::Value for query/key since they may be reassigned
    mlir::Value queryVal = query;
    mlir::Value keyVal = key;

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

    // Step 3: Add attention mask if provided and not all zeros
    if (attentionMask) {
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

    // Step 6: Compute attention_output = softmax_output * V using MatMul
    mlir::Value outputMatMul = createMatMul(builder, loc, softmaxOp.getOutput(), value, "output_matmul");

    origOp.getOutput().replaceAllUsesWith(outputMatMul);
    origOp.erase();

    // Step 7: Move V preprocessing operations right before the MatMul that uses V
    auto preprocessingOps = collectUniquePreprocessingOps(value);
    if (!preprocessingOps.empty()) {
        auto matmulOp = outputMatMul.getDefiningOp<IE::MatMulOp>();
        if (matmulOp) {
            builder.setInsertionPoint(matmulOp);
            mlir::Value originalInput = preprocessingOps[0]->getOperand(0);
            mlir::Value newVInput = clonePreprocessingOps(builder, preprocessingOps, originalInput);
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

    log.trace("Successfully decomposed SDPAExtended operation");
}

//
// DecomposeSDPAExtendedPass
//

class DecomposeSDPAExtendedPass final : public IE::impl::DecomposeSDPAExtendedBase<DecomposeSDPAExtendedPass> {
public:
    explicit DecomposeSDPAExtendedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeSDPAExtendedPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](IE::SDPAExtendedOp origOp) {
        if (isLegalSDPAExtended(origOp)) {
            return;
        }
        decomposeSDPAExtended(origOp, _log);
    });
}

}  // namespace

//
// createDecomposeSDPAExtendedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeSDPAExtendedPass(Logger log) {
    return std::make_unique<DecomposeSDPAExtendedPass>(log);
}
