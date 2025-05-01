//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEMEMPERMUTETHROUGHELTWISE
#define GEN_PASS_DEF_PROPAGATEMEMPERMUTETHROUGHELTWISE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

mlir::Operation* getInputPermuteLikeOp(mlir::Value addInput) {
    auto parentOp = addInput.getDefiningOp();
    while (parentOp) {
        if (mlir::isa<IE::MemPermuteOp, IE::PermuteQuantizeOp>(parentOp)) {
            return parentOp;
        } else if (auto parentShapeCast = mlir::dyn_cast<IE::ShapeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentShapeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentShapeCast.getSource().getDefiningOp();
            continue;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

mlir::Operation* getEltwiseOp(mlir::Value permuteInput) {
    auto parentOp = permuteInput.getDefiningOp();
    while (parentOp) {
        if (mlir::isa<IE::AddOp, IE::MultiplyOp, IE::SubtractOp>(parentOp)) {
            return parentOp;
        } else if (auto parentQuantizeCast = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentQuantizeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentQuantizeCast.getInput().getDefiningOp();
            continue;
        } else if (auto parentShapeCast = mlir::dyn_cast<IE::ShapeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentShapeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentShapeCast.getSource().getDefiningOp();
            continue;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

bool isEltwiseOpWithLegalShapeCastNumber(mlir::Operation* eltwiseOp) {
    size_t shapeCastNumb = 0;
    for (auto input : eltwiseOp->getOperands()) {
        if (mlir::isa_and_nonnull<IE::ShapeCastOp>(input.getDefiningOp())) {
            shapeCastNumb++;
        }
    }

    VPUX_THROW_UNLESS(eltwiseOp->getNumResults() == 1, "Eltwise operation should have only one output");
    if (mlir::isa_and_nonnull<IE::ShapeCastOp>(*eltwiseOp->getResult(0).getUsers().begin())) {
        shapeCastNumb++;
    }

    return shapeCastNumb == 0 || shapeCastNumb == 3;
}

mlir::Value createNewEltwiseOp(mlir::Value input1, mlir::Value input2, mlir::Type outType, mlir::Operation* eltwiseOp,
                               mlir::PatternRewriter& rewriter) {
    mlir::IRMapping mapper;
    mapper.map(eltwiseOp->getOperands(), SmallVector<mlir::Value>{input1, input2});
    auto* newOp = rewriter.clone(*eltwiseOp, mapper);
    newOp->getResult(0).setType(outType);
    return newOp->getResult(0);
}

// Search for pattern
// IE.MemPermute / PermuteQuantize -> [IE.ShapeCast]|
//                                                  | -> IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] -> IE.MemPermute
// IE.MemPermute / PermuteQuantize -> [IE.ShapeCast]|
bool canBeFolded(IE::PermuteQuantizeOp permuteQuantizeOp, mlir::AffineMap memPerm, mlir::Type permuteOutputType) {
    const auto permuteQuantizeOutElemType =
            mlir::cast<vpux::NDTypeInterface>(permuteQuantizeOp.getOutput().getType()).getElementType();
    // Can fuse MemPermute with PermuteQuantization in case only permutation (no quantization) is performed by this
    // PermuteQuantization Op.
    if (mlir::isa<mlir::quant::QuantizedType>(permuteQuantizeOutElemType)) {
        return false;
    }

    auto prevMemPerm = permuteQuantizeOp.getMemPerm();
    auto newMemPerm = memPerm.compose(prevMemPerm);

    const auto permuteQuantizeOpInType = permuteQuantizeOp.getInput().getType();
    auto permuteQuantizeOpInElemType = mlir::cast<vpux::NDTypeInterface>(permuteQuantizeOpInType).getElementType();
    // For the case that permutations can be folded, PermuteQuantizeOpInType and permuteOutType are expected to be
    // the same, except elemType.
    if (permuteQuantizeOpInType !=
                mlir::cast<vpux::NDTypeInterface>(permuteOutputType).changeElemType(permuteQuantizeOpInElemType) ||
        !newMemPerm.isIdentity()) {
        return false;
    }

    return true;
}

bool canBeFusedIntoPermuteCast(IE::PermuteQuantizeOp permuteQuantizeOp, mlir::AffineMap memPerm) {
    const auto inOrder = DimsOrder::fromValue(permuteQuantizeOp.getInput());
    const auto inShape = getShape(permuteQuantizeOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    auto prevMemPerm = permuteQuantizeOp.getMemPerm();
    auto composedMemPerm = memPerm.compose(prevMemPerm);

    return isTrivialPermute(inMemShape, composedMemPerm);
}

bool isSupportedMemPermute(mlir::AffineMap memPerm, mlir::Type permuteOutType, mlir::Operation* eltwiseOp, Logger log) {
    const SmallVector<mlir::Value> branches = eltwiseOp->getOperands();
    for (const auto& addInput : branches) {
        const auto inPermutationOp = getInputPermuteLikeOp(addInput);
        if (inPermutationOp != nullptr) {
            // Further checking for inPermuteQuantizeOp - propagate if PermuteQuantize and MemPermute can be folded.
            auto inPermuteQuantizeOp = mlir::dyn_cast<IE::PermuteQuantizeOp>(inPermutationOp);
            if (inPermuteQuantizeOp != nullptr && !canBeFolded(inPermuteQuantizeOp, memPerm, permuteOutType) &&
                !canBeFusedIntoPermuteCast(inPermuteQuantizeOp, memPerm)) {
                log.trace("IE::PermuteQuantize op: {0} and MemPerm: {1} can not be folded or fused into "
                          "permuteCast",
                          inPermuteQuantizeOp.getLoc(), memPerm);
                return false;
            }
        }
    }

    if ((getInputPermuteLikeOp(branches[0]) != nullptr) || (getInputPermuteLikeOp(branches[1]) != nullptr)) {
        // As long as one of the two inputs has PermuteLikeOp, the MemPermute should be propagated.
        // If one of the branches keeps a MemPermute, such MemPermute may be optimized in later passes.
        log.trace("MemPerm: {0} can be propagated", memPerm);
        return true;
    }

    return false;
}

bool isSupportedMemPermute(IE::MemPermuteOp memPermuteOp, mlir::Operation* eltwiseOp, Logger log) {
    if (!eltwiseOp->getResult(0).hasOneUse()) {
        log.trace("Eltwise operation has more than one consumer");
        return false;
    }
    return isSupportedMemPermute(memPermuteOp.getMemPerm(), memPermuteOp.getType(), eltwiseOp, log);
}

std::optional<SmallVector<Shape>> getNewAlignedShapeForPermuteCast(mlir::Operation* eltwiseOp,
                                                                   IE::MemPermuteOp memPermuteOp) {
    SmallVector<Shape> newAlignedShape;
    int64_t alignment = 1;
    if (auto alignedChannelsInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(eltwiseOp)) {
        alignment = alignedChannelsInterface.getInputChannelAlignment();
    }

    // If we want to go through OptimizeShapeCastedEltwise rewriter, since it don't have shape cast, we need to make
    // sure the lowest mem dim is divisible by the alignment, the assumption is like for NCE Add, lowest mem Dim is
    // divisible by 16. Here we propagate the output memPermute to the input, and check the new input memshape.
    for (auto operand : eltwiseOp->getOperands()) {
        auto shapeCastOp = operand.getDefiningOp<IE::ShapeCastOp>();
        auto memPermuteInput = (shapeCastOp == nullptr) ? operand : shapeCastOp.getSource();
        auto outType = mlir::cast<vpux::NDTypeInterface>(memPermuteInput.getType());
        auto memShape = outType.getMemShape();
        auto newInEltwiseMemShape = applyPerm(memShape, memPermuteOp.getMemPerm());
        if (newInEltwiseMemShape.back() % alignment != 0) {
            return std::nullopt;
        }
        auto dimsOrder = DimsOrder::fromValue(memPermuteInput);
        newAlignedShape.push_back(dimsOrder.toLogicalOrder(newInEltwiseMemShape));
    }

    auto memPermOutType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getResult().getType());
    auto memPermOutMemShape = memPermOutType.getMemShape();
    if (memPermOutMemShape.back() % alignment != 0) {
        return std::nullopt;
    }
    auto outOrder = DimsOrder::fromValue(eltwiseOp->getResult(0));
    newAlignedShape.push_back(outOrder.toLogicalOrder(memPermOutMemShape));

    return newAlignedShape;
}

mlir::Value processNonPermuteBranch(mlir::PatternRewriter& rewriter, IE::MemPermuteOp memPermuteOp, mlir::Value input,
                                    int64_t idx, std::optional<SmallVector<Shape>> newAlignedShape) {
    // For the branch without PermuteLike op like
    //       IE.Tile -> [IE.ShapeCast]|
    //                                | -> IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] ->IE.MemPermute
    // IE.MemPermute -> [IE.ShapeCast]|
    // If the new mem shape after permutation meets alignment requirement, the pattern will be converted into:
    //      IE.Tile -> IE.MemPermute -> IE.PermuteCast |
    //                                                 | -> IE.Add -> IE.PermuteCast -> [IE.QuantizeCast]
    // IE.MemPermute -> IE.MemPermute -> IE.PermuteCast|
    // else the ShapeCast is still needed, and the pattern will be converted into:
    // IE.Tile -> IE.MemPermute -> [IE.LayoutCast] -> [IE.ShapeCast]|
    //                                                              | -> IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast]
    //         IE.PermuteQuantize -> IE.MemPermute -> [IE.ShapeCast]|
    // For the branch IE.Tile -> IE.MemPermute -> [IE.ShapeCast], the MemPermute will be propagated to the front of the
    // tile op in the later pass, like IE.MemPermute -> IE.Tile -> [IE.ShapeCast], this MemPermute may be a trivial
    // permute or permute on a smaller tensor.
    const auto addInOrder = DimsOrder::fromValue(input);
    const auto orderInAttr = mlir::AffineMapAttr::get(addInOrder.toAffineMap(memPermuteOp.getContext()));
    auto shapeCastOp = input.getDefiningOp<IE::ShapeCastOp>();

    auto memPermuteInput = (shapeCastOp == nullptr) ? input : shapeCastOp.getSource();
    const auto newMemPermuteLoc = appendLoc(memPermuteOp.getLoc(), "_mem_permute_{0}", idx);
    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(newMemPermuteLoc, memPermuteInput,
                                                             memPermuteOp.getDstOrder(), memPermuteOp.getMemPerm());
    if (newAlignedShape.has_value()) {
        auto ctx = newMemPermuteOp->getContext();
        auto dimOrder = DimsOrder::fromValue(input);
        auto permuteCastOp = rewriter.createOrFold<IE::PermuteCastOp>(
                memPermuteOp.getLoc(), newMemPermuteOp.getResult(), dimOrder.toAffineMap(ctx),
                mlir::AffineMap::getMultiDimIdentityMap(dimOrder.numDims(), ctx));
        return permuteCastOp;
    }
    const auto newLayoutCastLoc = appendLoc(memPermuteOp.getLoc(), "_in_layout_cast_{0}", idx);
    auto newLayoutCastOp =
            rewriter.create<IE::LayoutCastOp>(newLayoutCastLoc, newMemPermuteOp.getOutput(), orderInAttr);
    const auto newShapeCastLoc = appendLoc(memPermuteOp.getLoc(), "_in_shape_cast_{0}", idx);
    const auto addInShape = getShape(input).toValues();
    const auto addInShapeAttr = getIntArrayAttr(memPermuteOp.getContext(), addInShape.raw());
    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(newShapeCastLoc, newLayoutCastOp.getOutput(), addInShapeAttr);
    return newShapeCastOp.getResult();
}

//
// OptimizeEltwise
//

class OptimizeEltwise final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    OptimizeEltwise(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("OptimizeEltwise");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Propagate last permute in the chain IE.MemPermute -> IE.ShapeCast -> IE.Add -> IE.ShapeCast -> IE.MemPermute
// This subgraph becomes IE.MemPermute -> IE.MemPermute -> IE.ShapeCast -> IE.Add -> IE.ShapeCast
// Two consecutive IE.MemPermute operations will be folded into one.
// VPU.NCE.Eltwise is layout agnostic, however, DPU operates on NHWC layouts. Layout casts must be applied.
// IE.LayoutCast (to NCHW) -> IE.Add (NHWC input, NHWC output) -> IE.LayoutCast (to original)
mlir::LogicalResult OptimizeEltwise::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());

    auto ctx = memPermuteOp.getContext();
    auto quantizeCastOp = memPermuteOp.getInput().getDefiningOp<IE::QuantizeCastOp>();

    auto eltwiseOp = getEltwiseOp(memPermuteOp.getInput());
    if (eltwiseOp == nullptr) {
        return matchFailed(_log, rewriter, memPermuteOp,
                           "IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] -> IE.MemPermute pattern not found");
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(eltwiseOp->getResult(0).getType());
    const auto elemType = outType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        return matchFailed(_log, rewriter, memPermuteOp, "IE.Add has per axis quant type");
    }

    if (!isEltwiseOpWithLegalShapeCastNumber(eltwiseOp)) {
        return matchFailed(_log, rewriter, memPermuteOp,
                           "All inputs and outputs of Add Op must have both or no ShapeCast");
    }

    if (!isSupportedMemPermute(memPermuteOp, eltwiseOp, _log.nest())) {
        return matchFailed(_log, rewriter, memPermuteOp, "MemPermute is not supported");
    }

    const SmallVector<mlir::Value> branches = eltwiseOp->getOperands();

    auto newAlignedShape = getNewAlignedShapeForPermuteCast(eltwiseOp, memPermuteOp);
    if (newAlignedShape.has_value()) {
        return matchFailed(
                _log, rewriter, memPermuteOp,
                "New shape is channel aligned, pattern should be optimized by OptimizeShapeCastedEltwise rewriter");
    }

    SmallVector<mlir::Value> newAddInputs;

    for (size_t inputIdx = 0; inputIdx < branches.size(); inputIdx++) {
        auto branchInput = branches[inputIdx];

        if (getInputPermuteLikeOp(branchInput) == nullptr) {
            // Process branch without PermuteLike op.
            const auto newOutput = processNonPermuteBranch(rewriter, memPermuteOp, branchInput, inputIdx, std::nullopt);
            newAddInputs.push_back(newOutput);
            continue;
        }

        const auto inPermutationOp = getInputPermuteLikeOp(branchInput);

        const auto newMemPermuteLoc = appendLoc(memPermuteOp.getLoc(), "_mem_permute_{0}", inputIdx);
        auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(newMemPermuteLoc, inPermutationOp->getResult(0),
                                                                 memPermuteOp.getDstOrder(), memPermuteOp.getMemPerm());

        const auto addInShape = getShape(branchInput).toValues();
        const auto addInShapeAttr = getIntArrayAttr(ctx, addInShape.raw());
        const auto origAddInType = mlir::cast<vpux::NDTypeInterface>(branchInput.getType());
        const auto newShapeCastOrder = DimsOrder::fromValue(newMemPermuteOp.getOutput());
        const auto newShapeCastType = origAddInType.changeDimsOrder(newShapeCastOrder);
        auto newShapeCastOp =
                rewriter.create<IE::ShapeCastOp>(memPermuteOp.getLoc(), newShapeCastType.changeShape(addInShape),
                                                 newMemPermuteOp.getOutput(), addInShapeAttr);

        const auto addInOrder = DimsOrder::fromValue(branchInput);
        const auto orderInAttr = mlir::AffineMapAttr::get(addInOrder.toAffineMap(ctx));
        const auto inLayoutCastLoc = appendLoc(memPermuteOp.getLoc(), "_in_layout_cast_{0}", inputIdx);
        auto inLayoutCastOp =
                rewriter.create<IE::LayoutCastOp>(inLayoutCastLoc, newShapeCastOp.getResult(), orderInAttr);

        newAddInputs.push_back(inLayoutCastOp.getOutput());
    }
    auto newEltwiseOp = createNewEltwiseOp(newAddInputs[0], newAddInputs[1], outType, eltwiseOp, rewriter);

    const auto nceOutLayout = DimsOrder::fromValue(memPermuteOp.getOutput());
    const auto orderOutAttr = mlir::AffineMapAttr::get(nceOutLayout.toAffineMap(ctx));
    const auto outLayoutCastLoc = appendLoc(memPermuteOp.getLoc(), "_out_layout_cast");
    auto outLayoutCastOp = rewriter.create<IE::LayoutCastOp>(outLayoutCastLoc, newEltwiseOp, orderOutAttr);

    const auto newOutShapeCastType = memPermuteOp.getOutput().getType();
    const auto newOutShapeCastLoc = appendLoc(memPermuteOp.getLoc(), "_out_shape_cast");

    const Shape targetShape = getShape(memPermuteOp.getOutput()).toValues();
    const auto targetShapeAttr = getIntArrayAttr(ctx, targetShape.raw());
    IE::ShapeCastOp newOutShapeCastOp;
    if (quantizeCastOp != nullptr) {
        const auto quantizeCastInElemType =
                mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getInput().getType()).getElementType();
        newOutShapeCastOp = rewriter.create<IE::ShapeCastOp>(
                newOutShapeCastLoc,
                mlir::cast<vpux::NDTypeInterface>(newOutShapeCastType).changeElemType(quantizeCastInElemType),
                outLayoutCastOp.getOutput(), targetShapeAttr);
        auto newQuantizeCastOp = rewriter.create<IE::QuantizeCastOp>(
                quantizeCastOp.getLoc(), newOutShapeCastOp.getResult(), quantizeCastOp.getDstElemTypeAttr());
        rewriter.replaceOp(memPermuteOp, newQuantizeCastOp.getOutput());
    } else {
        newOutShapeCastOp = rewriter.create<IE::ShapeCastOp>(newOutShapeCastLoc, newOutShapeCastType,
                                                             outLayoutCastOp.getOutput(), targetShapeAttr);
        rewriter.replaceOp(memPermuteOp, newOutShapeCastOp.getResult());
    }

    return mlir::success();
}

//
// OptimizeShapeCastedEltwise
//

class OptimizeShapeCastedEltwise final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    OptimizeShapeCastedEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("OptimizeShapeCastedEltwise");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

    mlir::Value createNewInputWithAlignedShape(IE::MemPermuteOp newMemPermuteInput, mlir::Operation* eltwiseOp,
                                               ShapeRef newAlignedShape, mlir::PatternRewriter& rewriter) const;
    void createNewOutputWithAlignedShape(IE::MemPermuteOp memPermuteOp, mlir::Operation* eltwiseOp,
                                         ShapeRef newAlignedShape, ArrayRef<mlir::Value> newInputs,
                                         mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

// Propagate last permute in the chain:
//     IE.MemPermute -> [IE.ShapeCast] -> IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] -> IE.MemPermute
// which will be optimized as:
//     IE.MemPermute -> IE.MemPermute -> IE.Add -> [IE.QuantizeCast]
// ShapeCast is removed since the new input memshape has meet alignment requirement.
mlir::LogicalResult OptimizeShapeCastedEltwise::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());

    auto eltwiseOp = getEltwiseOp(memPermuteOp.getInput());
    if (eltwiseOp == nullptr) {
        return matchFailed(_log, rewriter, memPermuteOp,
                           "IE.Add -> IE.ShapeCast -> [IE.QuantizeCast] -> IE.MemPermute pattern not found");
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(eltwiseOp->getResult(0).getType());
    const auto elemType = outType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        return matchFailed(_log, rewriter, memPermuteOp, "IE.Add has per axis quant type");
    }

    if (!isEltwiseOpWithLegalShapeCastNumber(eltwiseOp)) {
        return matchFailed(_log, rewriter, memPermuteOp, "All inputs and outputs of Add Op must have ShapeCast");
    }

    if (!isSupportedMemPermute(memPermuteOp, eltwiseOp, _log.nest())) {
        return matchFailed(_log, rewriter, memPermuteOp, "MemPermute is not supported");
    }

    const SmallVector<mlir::Value> branches = eltwiseOp->getOperands();

    auto newAlignedShape = getNewAlignedShapeForPermuteCast(eltwiseOp, memPermuteOp);
    if (!newAlignedShape.has_value()) {
        return matchFailed(_log, rewriter, memPermuteOp, "The shape is not channel aligned");
    }

    SmallVector<mlir::Value> newAddInputs;

    for (size_t inputIdx = 0; inputIdx < branches.size(); inputIdx++) {
        auto branchInput = branches[inputIdx];

        if (getInputPermuteLikeOp(branchInput) == nullptr) {
            // Process branch without PermuteLike op.
            const auto newOutput =
                    processNonPermuteBranch(rewriter, memPermuteOp, branchInput, inputIdx, newAlignedShape);
            newAddInputs.push_back(newOutput);
            continue;
        }

        const auto inPermutationOp = getInputPermuteLikeOp(branchInput);

        const auto newMemPermuteLoc = appendLoc(memPermuteOp.getLoc(), "_mem_permute_{0}", inputIdx);
        auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(newMemPermuteLoc, inPermutationOp->getResult(0),
                                                                 memPermuteOp.getDstOrder(), memPermuteOp.getMemPerm());

        // the newMemPermuteOp's output mem shape has meet alignment requirement, so for the original pattern:
        //     IE.MemPermute -> IE.ShapeCast -> IE.Add -> IE.ShapeCast -> IE.MemPermute
        // the ShapeCast input will be replaced with PermuteCast:
        //     IE.MemPermute -> IE.MemPermute -> IE.PermuteCast -> IE.Add -> ...
        auto newInput =
                createNewInputWithAlignedShape(newMemPermuteOp, eltwiseOp, newAlignedShape.value()[inputIdx], rewriter);
        newAddInputs.push_back(newInput);
    }
    createNewOutputWithAlignedShape(memPermuteOp, eltwiseOp, newAlignedShape.value().back(), newAddInputs, rewriter);
    return mlir::success();
}

mlir::Value OptimizeShapeCastedEltwise::createNewInputWithAlignedShape(IE::MemPermuteOp newMemPermuteInput,
                                                                       mlir::Operation* eltwiseOp,
                                                                       ShapeRef newAlignedShape,
                                                                       mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto dimOrder = DimsOrder::fromValue(eltwiseOp->getOperand(0));
    auto newInType = mlir::cast<vpux::NDTypeInterface>(newMemPermuteInput.getResult().getType())
                             .changeShape(newAlignedShape)
                             .changeDimsOrder(dimOrder);
    return rewriter.createOrFold<IE::PermuteCastOp>(newMemPermuteInput.getLoc(), newInType,
                                                    newMemPermuteInput.getResult(), dimOrder.toAffineMap(ctx),
                                                    mlir::AffineMap::getMultiDimIdentityMap(dimOrder.numDims(), ctx));
}

void OptimizeShapeCastedEltwise::createNewOutputWithAlignedShape(IE::MemPermuteOp memPermuteOp,
                                                                 mlir::Operation* eltwiseOp, ShapeRef newAlignedShape,
                                                                 ArrayRef<mlir::Value> newInputs,
                                                                 mlir::PatternRewriter& rewriter) const {
    // the newMemPermuteOp's output mem shape has meet alignment requirement, so for the original pattern:
    //     IE.MemPermute -> IE.ShapeCast -> IE.Add -> IE.ShapeCast -> [IE.QuantCast] -> IE.MemPermute
    // the ShapeCast output will be replaced with PermuteCast:
    //     IE.MemPermute -> IE.MemPermute -> IE.PermuteCast -> IE.Add -> IE.PermuteCast -> [IE.QuantCast]
    auto ctx = memPermuteOp->getContext();
    auto outType = mlir::cast<vpux::NDTypeInterface>(eltwiseOp->getResult(0).getType());
    auto newEltwiseOpType = outType.changeShape(newAlignedShape);
    auto newEltwizeOp = createNewEltwiseOp(newInputs[0], newInputs[1], newEltwiseOpType, eltwiseOp, rewriter);

    auto memPermOutType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getResult().getType());
    auto newOutPermuteCastType = memPermOutType.changeElemType(newEltwiseOpType.getElementType());

    auto dstOrder = memPermuteOp.getDstOrder();
    auto memPerm = mlir::AffineMap::getMultiDimIdentityMap(dstOrder.getNumDims(), ctx);
    auto outCastOp = rewriter.create<IE::PermuteCastOp>(memPermuteOp->getLoc(), newOutPermuteCastType, newEltwizeOp,
                                                        dstOrder, memPerm);

    auto quantizeCastOp = memPermuteOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
    if (quantizeCastOp == nullptr) {
        rewriter.replaceOp(memPermuteOp, outCastOp);
        return;
    }

    auto newQuantizeCastOp = rewriter.create<IE::QuantizeCastOp>(quantizeCastOp.getLoc(), outCastOp.getResult(),
                                                                 quantizeCastOp.getDstElemTypeAttr());
    rewriter.replaceOp(memPermuteOp, newQuantizeCastOp.getOutput());
}

//
// SwapMemPermuteWithSoftmax
//

class SwapMemPermuteWithSoftmax final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    SwapMemPermuteWithSoftmax(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("SwapMemPermuteWithSoftmax");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SwapMemPermuteWithSoftmax::matchAndRewrite(IE::MemPermuteOp memPermuteOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), memPermuteOp->getName(), memPermuteOp->getLoc());

    auto softmaxOp = memPermuteOp.getInput().getDefiningOp<IE::SoftMaxOp>();
    if (softmaxOp == nullptr) {
        return matchFailed(_log, rewriter, memPermuteOp, "No parent softmaxOp found");
    }

    if (!softmaxOp->hasOneUse()) {
        return matchFailed(_log, rewriter, memPermuteOp, "Parent softmaxOp has multiple uses");
    }

    auto eltwiseOp = getEltwiseOp(softmaxOp.getInput());
    if (eltwiseOp == nullptr || !isSupportedMemPermute(memPermuteOp, eltwiseOp, _log.nest())) {
        return matchFailed(
                _log, rewriter, memPermuteOp,
                "IE.Add -> [IE.ShapeCast] -> [IE.QuantizeCast] -> IE.SoftMax -> IE.MemPermute pattern not found");
    }

    auto memPerm = DimsOrder::fromAffineMap(memPermuteOp.getMemPerm());
    auto permuteOutOrder = DimsOrder::fromValue(memPermuteOp.getOutput());
    auto softmaxOrder = DimsOrder::fromValue(softmaxOp.getInput());

    auto softmaxAxisMemDim = softmaxOrder.toMemDim(Dim(softmaxOp.getAxisInd()));
    auto newSoftmaxAxisMemDim = MemDim(memPerm.dimPos(Dim(softmaxAxisMemDim.ind())));
    auto newSoftmaxAxisDim = permuteOutOrder.toDim(newSoftmaxAxisMemDim);

    auto newMemPermute = rewriter.create<IE::MemPermuteOp>(
            memPermuteOp.getLoc(), softmaxOp.getInput(), memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());
    auto newSoftmaxOp = rewriter.create<IE::SoftMaxOp>(softmaxOp.getLoc(), newMemPermute.getOutput(),
                                                       getIntAttr(getContext(), newSoftmaxAxisDim.ind()),
                                                       softmaxOp.getPadSizeAttr());

    rewriter.replaceOp(memPermuteOp, newSoftmaxOp.getOutput());

    return mlir::success();
}

//
// ExtractODUPermuteFromAdd
//

class ExtractODUPermuteFromAdd final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    ExtractODUPermuteFromAdd(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
        this->setDebugName("ExtractODUPermuteFromAdd");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/*
For subgraph:

// IE.MemPermute / PermuteQuantize -> |
//                                    | -> IE.Add with ODU Permute ->
// IE.MemPermute / PermuteQuantize -> |

convert to:


// IE.MemPermute / PermuteQuantize -> |
//                                    | -> IE.Add ->  IE.MemPermute ->
// IE.MemPermute / PermuteQuantize -> |

and the converted pattern could be further optimized by other rewriters

*/

mlir::LogicalResult ExtractODUPermuteFromAdd::matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), addOp->getName(), addOp->getLoc());

    auto outType = addOp.getType();
    auto elemType = outType.getElementType();

    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        return matchFailed(_log, rewriter, addOp, "IE.Add has per axis quant type");
    }

    const auto outDimOrder = DimsOrder::fromValue(addOp.getResult());
    if (outDimOrder.numDims() != 4 || outDimOrder == DimsOrder::NHWC) {
        return matchFailed(_log, rewriter, addOp, "IE.Add doesn't have ODU permute");
    }

    auto hasInputShapeCast = llvm::any_of(addOp->getOperands(), [](const auto& operand) {
        return mlir::isa_and_nonnull<IE::ShapeCastOp>(operand.getDefiningOp());
    });

    if (hasInputShapeCast) {
        return matchFailed(_log, rewriter, addOp, "Inputs have ShapeCast");
    }

    auto ctx = addOp.getContext();
    auto inDimOrder = DimsOrder::fromValue(addOp.getInput1());
    const auto memPerm = getPermutationFromOrders(inDimOrder, outDimOrder, ctx);

    if (!isSupportedMemPermute(memPerm, outType, addOp, _log.nest())) {
        return mlir::failure();
    }
    auto newType = mlir::cast<vpux::NDTypeInterface>(outType).changeDimsOrder(inDimOrder);
    addOp->getResult(0).setType(newType);
    auto memPermuteOp =
            rewriter.create<IE::MemPermuteOp>(addOp->getLoc(), addOp, outDimOrder.toAffineMap(ctx), memPerm);
    rewriter.replaceAllUsesExcept(addOp.getResult(), memPermuteOp.getResult(), memPermuteOp);
    return mlir::success();
}

//
// PropagateMemPermuteThroughEltwisePass
//

class PropagateMemPermuteThroughEltwisePass final :
        public IE::impl::PropagateMemPermuteThroughEltwiseBase<PropagateMemPermuteThroughEltwisePass> {
public:
    explicit PropagateMemPermuteThroughEltwisePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void PropagateMemPermuteThroughEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeEltwise>(&ctx, _log);
    patterns.add<OptimizeShapeCastedEltwise>(&ctx, _log);
    patterns.add<SwapMemPermuteWithSoftmax>(&ctx, _log);
    patterns.add<ExtractODUPermuteFromAdd>(&ctx, _log);
    IE::PermuteCastOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateMemPermuteThroughEltwisePass(Logger log) {
    return std::make_unique<PropagateMemPermuteThroughEltwisePass>(log);
}
