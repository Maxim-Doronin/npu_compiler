//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_unroll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"

#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_FULLUNROLLSCFLOOP
#define GEN_PASS_DEF_FULLUNROLLSCFLOOP
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

class ConvertSlice final : public mlir::OpInterfaceRewritePattern<mlir::OffsetSizeAndStrideOpInterface> {
public:
    ConvertSlice(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::OffsetSizeAndStrideOpInterface>(ctx, vpux::benefitLow), _log(log) {
        setDebugName("ConvertSlice");
    }

    mlir::LogicalResult matchAndRewrite(mlir::OffsetSizeAndStrideOpInterface convOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSlice::matchAndRewrite(mlir::OffsetSizeAndStrideOpInterface sliceOp,
                                                  mlir::PatternRewriter& rewriter) const {
    if (sliceOp->getParentOfType<mlir::scf::ForOp>() != nullptr ||
        sliceOp->getName().getDialectNamespace() != mlir::tensor::TensorDialect::getDialectNamespace()) {
        return mlir::failure();
    }

    auto offsets = mlir::getConstantIntValues(sliceOp.getMixedOffsets());
    auto sizes = mlir::getConstantIntValues(sliceOp.getMixedSizes());
    auto strides = mlir::getConstantIntValues(sliceOp.getMixedStrides());

    if (!offsets.has_value() || !sizes.has_value()) {
        return mlir::failure();
    }

    auto insertOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(sliceOp.getOperation());

    if (insertOp != nullptr) {
        auto destinations = insertOp.getDpsInits();
        if (destinations.size() > 1) {
            return mlir::failure();
        }

        auto destOperation = destinations.front().getDefiningOp();

        if (getShape(insertOp.getDpsInputOperand(0)->get()).isDynamic()) {
            return mlir::failure();
        }

        auto input = insertOp.getDpsInputOperand(0)->get();

        if (auto emptyOp = mlir::dyn_cast<mlir::tensor::EmptyOp>(destOperation)) {
            /*
                For situation when insert_slice writes directly into result buffer,

                result_buffer = tensor.empty
                operation = VPU.Operation
                slice = tensor.insert_slice operation into result_buffer[offsets][sizes]

                just replace insert slice to operation
            */
            rewriter.replaceOp(sliceOp, input);
        } else if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(destOperation)) {
            /*
                For situation when insert_slice writes its result to the Concat,
                extend the concat instead of the slice
                result_buffer = VPU.Concat operations [offsets0][sizes0]
                operation0 = VPU.Operation
                slice = tensor.insert_slice operation into result_buffer[offsets1][sizes1]

                ->

                operation0 = VPU.Operation
                VPU.Concat operations + operation0 [offsets0 + offsets1][sizes0 + sizes1]

            */
            auto inputs = to_small_vector(concatOp.getInputs());
            inputs.emplace_back(input);

            auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsets().value());
            concatOffsets.emplace_back(offsets.value());

            rewriter.replaceOpWithNewOp<VPU::ConcatOp>(sliceOp, inputs, /*per_axis=*/nullptr,
                                                       getIntArrayOfArray(rewriter.getContext(), concatOffsets));
        } else {
            /*
               For situation when insert_slice writes its result to operation
               substituted on the previous steps,
               create the concat instead
               result_buffer = VPU.Operation
               operation1 = VPU.Operation
               slice = tensor.insert_slice operation into result_buffer[offsets1][sizes1]

               ->

               operation0 = VPU.Operation
               operation1 = VPU.Operation
               VPU.Concat operation0 + operation1 [offsets][sizes]

           */
            SmallVector<mlir::Value> inputs;
            inputs.emplace_back(destinations.front());
            inputs.emplace_back(input);

            SmallVector<SmallVector<int64_t>> concatOffsets;
            concatOffsets.emplace_back(SmallVector<int64_t>(offsets.value().size(), 0));
            concatOffsets.emplace_back(offsets.value());

            rewriter.replaceOpWithNewOp<VPU::ConcatOp>(sliceOp, inputs, /*per_axis=*/nullptr,
                                                       getIntArrayOfArray(rewriter.getContext(), concatOffsets));
        }

    } else {
        /*
             extract_slice -> VPU.Slice
        */

        const auto notEqualOne = [](int64_t stride) {
            return stride != 1;
        };
        if (strides.has_value() && std::any_of(strides.value().begin(), strides.value().end(), notEqualOne)) {
            return mlir::failure();
        }

        auto source = sliceOp->getOperand(0);
        const auto oneShape = Shape(sizes.value().size(), 1);
        const auto tileInfo = TileInfo(ShapeRef(sizes.value()), ShapeRef(offsets.value()), oneShape);

        auto newSliceOp = VPU::makeTile(rewriter, sliceOp.getLoc(), source, tileInfo, "converted_from_extract_slice");
        rewriter.replaceOp(sliceOp, newSliceOp);
    }

    return mlir::success();
}

class SimplifyDynamicCast final : public mlir::OpRewritePattern<mlir::tensor::CastOp> {
public:
    SimplifyDynamicCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::tensor::CastOp>(ctx, vpux::benefitHigh), _log(log) {
        setDebugName("SimplifyDynamicCast");
    }

    mlir::LogicalResult matchAndRewrite(mlir::tensor::CastOp castOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    mlir::Operation* adaptSourceOp(mlir::scf::ForallOp loop, mlir::Value castInput, mlir::ShapedType resultType,
                                   mlir::PatternRewriter& rewriter, Logger log) const;
    mlir::Operation* adaptSourceOp(mlir::InferTypeOpInterface inferRetTypeIfOp, mlir::PatternRewriter& rewriter,
                                   Logger log) const;
};

mlir::Operation* SimplifyDynamicCast::adaptSourceOp(mlir::scf::ForallOp loop, mlir::Value castInput,
                                                    mlir::ShapedType resultType, mlir::PatternRewriter& rewriter,
                                                    Logger log) const {
    if (loop == nullptr) {
        return nullptr;
    }

    auto sharedOutputs = loop.getOutputsMutable();

    mlir::IRMapping mapper;

    int64_t resultIdx = -1;
    for (auto& outOperand : sharedOutputs) {
        auto result = loop.getTiedOpResult(&outOperand);
        if (castInput != result) {
            // current shared_out is not connected to the tensor.cast input
            continue;
        }

        auto loopOutBuffer = mlir::dyn_cast<mlir::tensor::EmptyOp>(outOperand.get().getDefiningOp());
        if (loopOutBuffer == nullptr) {
            // current shared_out is not provided by a tensor.empty op
            continue;
        }

        rewriter.setInsertionPoint(loop);
        auto newEmptyOp = rewriter.create<mlir::tensor::EmptyOp>(loopOutBuffer->getLoc(), resultType.getShape(),
                                                                 resultType.getElementType());
        mapper.map(outOperand.get(), newEmptyOp);
        resultIdx = result.getResultNumber();
        break;
    }

    if (resultIdx < 0) {
        log.trace("Cannot simplify dynamic cast after forall op: tensor.cast input is not tied to a tensor.empty op.");
        return nullptr;
    }

    auto newOp = rewriter.clone(*loop, mapper);
    auto newLoop = mlir::cast<mlir::scf::ForallOp>(newOp);

    rewriter.modifyOpInPlace(newLoop, [&]() {
        // update result type of loop
        auto currentResult = newLoop->getOpResult(resultIdx);
        currentResult.setType(resultType);
        // update type of the OpOperand tied to the above shared_out
        auto tiedOpOperand = newLoop.getTiedOpOperand(currentResult);
        tiedOpOperand->get().setType(resultType);

        auto* terminator = newLoop.getBody()->getTerminator();
        if (auto inParallelOp = mlir::dyn_cast_or_null<mlir::scf::InParallelOp>(terminator)) {
            auto parallelInsertSliceOps = inParallelOp.getOps<mlir::tensor::ParallelInsertSliceOp>();

            auto tiedBlockArg = newLoop.getTiedBlockArgument(tiedOpOperand);
            // find the tensor.parallel_insert_slice op which inserts into the current result
            auto insertOpIt =
                    llvm::find_if(parallelInsertSliceOps, [&](mlir::tensor::ParallelInsertSliceOp insertSliceOp) {
                        auto dst = insertSliceOp.getDest();
                        return dst == tiedBlockArg;
                    });

            if (insertOpIt != parallelInsertSliceOps.end()) {
                log.trace("Found insertOp.");
                auto insertOp = *insertOpIt;

                // update type of parallel_insert_slice's destination
                insertOp.getDestMutable().get().setType(resultType);

                auto origOutType = mlir::cast<NDTypeInterface>(insertOp->getOperand(0).getType());
                auto newShape = Shape(origOutType.getShape());
                auto newBounds = Shape(getBoundedShape(origOutType));
                auto dstShape = resultType.getShape();

                for (auto idx : irange(newShape.size())) {
                    if (origOutType.getShape()[Dim(idx)] != mlir::ShapedType::kDynamic) {
                        newShape[Dim(idx)] = dstShape[idx];
                        newBounds[Dim(idx)] = dstShape[idx];
                    }
                }

                // set new static sizes to reflect the updated shapes after dynamic cast propagation
                insertOp.setStaticSizes(newShape.raw());

                // if there is a tensor.cast op inside the loop body, which goes into the parallel_insert_slice,
                // it should also be updated with the new shapes in mind
                auto insertedSliceCast = insertOp->getOperand(0).getDefiningOp<mlir::tensor::CastOp>();
                if (insertedSliceCast != nullptr) {
                    rewriter.setInsertionPoint(insertedSliceCast);
                    mlir::Type newType = origOutType.changeShape(newShape);
                    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
                        const auto bounds = Bounds(newBounds.raw());
                        newType = boundedType.changeBounds(bounds);
                    }

                    log.trace("Replace tensor.cast found inside the forall loop; new cast output type: {0}", newType);
                    auto newCastOp = rewriter.create<mlir::tensor::CastOp>(
                            insertedSliceCast->getLoc(), mlir::TypeRange{newType}, insertedSliceCast->getOperands());

                    rewriter.replaceOp(insertedSliceCast, newCastOp);
                }
            }
        }
    });

    return newLoop;
}

mlir::Operation* SimplifyDynamicCast::adaptSourceOp(mlir::InferTypeOpInterface inferRetTypeIfOp,
                                                    mlir::PatternRewriter& rewriter, Logger log) const {
    if (inferRetTypeIfOp == nullptr) {
        return nullptr;
    }

    auto operands = inferRetTypeIfOp->getOperands();
    mlir::IRMapping mapper;

    for (auto operand : operands) {
        auto upperCast = operand.getDefiningOp<mlir::tensor::CastOp>();
        if (upperCast == nullptr) {
            continue;
        }
        log.trace("Found upper cast op.");

        auto upperSourceType = mlir::cast<mlir::ShapedType>(upperCast.getSource().getType());
        auto upperResultType = mlir::cast<mlir::ShapedType>(upperCast.getResult().getType());

        if (!upperSourceType.hasStaticShape() || upperResultType.hasStaticShape()) {
            log.trace("Cannot simplify dynamic cast: upper cast is not supported.");
            return nullptr;
        }

        mapper.map(operand, upperCast.getSource());
    }

    auto newOp = rewriter.clone(*inferRetTypeIfOp.getOperation(), mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    return newOp;
}

mlir::LogicalResult SimplifyDynamicCast::matchAndRewrite(mlir::tensor::CastOp castOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Simplify dynamic tensor.cast op @ {0}", castOp->getLoc());
    // check if cast is dynamic to static
    auto sourceType = mlir::cast<mlir::ShapedType>(castOp.getSource().getType());
    auto resultType = mlir::cast<mlir::ShapedType>(castOp.getResult().getType());

    const auto& nestedLog = _log.nest();

    if (sourceType.hasStaticShape() || !resultType.hasStaticShape()) {
        if (sourceType.hasStaticShape() && llvm::all_of(castOp.getResult().getUsers(), [&](mlir::Operation* user) {
                auto userType = mlir::cast<mlir::ShapedType>(user->getResult(0).getType());
                return userType.hasStaticShape();
            })) {
            rewriter.replaceAllUsesWith(castOp, castOp.getSource());
            return mlir::success();
        }
        return matchFailed(nestedLog, rewriter, castOp,
                           "Cannot simplify dynamic cast with static source shape and dynamic result shape.");
    }

    auto sourceOp = castOp.getSource().getDefiningOp();
    mlir::Operation* updatedOp = nullptr;
    if (auto inferRetTypeIfOp = mlir::dyn_cast_or_null<mlir::InferTypeOpInterface>(sourceOp)) {
        updatedOp = adaptSourceOp(inferRetTypeIfOp, rewriter, nestedLog);
    } else if (auto loopOp = mlir::dyn_cast_or_null<mlir::scf::ForallOp>(sourceOp)) {
        updatedOp = adaptSourceOp(loopOp, castOp.getSource(), resultType, rewriter, nestedLog);
    } else {
        return matchFailed(nestedLog, rewriter, castOp, "Cannot simplify dynamic cast with unsupported source op.");
    }

    if (updatedOp == nullptr) {
        return matchFailed(nestedLog, rewriter, castOp,
                           "Could not update source op in order to propagate the dynamic cast.");
    }

    rewriter.replaceOp(castOp, updatedOp->getResults());

    return mlir::success();
}

//
// FullUnrollSCFLoopPass
//
class FullUnrollSCFLoopPass final : public VPU::impl::FullUnrollSCFLoopBase<FullUnrollSCFLoopPass> {
public:
    explicit FullUnrollSCFLoopPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

static std::optional<int64_t> getConstantTripCount(mlir::scf::ForOp forOp) {
    std::optional<int64_t> lbCstOp = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ubCstOp = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
    if (!lbCstOp.has_value() || !ubCstOp.has_value() || !stepCstOp.has_value()) {
        return {};
    }

    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst > 0 && "expected positive loop bounds and step");
    return llvm::divideCeilSigned(ubCst - lbCst, stepCst);
}

void FullUnrollSCFLoopPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    // full unrolling is not applicable for host pipeline
    if (config::getCompilationMode(moduleOp) == config::CompilationMode::HostCompile) {
        return;
    }

    mlir::OpBuilder builder(moduleOp);

    SmallVector<mlir::scf::ForOp> loopVector;
    collectLoops(moduleOp.getOperation(), loopVector);

    if (loopVector.empty()) {
        _log.trace("No loops found. Skipping unroll scf pass");
        return;
    }

    // full unrolling of tiling loops
    for (auto& loop : loopVector) {
        const auto tripCountOpt = getConstantTripCount(loop);

        VPUX_THROW_WHEN(!tripCountOpt.has_value(),
                        "Full unrolling is not supported for loop {0} with dynamic trip count", loop);

        (void)mlir::loopUnrollByFactor(loop, tripCountOpt.value());
    }

    // canonicalization patterns
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    ctx.getLoadedDialect<mlir::arith::ArithDialect>()->getCanonicalizationPatterns(patterns);
    ctx.getLoadedDialect<mlir::affine::AffineDialect>()->getCanonicalizationPatterns(patterns);
    ctx.getLoadedDialect<mlir::tensor::TensorDialect>()->getCanonicalizationPatterns(patterns);
    mlir::scf::IfOp::getCanonicalizationPatterns(patterns, &ctx);
    patterns.add<ConvertSlice>(&ctx, _log);
    patterns.add<SimplifyDynamicCast>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(moduleOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    const auto inferShapeOps = [&]() {
        moduleOp->walk([&](mlir::Operation* operation) {
            if (operation->getNumResults() == 0 || !mlir::isa<mlir::InferTypeOpInterface>(operation)) {
                return;
            }
            auto type = mlir::dyn_cast<vpux::NDTypeInterface>(operation->getResult(0).getType());
            if (type != nullptr && type.getShape().isDynamic()) {
                vpux::inferReturnTypes(operation, vpux::InferShapedTypeMode::SHAPE);
            }
        });
    };

    inferShapeOps();

    restorePaddingAttribute(moduleOp, _log.nest());

    inferShapeOps();

    // operations' transformation
    patterns.clear();
    patterns.add<ConvertSlice>(&ctx, _log);
    patterns.add<SimplifyDynamicCast>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(moduleOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createUnrollSCFLoopPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFullUnrollSCFLoopPass(Logger log) {
    return std::make_unique<FullUnrollSCFLoopPass>(log);
}
