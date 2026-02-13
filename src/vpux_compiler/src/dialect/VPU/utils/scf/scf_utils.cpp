//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

#include <iterator>
#include <numeric>

using namespace vpux;
using namespace VPU;

mlir::LogicalResult vpux::VPU::getResultTileBounds(mlir::Operation* operation, unsigned resultNumber,
                                                   DimArrRef tilingDims, ArrayRef<mlir::OpFoldResult> sizes,
                                                   Bounds& resultBounds) {
    if (tilingDims.empty() || sizes.empty()) {
        return mlir::failure();
    }

    auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(resultNumber).getType());
    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType);
    if (boundedType == nullptr) {
        return mlir::failure();
    }

    resultBounds = boundedType.getBounds().toValues();

    mlir::tensor::ExtractSliceOp extractSliceOp = nullptr;
    Logger log = Logger::global().nest("result_tile_bounds");
    for (auto* user : operation->getUsers()) {
        if (!mlir::isa<mlir::tensor::ExtractSliceOp>(user)) {
            continue;
        }
        if (extractSliceOp != nullptr) {
            log.error("Multiple extract slices in users of op {0}", operation);
            return mlir::failure();
        }
        extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(user);
    }

    if (extractSliceOp == nullptr) {
        OpChainAnalysis affineUtils;

        for (auto dim : tilingDims) {
            llvm::DenseMap<mlir::Value, SmallVector<int64_t>> valueMap;
            auto sizeValue = affineUtils.getOpFoldResultValue(sizes[dim.ind()], valueMap);

            if (!sizeValue.has_value() || sizeValue.value().empty()) {
                continue;
            }
            resultBounds[dim] = sizeValue.value().front();
        }
        return mlir::success();
    }

    auto boundedSliceType = mlir::dyn_cast<Core::BoundedTensorType>(extractSliceOp.getType());
    if (boundedSliceType == nullptr) {
        return mlir::failure();
    }

    resultBounds = boundedSliceType.getBounds().toValues();
    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim) {
    const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());

    mlir::ReifiedRankedShapedTypeDims resultShape;
    if (mlir::failed(reifyResultShapes(builder, operation, resultShape))) {
        return builder.getIndexAttr(outputType.getDimSize(dim));
    }

    return resultShape[0][dim];
}

mlir::Value vpux::VPU::generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                                    const SCFTileInfo& inputTileInfo, SmallVector<mlir::Operation*>& generatedSlices) {
    auto origType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());

    auto staticNewShape = mlir::getConstantIntValues(inputTileInfo.shape);
    if (origType.getShape().isStatic() && staticNewShape.has_value() &&
        llvm::equal(origType.getShape().raw(), staticNewShape.value())) {
        return origInput;
    }

    SmallVector<mlir::OpFoldResult> defaultStrides(inputTileInfo.offsets.size(), builder.getIndexAttr(1));

    auto extractTile = builder.create<mlir::tensor::ExtractSliceOp>(
            appendLoc(loc, "extractSlice"), origInput, inputTileInfo.offsets, inputTileInfo.shape, defaultStrides);

    auto newShape = getShape(extractTile.getResult());

    /*For cases where we cannot evenly divide tensor with static shapes into pieces of equal size, SCF tiling
     * generates dynamic shapes for extract and pad from static shapes. For such cases setting the bounds here
     * based on the original input shape. For other cases bounds are inferred from available input bounds*/

    auto newType = origType.changeShape(ShapeRef(newShape));
    if (newShape.isDynamic()) {
        if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
            newType = boundedType.changeBounds(inputTileInfo.bounds);
        } else {
            auto origInputShape = to_small_vector(getShape(origInput));
            SmallVector<int64_t, 4> boundsValue(newShape.begin(), newShape.end());
            for (size_t ind = 0, sz = boundsValue.size(); ind < sz; ++ind) {
                if (boundsValue[ind] == mlir::ShapedType::kDynamic) {
                    boundsValue[ind] = origInputShape[ind];
                }
            }
            auto newBounds = vpux::BoundsRef(ArrayRef<int64_t>(boundsValue.data(), boundsValue.size()));
            newType = Core::BoundedTensorType::get(newType, newBounds);
        }
    }

    // by default output type loses NPU-specific attributes so we have to set it manually
    extractTile->getResult(0).setType(newType);

    generatedSlices.emplace_back(extractTile);

    return extractTile;
}

mlir::Type vpux::VPU::extractResultType(mlir::Type origType, SCFShapeRef newShape, BoundsRef bounds) {
    auto ndTensorType = mlir::cast<vpux::NDTypeInterface>(origType);
    auto origElemType = ndTensorType.getElementType();

    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(origElemType),
                    "Per axis quantized types are not supported in scf");

    const auto tensorDesc =
            vpux::getTensorAttr(origElemType.getContext(), ndTensorType.getDimsOrder(), ndTensorType.getMemSpace(),
                                mlir::isa<Core::BoundedTensorType>(origType) ? bounds : BoundsRef{});

    SmallVector<mlir::Value> dynamicDims;  // unused cause for shape static dims are enough
    SmallVector<int64_t> staticDims;
    mlir::dispatchIndexOpFoldResults(newShape, dynamicDims, staticDims);
    return mlir::RankedTensorType::get(staticDims, origElemType, tensorDesc);
}

SCFTileInfo vpux::VPU::getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                              const SCFTileInfo& outputTile) {
    auto origWeightsTableShape = mlir::cast<mlir::ShapedType>(origWeightsTableType).getShape();

    SCFTileInfo weightsTableTile(origWeightsTableShape, builder);
    weightsTableTile.offsets[0] = outputTile.offsets[Dims4D::Act::C.ind()];
    weightsTableTile.shape[0] = outputTile.shape[Dims4D::Act::C.ind()];
    return weightsTableTile;
}

std::pair<std::optional<mlir::Range>, std::optional<int64_t>> vpux::VPU::solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
        const int64_t stride, mlir::OpFoldResult origInputSize, int64_t origOutputSize,
        const std::pair<int64_t, int64_t>& origPadding, mlir::OpFoldResult& padBefore, mlir::OpFoldResult& padAfter) {
    auto zero = builder.getIndexAttr(0);
    auto one = builder.getIndexAttr(1);
    mlir::Range inputRange = {zero, zero, one};
    mlir::Range outputRange = {outputTile.offsets[dim.ind()], outputTile.shape[dim.ind()], one};

    // define dimensions (d0, d1, ...) as variables which are represented by loop dim identifier
    // and symbols (s0, s1, ...) which are either known constants or known attributes of operation
    mlir::AffineExpr s0, s1, d0, d1, d2;
    bindDims(builder.getContext(), d0, d1, d2);
    bindSymbols(builder.getContext(), s0, s1);

    mlir::AffineExpr requiredInputSizeExpr = (d0 - 1) * stride + kernel;
    auto requiredInputSizeMap = mlir::AffineMap::get(1, 0, {requiredInputSizeExpr}, builder.getContext());

    std::optional<int64_t> dimBound;
    if (!outputTile.bounds.raw().empty()) {
        auto outputTileBound = builder.getIntegerAttr(builder.getIndexType(), outputTile.bounds[dim]);
        SmallVector<mlir::Attribute> resultsAttrs;
        if (requiredInputSizeMap.constantFold({outputTileBound}, resultsAttrs).succeeded()) {
            if (auto result = mlir::dyn_cast<mlir::IntegerAttr>(resultsAttrs.front())) {
                dimBound = result.getInt() - origPadding.first - origPadding.second;
            }
        }
    }

    if (mlir::isConstantIntValue(outputTile.axis[dim.ind()], 1)) {
        return {std::nullopt, dimBound};
    }

    const auto hasPadBefore = origPadding.first != 0;
    const auto hasPadAfter = origPadding.second != 0;

    // input offset is based on output tile offset and operation's parameters
    // current calculation is
    // offset: max((output offset) * stride - padding, 0).
    // size: (output size - 1) * stride + kernel - padding
    // if operation has padding, the median tile size will be corrected later if needed
    if (!hasPadBefore && stride == 1) {
        inputRange.offset = outputRange.offset;
    } else {
        mlir::AffineExpr offsetExpr = d0 * stride - origPadding.first;
        auto offsetMap = mlir::AffineMap::get(1, 1, {offsetExpr, s0}, builder.getContext());
        inputRange.offset = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "inputOffset"), offsetMap,
                                                                      {outputRange.offset, zero});

        auto maxDiffMap = mlir::AffineMap::get(1, 2, {s0 - offsetExpr, s1}, builder.getContext());
        auto maxDiffValue = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "maxDiff"), maxDiffMap,
                                                                      {outputRange.offset, zero, zero});
        auto padBeforeMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padBefore = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "paddingBefore"), padBeforeMap,
                                                              {maxDiffValue, builder.getIndexAttr(origPadding.first)});
    }

    if (hasPadAfter) {
        auto maxDiffMap = mlir::AffineMap::get(2, 2, {d1 + requiredInputSizeExpr - s0, s1}, builder.getContext());
        auto maxDiffValue =
                mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "maxDiff"), maxDiffMap,
                                                          {outputRange.size, inputRange.offset, origInputSize, zero});
        auto padAfterMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padAfter = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "paddingAfter"), padAfterMap,
                                                             {maxDiffValue, builder.getIndexAttr(origPadding.second)});
    }

    const auto numTiles = mlir::getConstantIntValue(outputTile.axis[dim.ind()]);
    const bool hasTwoTiles = numTiles.has_value() && numTiles.value() == 2;
    const bool outputSizeDivisible = numTiles.has_value() && !mlir::ShapedType::isDynamic(origOutputSize) &&
                                     origOutputSize % numTiles.value() == 0;
    const bool symmetricPadding = origPadding.first == origPadding.second;
    if (hasTwoTiles && outputSizeDivisible && symmetricPadding) {
        auto sizeMap = mlir::AffineMap::get(1, 1, {requiredInputSizeExpr - s0}, builder.getContext());
        inputRange.size = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputSize"), sizeMap,
                {outputRange.size, builder.getIndexAttr(origPadding.first)});
    } else {
        auto sizeMap = mlir::AffineMap::get(3, 0, {requiredInputSizeExpr - d1 - d2}, builder.getContext());
        inputRange.size = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputSize"), sizeMap,
                {outputRange.size, hasPadBefore ? padBefore : zero, hasPadAfter ? padAfter : zero});
    }

    return {inputRange, dimBound};
}

bool vpux::VPU::checkFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate,
                            const llvm::SetVector<mlir::Operation*>& producers) {
    // TODO E-172888 rewrite unified code for checking compatibility with current VF

    auto* producer = producerCandidate.getOwner();

    if (!mlir::isa<mlir::TilingInterface>(producer)) {
        return false;
    }

    if (VPU::isPureViewOp(producer) || VPU::isPureViewOp(consumer.getOwner())) {
        return true;
    }

    auto producerShape = getShape(producerCandidate);
    auto alignment = getAlignment(producer, producerShape, producerShape);
    const auto opAddsComputationalCost = [](auto* operation) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);

        return nceOp != nullptr && llvm::any_of(nceOp.getKernelSizeVal(), [](int64_t k) {
                   return k > 1;
               });
    };
    if (llvm::any_of(producers, opAddsComputationalCost)) {
        if (alignment[Dims4D::Act::W.ind()] > 1 || alignment[Dims4D::Act::H.ind()] > 1) {
            return false;
        }
    }

    const auto hasMCStategy = [](mlir::Operation* operation) {
        auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    auto consumerHasStrategy = hasMCStategy(consumer.getOwner());
    auto producerHasStrategy = hasMCStategy(producer);

    if (!consumerHasStrategy && !producerHasStrategy) {
        return true;
    }

    if (consumerHasStrategy ^ producerHasStrategy) {
        return false;
    }

    auto producerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(producerCandidate.getOwner());
    auto consumerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(consumer.getOwner());

    VPU::SiblingOpsAnalysis siblingAnalisys(consumer.getOwner());

    auto consumerDistrType = mlir::cast<VPU::DistributedTensorType>(
            consumerClusterOp.getDistributedTypeForOpOperand(consumer, false, siblingAnalisys));
    auto producerDistrType = mlir::cast<VPU::DistributedTensorType>(producerClusterOp.getDistributedTypeForOpResult(
            producerCandidate, producerClusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false));

    return areDistributionAttrsCompatible(producerDistrType, consumerDistrType, true).succeeded();
}

bool vpux::VPU::isNceOpWithPadAttr(mlir::Operation* op) {
    return (mlir::isa<mlir::TilingInterface>(op) && mlir::isa<VPU::NCEOpInterface>(op) && op->hasAttr("pad"));
}

/**
 * @brief Decodes block position from string code into an array of TilePosition
 *
 * Each digit in the string represents the TilePosition for one tiling axis:
 * 0 = MIDDLE, 1 = END, 2 = START, 3 = FULLBLK
 *
 * Example: "123" → [END, START, FULLBLK]
 *
 * @param blockPositionCode String of position digits (e.g. "22", "123", "01")
 * @return SmallVector<TilePosition> Positions for each tiling axis
 */
static SmallVector<TilePosition> decodeBlockPositionFromString(llvm::StringRef blockPositionCode) {
    SmallVector<TilePosition> positions;

    for (char c : blockPositionCode) {
        switch (c) {
        case '0':
            positions.push_back(TilePosition::MIDDLE);
            break;
        case '1':
            positions.push_back(TilePosition::END);
            break;
        case '2':
            positions.push_back(TilePosition::START);
            break;
        case '3':
            positions.push_back(TilePosition::FULLBLK);
            break;
        default:
            VPUX_THROW("Invalid block position code character: {0}", c);
            break;
        }
    }

    return positions;
}

/**
 * @brief Calculates the correct slice offsets for tensor.cast operations based on block position
 *
 * This function determines the appropriate offsets for slicing a tensor
 * Sophisticated offset calculation logic based on:
 *    1. Block position (START, MIDDLE, END) - can be inferred from function name or context
 *    2. Padding requirements for convolutions in different positions
 *    3. Alignment requirements for the target architecture
 *    4. Original tensor coordinates in the full image
 *

 * @param castOp The original tensor::CastOp operation (for context analysis)
 * @return SmallVector<int64_t> The calculated offsets for each dimension
 */
SmallVector<int64_t> calculateSliceOffsets(mlir::tensor::CastOp castOp) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(castOp.getOperand().getType());

    // Slice logic depends on block position, so we need to figure out where we are.
    // Block position is encoded into parent func name,  for example: func.func private @main_func0_dims_CH_cases_22
    // 22 means we are in the END position for both height and width dimensions.
    auto parentFunc = castOp->getParentOfType<mlir::func::FuncOp>();
    SmallVector<llvm::StringRef, 8> funcNameParts;
    parentFunc.getName().split(funcNameParts, "_");
    auto blockPositionCode = funcNameParts.empty() ? llvm::StringRef("") : funcNameParts.back();
    auto blockPositions = decodeBlockPositionFromString(blockPositionCode);

    // Make sure amount of block positions matches amount of dynamic dims in input tensor shape
    size_t dynamicDimCount =
            std::accumulate(inputType.getShape().begin(), inputType.getShape().end(), 0, [](size_t count, int64_t dim) {
                return dim == mlir::ShapedType::kDynamic ? count + 1 : count;
            });
    if (blockPositions.size() != dynamicDimCount) {
        VPUX_THROW("Mismatch in block positions: expected {0}, got {1}", dynamicDimCount, blockPositions.size());
    }
    auto blockPositionsIter = blockPositions.begin();
    SmallVector<int64_t> offsets;
    offsets.reserve(dynamicDimCount);

    auto inputBounds = mlir::cast<vpux::Core::BoundedTensorType>(castOp.getOperand().getType()).getBounds().raw();
    auto outputBounds = mlir::cast<vpux::Core::BoundedTensorType>(castOp.getResult().getType()).getBounds().raw();

    for (const auto& indexedDim : inputType.getShape() | vpux::indexed) {
        auto indx = indexedDim.index();
        auto dim = indexedDim.value();
        if (dim == mlir::ShapedType::kDynamic) {
            // For dynamic dimensions, use simple heuristic:
            // Offsets set to 0 for START and FULLBLK positions
            // Offsets set to (inputBound[i] - outputBound[i]) for END position
            // MIDDLE position offset calculation can be more complex depending on tile size and alignment, but we could
            // try (inputBound[i] - outputBound[i])/2 as a placeholder for now
            // Correct logic will be introduced later by:
            // TRACK: E#191734
            switch (*blockPositionsIter) {
            case TilePosition::START:
                // Offset is 0 for START position
                offsets.push_back(0);
                break;
            case TilePosition::MIDDLE:
                // Offset is calculated based on tile size - placeholder logic
                offsets.push_back((inputBounds[indx] - outputBounds[indx]) / 2);
                break;
            case TilePosition::END:
                // Max possible offset
                offsets.push_back(inputBounds[indx] - outputBounds[indx]);
                break;
            case TilePosition::FULLBLK:
                // No offset for FULLBLK
                offsets.push_back(0);
                break;
            }
            blockPositionsIter = std::next(blockPositionsIter);
        } else {
            // For static dimensions, offset is always 0
            offsets.push_back(0);
        }
    }
    return offsets;
}
/**
 * @brief Clones an MLIR operation with ability to remap operations
 * based on specific cases.
 *
 * This function takes an existing MLIR operation and creates a clone of it using the provided
 * OpBuilder. It allows for remapping of certain operations based on predefined cases. For example,
 * if the operation is a tensor::CastOp, it is replaced with a VPU::SliceOp with appropriate parameters.
 * For all other operations, a standard clone is performed using the provided IRMapping.
 * @param oldOp The original MLIR operation to be cloned
 * @param builder The OpBuilder used to create the new operation
 * @param mapper An IRMapping used for cloning operations
 * @return A pointer to the newly created MLIR operation
 */
mlir::Operation* cloneOperationMapped(mlir::Operation& oldOp, mlir::OpBuilder& builder, mlir::IRMapping& mapper) {
    return llvm::TypeSwitch<mlir::Operation&, mlir::Operation*>(oldOp)
            .Case<mlir::tensor::CastOp>([&](mlir::tensor::CastOp castOp) {
                auto outputBounds =
                        mlir::cast<vpux::Core::BoundedTensorType>(castOp.getResult().getType()).getBounds().raw();

                // Use the dedicated function to calculate slice offsets
                auto offsets = calculateSliceOffsets(castOp);

                return builder.create<SliceOp>(castOp.getLoc(), mapper.lookup(castOp.getOperand()), offsets,
                                               outputBounds);
            })
            .Default([&builder, &mapper](mlir::Operation& defaultOp) -> mlir::Operation* {
                return builder.clone(defaultOp, mapper);
            });
};

/**
 * @brief Clones an existing MLIR function operation with optional type modification
 *
 * This function creates a deep copy of an MLIR function operation, allowing for optional
 * function type changes. It performs a complete clone of the function body, including
 * all blocks, operations, and value mappings. When a new function type is provided,
 * it also performs type inference for VPU dialect operations and updates the function's
 * return types based on the terminator operations.
 *
 * @param originalFunc The source function operation to be cloned
 * @param newName The name for the newly created function
 * @param moduleOp The target module where the new function will be inserted
 * @param newFuncType Optional new function type for the cloned function. If nullptr,
 *                    the original function's type is preserved
 *
 * @return The newly created function operation that is a clone of the original
 *
 * @note The cloned function is positioned immediately after the original function
 *       in the module. When newFuncType is provided, VPU dialect operations undergo
 *       shape inference and the function's return types are updated based on the
 *       actual terminator operand types.
 */
mlir::func::FuncOp vpux::VPU::cloneFuncOp(mlir::func::FuncOp originalFunc, const std::string& newName,
                                          mlir::FunctionType newFuncType) {
    assert(newFuncType != nullptr && "newFuncType should be a valid FunctionType");
    auto moduleOp = vpux::getModuleOp(originalFunc);
    mlir::OpBuilder builder(moduleOp.getContext());
    auto newFunc = builder.create<mlir::func::FuncOp>(takeOpLoc(originalFunc, newName), newName, newFuncType);
    moduleOp.push_back(newFunc);
    newFunc->moveAfter(originalFunc);

    mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
    for (auto& oldBlock : originalFunc.getBody()) {
        auto* newBlock = newFunc.addEntryBlock();
        for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
            oldToNewMap[oldArg] = newArg;
        }

        builder.setInsertionPointToStart(newBlock);
        for (auto& oldOp : oldBlock.getOperations()) {
            mlir::IRMapping mapper;
            for (auto operand : oldOp.getOperands()) {
                mapper.map(operand, oldToNewMap[operand]);
            }

            auto newOp = cloneOperationMapped(oldOp, builder, mapper);
            if (mlir::isa<VPU::VPUDialect>(newOp->getDialect())) {
                vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);
            }

            // Unique to SliceOp as we cannot modify the static size attribute in the inferReturnType interface
            // TODO: will be removed by E#198282
            if (mlir::isa<VPU::SliceOp>(newOp)) {
                auto sliceOp = mlir::cast<VPU::SliceOp>(newOp);
                auto newShape = mlir::cast<mlir::ShapedType>(sliceOp.getOutput().getType()).getShape();
                sliceOp.setStaticSizesAttr(getIntArrayAttr(newOp->getContext(), newShape));
            }

            for (size_t i = 0; i < oldOp.getNumResults(); ++i) {
                oldToNewMap[oldOp.getResult(i)] = newOp->getResult(i);
            }
        }
    }

    SmallVector<mlir::Type> newReturnTypes;
    for (auto& block : newFunc.getBody()) {
        auto terminatorOp = block.getTerminator();
        for (auto operands : terminatorOp->getOperands()) {
            newReturnTypes.push_back(operands.getType());
        }
    }

    auto modifiedType =
            mlir::FunctionType::get(newFunc.getContext(), newFunc.getFunctionType().getInputs(), newReturnTypes);
    newFunc.setType(modifiedType);
    return newFunc;
}

mlir::RankedTensorType vpux::VPU::removeBoundsAttr(mlir::RankedTensorType type) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    const auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                                ndType.getMemSpace(), {}, {});
    return tensorType;
}

// Check if all operands of op are defined before 'beforeOp' and op has no side effects
bool isSafeToMoveBefore(mlir::Operation* op, mlir::Operation* beforeOp) {
    // Check SSA dependencies
    for (auto operand : op->getOperands()) {
        if (auto definingOp = operand.getDefiningOp()) {
            if (definingOp->getBlock() != op->getBlock()) {
                continue;
            }

            if (!definingOp->isBeforeInBlock(beforeOp)) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Moves affine and arithmetic operations early in the execution order within a block
 *
 * This function optimizes the placement of affine dialect and arithmetic dialect operations
 * by moving them as early as possible in the block's execution order, while maintaining
 * correctness. It also handles scf.if operations that return only index types by treating
 * them similarly to affine/arithmetic operations.
 *
 * The function performs a two-phase operation:
 * 1. Identifies all affine/arithmetic operations and scf.if operations that return only
 *    index types as candidates for early movement
 * 2. For each candidate operation, finds the earliest safe position in the block where
 *    it can be moved without violating dependencies
 *
 * @param block The MLIR block within which to perform the optimization
 */
void vpux::VPU::moveAffineArithOpsEarly(mlir::Block& block) {
    SmallVector<mlir::Operation*> affineArithOps;

    // Collect affine/arithmetic operations and qualifying scf.if operations
    for (auto& op : block) {
        if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(op.getDialect())) {
            affineArithOps.push_back(&op);
            continue;
        }

        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            // Only move scf.if operations that return only index types
            bool allIndexTypes = llvm::all_of(ifOp.getResultTypes(), [](mlir::Type type) {
                return mlir::isa<mlir::IndexType>(type);
            });
            if (allIndexTypes) {
                affineArithOps.push_back(&op);
            }
        }
    }

    // Move operations to earliest safe positions
    for (auto* op : affineArithOps) {
        mlir::Operation* targetPosition = nullptr;

        // Find the earliest position where this operation can be safely moved
        for (auto& candidate : block) {
            if (&candidate == op) {
                break;  // Don't move past current position
            }

            // Skip other affine/arithmetic operations - we want to move before them
            if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(candidate.getDialect())) {
                continue;
            }

            // Check if it's safe to move before this candidate
            if (isSafeToMoveBefore(op, &candidate)) {
                targetPosition = &candidate;
                break;
            }
        }

        // Only move if we found a valid earlier position
        if (targetPosition != nullptr) {
            op->moveBefore(targetPosition);
        }
    }
}

void vpux::VPU::addCheckForBlockSize(mlir::OpBuilder& builder, mlir::tensor::DimOp dimOp, mlir::Value blockSize,
                                     mlir::func::FuncOp funcOp, llvm::StringRef errorMsg) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto blockSizeDefiningOp = blockSize.getDefiningOp();
    mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
    if (dimOp->getBlock() == blockSizeDefiningOp->getBlock()) {
        if (dimOp->isBeforeInBlock(blockSizeDefiningOp)) {
            builder.setInsertionPointAfter(blockSizeDefiningOp);
            loc = appendLoc(blockSizeDefiningOp->getLoc(), "tile0_check");
        } else {
            builder.setInsertionPointAfter(dimOp);
            loc = appendLoc(dimOp->getLoc(), "tile0_check");
        }
    } else {
        mlir::DominanceInfo dom(funcOp);
        if (dom.dominates(blockSizeDefiningOp, dimOp)) {
            builder.setInsertionPointAfter(dimOp);
            loc = appendLoc(dimOp->getLoc(), "tile0_check");
        } else {
            builder.setInsertionPointAfter(blockSizeDefiningOp);
            loc = appendLoc(blockSizeDefiningOp->getLoc(), "tile0_check");
        }
    }

    auto stepGreaterThanBound =
            builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, dimOp, blockSize);
    builder.create<mlir::cf::AssertOp>(takeOpLoc(stepGreaterThanBound, "assert"), stepGreaterThanBound, errorMsg);
}

mlir::LogicalResult vpux::VPU::getTensorDimOpFromIndex(mlir::OpBuilder& builder, mlir::Value tensor, int64_t dimIdx,
                                                       mlir::tensor::DimOp& dimOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    bool tensorIsBlockArg = false;
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
        auto blockParentOp = blockArg.getOwner()->getParentOp();
        tensorIsBlockArg = true;
        while (blockParentOp != nullptr) {
            if (mlir::isa<mlir::func::FuncOp>(blockParentOp)) {
                break;
            }

            auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockParentOp);
            if (forOp == nullptr) {
                return vpux::errorAt(blockParentOp->getLoc(),
                                     "Expected parent scf.for operation for block argument but got {0}",
                                     blockParentOp->getName());
            }

            unsigned idx = blockArg.getArgNumber() - 1;
            mlir::Value initVal = forOp.getInitArgs()[idx];
            auto definingOp = initVal.getDefiningOp();
            if (definingOp != nullptr) {
                tensor = initVal;
                tensorIsBlockArg = false;
                break;
            }

            blockParentOp = blockParentOp->getParentOp();
        }
    }

    if (tensorIsBlockArg) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor);
        auto blockParentOp = blockArg.getOwner()->getParentOp();
        builder.setInsertionPointToStart(blockArg.getOwner());
        dimOp = builder.create<mlir::tensor::DimOp>(takeOpLoc(blockParentOp, "dim_" + std::to_string(dimIdx)), tensor,
                                                    dimIdx);
    } else {
        auto loc = tensor.getDefiningOp();
        builder.setInsertionPointAfter(loc);
        dimOp = builder.create<mlir::tensor::DimOp>(takeOpLoc(loc, "dim_" + std::to_string(dimIdx)), tensor, dimIdx);
    }
    return mlir::success();
}

void replaceUsesOfOpWithDominanceCheck(mlir::Value toReplace, mlir::Value newValue) {
    auto newValueDefiningOp = newValue.getDefiningOp();
    if (newValueDefiningOp == nullptr) {
        return;
    }

    mlir::DominanceInfo dom(newValueDefiningOp);
    for (auto& use : llvm::make_early_inc_range(toReplace.getUses())) {
        mlir::Operation* userOp = use.getOwner();

        if (!dom.properlyDominates(newValue, userOp)) {
            continue;  // Skip users not dominated by the new value
        }

        use.set(newValue);
    }
}

SmallVector<mlir::Value> vpux::VPU::applyIndexBacktracking(mlir::tensor::InsertSliceOp insertSliceOp,
                                                           ArrayRef<size_t> dimsToAdjust) {
    auto parentForOp = insertSliceOp->getParentOfType<mlir::scf::ForOp>();
    assert(parentForOp && "InsertSliceOp not inside scf.for");

    auto getLoopIV = [&](mlir::Value val) {
        mlir::Value inductionVar = nullptr;
        mlir::scf::ForOp parentOp = nullptr;

        if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
            auto blockOp = blockArg.getOwner()->getParentOp();
            if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockOp)) {
                inductionVar = val;
                parentOp = forOp;
            }
        }

        if (inductionVar == nullptr) {
            OpChainAnalysis utils;
            auto opChain = utils.collectParentOpsChain(val);

            for (auto op : opChain) {
                for (auto operand : op->getOperands()) {
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
                        auto blockOp = blockArg.getOwner()->getParentOp();
                        if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockOp)) {
                            inductionVar = operand;
                            parentOp = forOp;
                            break;
                        }
                    }
                }
            }
        }

        return std::pair<mlir::Value, mlir::scf::ForOp>{inductionVar, parentOp};
    };

    auto insertionPoint = &(parentForOp.getBody()->front());
    mlir::OpBuilder localBuilder(insertSliceOp);
    localBuilder.setInsertionPointToStart(parentForOp.getBody());

    SmallVector<mlir::OpFoldResult> offsetsToAdjust;
    if (!dimsToAdjust.empty()) {
        for (auto dim : dimsToAdjust) {
            offsetsToAdjust.push_back(insertSliceOp.getMixedOffsets()[dim]);
        }
    } else {
        offsetsToAdjust = insertSliceOp.getMixedOffsets();
    }

    SmallVector<mlir::Value> newOffsets;
    for (auto offset : offsetsToAdjust) {
        auto offsetValue = mlir::dyn_cast_or_null<mlir::Value>(offset);
        if (offsetValue == nullptr) {
            continue;
        }

        if (!mlir::isa<mlir::BlockArgument>(offsetValue) &&
            mlir::isa<mlir::arith::ConstantOp>(offsetValue.getDefiningOp())) {
            continue;
        }

        auto [loopIv, parentOp] = getLoopIV(offsetValue);
        assert(loopIv && "Could not find loop induction variable for offset");

        auto upperBound = parentOp.getUpperBound();
        auto stepSize = parentOp.getStep();

        auto nextOffset =
                localBuilder.create<mlir::arith::AddIOp>(takeOpLoc(insertionPoint, "next_offset"), loopIv, stepSize);
        auto exceedsBound = localBuilder.create<mlir::arith::CmpIOp>(
                takeOpLoc(insertionPoint, "exceeds_bound"), mlir::arith::CmpIPredicate::sgt, nextOffset, upperBound);
        auto ifExceedsBound = localBuilder.create<mlir::scf::IfOp>(
                takeOpLoc(insertionPoint, "if_exceeds_bound"), llvm::ArrayRef<mlir::Type>{localBuilder.getIndexType()},
                exceedsBound,
                /*withElseRegion=*/true);

        {
            // create affine map that takes the step size and upper bound and return upper_bound - step_size

            mlir::OpBuilder thenBuilder = ifExceedsBound.getThenBodyBuilder();
            // Create affine expression: s0 - s1 (upperBound - stepSize)
            mlir::AffineExpr s0, s1;
            bindSymbols(thenBuilder.getContext(), s0, s1);
            mlir::AffineExpr subExpr = s0 - s1;

            // Create the affine map with 0 dimensions and 2 symbols
            auto affineMap = mlir::AffineMap::get(0, 2, {subExpr}, thenBuilder.getContext());

            // Apply the affine map to compute upperBound - stepSize
            auto adjustedOffset = mlir::affine::makeComposedFoldedAffineApply(
                    thenBuilder, appendLoc(ifExceedsBound->getLoc(), "adjusted_offset"), affineMap,
                    {upperBound, stepSize});

            // Convert to a concrete value if needed
            auto adjustedOffsetVal = mlir::getValueOrCreateConstantIndexOp(
                    thenBuilder, appendLoc(ifExceedsBound->getLoc(), "adjusted_offset_val"), adjustedOffset);

            thenBuilder.create<mlir::scf::YieldOp>(appendLoc(ifExceedsBound->getLoc(), "yield"),
                                                   mlir::ValueRange{adjustedOffsetVal});
        }
        {
            mlir::OpBuilder elseBuilder = ifExceedsBound.getElseBodyBuilder();
            elseBuilder.create<mlir::scf::YieldOp>(appendLoc(ifExceedsBound->getLoc(), "yield"),
                                                   mlir::ValueRange{loopIv});
        }

        auto newOffsetValue = ifExceedsBound.getResult(0);
        replaceUsesOfOpWithDominanceCheck(loopIv, newOffsetValue);
        replaceUsesOfOpWithDominanceCheck(offsetValue, newOffsetValue);
        newOffsets.push_back(newOffsetValue);
    }

    return newOffsets;
}

mlir::Operation* vpux::VPU::castOutputForInsertion(mlir::OpBuilder& builder, const SCFTileInfo& outputTile,
                                                   DimArrRef dims, mlir::Operation* operation,
                                                   mlir::Operation* tiledOperation) {
    const auto outputShape = getShape(operation->getResult(0));
    const auto nonTilingDynDims = [&]() -> SmallVector<Dim> {
        SmallVector<Dim> dynDims;
        for (auto idx : irange(outputShape.size())) {
            const auto dim = Dim(idx);
            if (outputShape[dim] != mlir::ShapedType::kDynamic) {
                continue;
            }

            if (!llvm::is_contained(dims, dim)) {
                dynDims.push_back(dim);
            }
        }

        return dynDims;
    }();

    if (nonTilingDynDims.empty()) {
        return tiledOperation;
    }

    auto tiledOpType = mlir::cast<vpux::NDTypeInterface>(tiledOperation->getResult(0).getType());
    auto castedOutputShape = Shape(tiledOpType.getShape());
    auto tiledOpBoundedShape = getBoundedShape(tiledOpType);
    for (const auto dynDim : nonTilingDynDims) {
        const auto evaluatedDim = mlir::getConstantIntValue(outputTile.shape[dynDim.ind()]);
        castedOutputShape[dynDim] = evaluatedDim.value_or(tiledOpBoundedShape[dynDim]);
    }

    mlir::Type castedTiledOutputType = tiledOpType.changeShape(castedOutputShape);
    return builder.create<mlir::tensor::CastOp>(appendLoc(tiledOperation->getLoc(), "insert_cast"),
                                                castedTiledOutputType, tiledOperation->getResults());
}

/**
 * @brief For tensors with dynamic dimensions along padded axes, the PadOp might have identical input and output
 * types. However, for static dimensions, the shapes of PadOp source and destination differ along the padded axes.
 * Add a tensor.cast to make the input dynamic when required, as the input tensor on which the NCE operation operates
 * will determine the shape of the tensor.
 */
mlir::Value getInputOperand(mlir::tensor::PadOp padOp, mlir::OpBuilder builder) {
    mlir::RankedTensorType srcType = padOp.getSourceType(), dstType = padOp.getResultType();
    bool isDynamic = false;
    VPUX_THROW_WHEN(srcType == nullptr || dstType == nullptr, "Expected RankedTensorType for PadOp source and result");

    auto newShape = SmallVector<int64_t>{};
    vpux::ShapeRef newBounds = vpux::ShapeRef{};
    transform(enumerate(padOp.getMixedLowPad()), std::back_inserter(newShape),
              [&srcType, &padOp, &isDynamic](auto&& indexedOffset) {
                  auto [idx, lowOffset] = indexedOffset;
                  auto highOffset = padOp.getMixedHighPad()[idx];

                  // Check if both low and high padding are static integer attributes
                  bool lowIsStatic = false, highIsStatic = false;

                  if (auto attr = mlir::dyn_cast<mlir::Attribute>(lowOffset)) {
                      lowIsStatic = mlir::isa<mlir::IntegerAttr>(attr);
                  }

                  if (auto attr = mlir::dyn_cast<mlir::Attribute>(highOffset)) {
                      highIsStatic = mlir::isa<mlir::IntegerAttr>(attr);
                  }

                  // If either padding is not static, make the dimension dynamic
                  if (!lowIsStatic || !highIsStatic) {
                      isDynamic = true;
                      return mlir::ShapedType::kDynamic;
                  }

                  return srcType.getShape()[idx];
              });

    if (isDynamic) {
        newBounds = getBoundedShape(srcType);
    }

    auto dstBounds = vpux::BoundsRef(newBounds);
    const auto inType = mlir::cast<NDTypeInterface>(srcType);
    auto outDesc = vpux::getTensorAttr(builder.getContext(), inType.getDimsOrder(), inType.getMemSpace(), dstBounds);
    auto newDstType = mlir::RankedTensorType::get(newShape, dstType.getElementType(), outDesc);
    if (newDstType != srcType) {
        return builder.create<mlir::tensor::CastOp>(appendLoc(padOp.getLoc(), "cast"), newDstType, padOp.getSource());
    }

    return padOp.getSource();
}

void vpux::VPU::restorePaddingAttribute(mlir::Operation* region, Logger log) {
    SmallVector<std::pair<mlir::Operation*, mlir::tensor::PadOp>> worklist;
    region->walk([&](mlir::tensor::PadOp padOp) {
        for (auto user : padOp->getUsers()) {
            if (VPU::isNceOpWithPadAttr(user)) {
                worklist.push_back({user, padOp});
            } else if (auto extractSlice = mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(user)) {
                if (user->getParentOfType<mlir::scf::ForallOp>() == nullptr) {
                    continue;
                }
                auto offsets = extractSlice.getMixedOffsets();
                bool isSpatialMC = false;
                for (auto [idx, offset] : enumerate(offsets)) {
                    if (!mlir::isa_and_nonnull<mlir::Value>(offset)) {
                        continue;
                    }

                    if (idx > static_cast<size_t>(Dims4D::Act::C.ind())) {
                        isSpatialMC = true;
                        break;
                    }
                }

                if (!isSpatialMC) {
                    for (auto extractUser : user->getUsers()) {
                        if (VPU::isNceOpWithPadAttr(extractUser)) {
                            worklist.push_back({extractUser, padOp});
                        }
                    }
                }
            }
        }
    });

    OpChainAnalysis opChainAnalysis;
    auto getPadAttribute = [&](mlir::tensor::PadOp padOp) {
        auto spatialDims = {Dims4D::Act::W, Dims4D::Act::H};
        llvm::SmallVector<int64_t> padValues;
        for (auto dim : spatialDims) {
            auto lowPad = padOp.getMixedLowPad()[dim.ind()];
            auto highPad = padOp.getMixedHighPad()[dim.ind()];

            ValueRangeMap emptyMap;
            auto lowValue = opChainAnalysis.getOpFoldResultValue(lowPad, emptyMap);
            auto highValue = opChainAnalysis.getOpFoldResultValue(highPad, emptyMap);
            VPUX_THROW_WHEN(!lowValue.has_value() || !highValue.has_value(),
                            "Failed to compute static padding values for {0} operation", padOp->getName());
            padValues.emplace_back(lowValue.value()[0]);
            padValues.emplace_back(highValue.value()[0]);
        }

        return VPU::getPaddingAttr(padOp.getContext(), padValues[0], padValues[1], padValues[2], padValues[3]);
    };

    mlir::IRRewriter rewriter(region->getContext());
    for (auto [nceOp, padOp] : llvm::make_early_inc_range(worklist)) {
        log.trace("Found convolution operation {0} with tensor.pad parent", nceOp->getName());

        rewriter.setInsertionPoint(nceOp);
        auto inputOperand = getInputOperand(padOp, rewriter);
        auto restoredPadAttr = getPadAttribute(padOp);

        mlir::IRMapping mapper;
        mapper.map(nceOp->getOperand(0), inputOperand);
        auto newNceOp = rewriter.clone(*nceOp, mapper);
        newNceOp->setAttr("pad", restoredPadAttr);
        vpux::inferReturnTypes(newNceOp, vpux::InferShapedTypeMode::SHAPE);

        // Replace all uses of the original convolution operation with the new one
        rewriter.replaceOp(nceOp, newNceOp->getResults());

        // TODO: Remove once E195927 is solved
        for (auto user : llvm::make_early_inc_range(padOp->getUsers())) {
            if (!mlir::isa<mlir::tensor::ExtractSliceOp>(user)) {
                continue;
            }

            if (user->getUsers().empty()) {
                user->erase();
            }
        }
        if (padOp->getUsers().empty()) {
            padOp.erase();
        }
    }
}
