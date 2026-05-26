//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<mlir::Type> inferElemType(VPU::AffineReshapeOpAdaptor affineReshapeOp, mlir::Type inputElemType) {
    const auto perAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    if (perAxisQType == nullptr) {
        return inputElemType;
    }

    const auto inputQAxis = perAxisQType.getQuantizedDimension();

    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
    const auto outputShape = parseIntArrayAttr<int64_t>(affineReshapeOp.getShapeValue());
    const auto inputShape = getShape(affineReshapeOp.getInput()).raw();

    // get output dims for input Q axis
    const auto outputDims = dimMapping[inputQAxis];
    int64_t outQAxis = -1;
    int64_t inputQAxisSize = inputShape[inputQAxis];

    if (inputQAxisSize == 1) {
        // Per tensor, but must be per channel, do not handle it here
        return mlir::failure();
    }

    for (const auto& dim : outputDims) {
        if (inputQAxisSize == outputShape[dim]) {
            // firstly check that element is unique and others == 1
            if (std::find_if(outputDims.begin(), outputDims.end(), [&](int64_t elem) {
                    return (outputShape[elem] != 1 && outputShape[elem] != inputQAxisSize);
                }) != outputDims.end()) {
                return mlir::failure();
            }
            outQAxis = dim;
            break;
        }
    }

    if (outQAxis == -1) {
        return mlir::failure();
    }

    if (const auto perAxisUniformQType =
                mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElemType)) {
        if (auto quantileStorage = mlir::dyn_cast<vpux::type::QuantileType>(perAxisUniformQType.getStorageType())) {
            return mlir::quant::UniformQuantizedPerAxisType::get(
                    perAxisUniformQType.getFlags(), quantileStorage, perAxisUniformQType.getExpressedType(),
                    perAxisUniformQType.getScales(), perAxisUniformQType.getZeroPoints(),
                    static_cast<int32_t>(outQAxis), perAxisUniformQType.getStorageTypeMin(),
                    perAxisUniformQType.getStorageTypeMax());
        }
    }
    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), static_cast<int32_t>(outQAxis),
            perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
}

}  // namespace

mlir::LogicalResult vpux::VPU::AffineReshapeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AffineReshapeOpAdaptor affineReshape(operands, attrs, prop);
    if (mlir::failed(affineReshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = Shape(parseIntArrayAttr<int64_t>(affineReshape.getShapeValue()));
    const auto input = affineReshape.getInput();
    const auto inType = input.getType();
    const auto ndInType = mlir::cast<vpux::NDTypeInterface>(inType);
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outputLayout =
            Const::inferAffineReshapeOutputLayout(inOrder.toPermutation(), affineReshape.getDimMapping());
    if (!outputLayout.has_value()) {
        return mlir::failure();
    }

    auto typeComponents = TypeComponents().setShape(outShape).setDimsOrder(outputLayout.value());
    const auto elemTypeInferResult = inferElemType(affineReshape, ndInType.getElementType());
    if (mlir::succeeded(elemTypeInferResult)) {
        typeComponents = typeComponents.setElementType(elemTypeInferResult.value());
    }

    auto getOutputType = [&](NDTypeInterface type, const TypeComponents& components) -> NDTypeInterface {
        auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(type);
        if (distributedType == nullptr ||
            !VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedType.getDistribution())) {
            return type.changeTypeComponents(components);
        }

        auto origDistribution = distributedType.getDistribution();
        auto distribWithExplicitAttr = VPU::getNonOverlappedDistributedAttr(
                outShape, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getNumClusters(),
                origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(),
                distributedType.getElementType(), ctx);

        return distributedType.changeTypeComponentsForExplicitDistribution(components, distribWithExplicitAttr);
    };

    NDTypeInterface outType;
    if (auto sparseInputType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(ndInType)) {
        const NDTypeInterface dataType = sparseInputType.getData();
        outType = VPU::SparseTensorType::get(getOutputType(dataType, typeComponents));
    } else {
        outType = getOutputType(ndInType, typeComponents);
    }
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>>
vpux::VPU::AffineReshapeOp::inferCastedTypeAndDistribution(vpux::NDTypeInterface inType,
                                                           VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }

    // TODO: E-128707 - extend for other distribution modes
    if (distribution.getDistributionMode() != VPU::DistributionMode::DUPLICATED) {
        return mlir::failure();
    }

    const auto dstType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outShape = dstType.getShape();
    const auto dstElemType = dstType.getElementType();

    if (inType.getShape().size() != outShape.size()) {
        return mlir::failure();
    }

    if (!VPU::isDistributionWithExplicitShapesAndOffsets(distribution)) {
        const auto typeComponents =
                TypeComponents().setShape(outShape).setDimsOrder(dstType.getDimsOrder()).setElementType(dstElemType);
        return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)), distribution);
    }

    auto distribWithExplicitAttr = VPU::getNonOverlappedDistributedNative(
            outShape, distribution.getDistributionMode(), distribution.getNumTiles(), distribution.getNumClusters(),
            distribution.getAlignment(), distribution.hasUniformDistributedSegments(), dstElemType);
    const auto typeComponents =
            TypeComponents().setShape(outShape).setDimsOrder(dstType.getDimsOrder()).setElementType(dstElemType);
    return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)), distribWithExplicitAttr);
}

//
// TilingViewLikeOpInterface
//

namespace {

// Build reverse mapping: for each output dim, find which input dim(s) map to it.
// Returns a vector of size outputRank, where each element is a list of input dims that map to that output dim.
SmallVector<SmallVector<int64_t>> buildReverseDimMapping(ArrayRef<SmallVector<int64_t>> dimMapping,
                                                         int64_t outputRank) {
    SmallVector<SmallVector<int64_t>> reverseMap(outputRank);
    for (auto inputDim : irange(dimMapping.size())) {
        for (auto outputDim : dimMapping[inputDim]) {
            reverseMap[outputDim].push_back(checked_cast<int64_t>(inputDim));
        }
    }
    return reverseMap;
}

// Check if the given tiling dim has a simple 1:1 mapping (no merge/split) in the AffineReshape.
// A dim is "simple" if:
//   - It maps to exactly one output dim (not split)
//   - That output dim maps back to exactly one input dim (not merged)
//   - The size doesn't change between input and output on that dim
bool isSimpleDimMapping(ArrayRef<SmallVector<int64_t>> dimMapping, ArrayRef<SmallVector<int64_t>> reverseMap,
                        ShapeRef inputShape, ShapeRef outputShape, int64_t outDimIdx) {
    // The output dim must come from exactly one input dim
    if (reverseMap[outDimIdx].size() != 1) {
        return false;
    }
    const auto inputDimIdx = reverseMap[outDimIdx].front();
    // That input dim must map to exactly one output dim
    if (dimMapping[inputDimIdx].size() != 1) {
        return false;
    }
    // The sizes must match
    return inputShape[Dim(inputDimIdx)] == outputShape[Dim(outDimIdx)];
}

// Check if the output dim is a tileable dim of a split from a single input dim.
// Split pattern: one input dim maps to multiple consecutive output dims (dimMapping[i] = [j0, j1, ...]).
// Tiling is supported on:
//   - The outermost output dim j0 (split outer): tile with ratio scaling.
//     Example: dimMapping = [[0], [1], [2, 3], [3]], input H=256 splits to output H=64, W=4.
//             Output dim 2 is the split outer dim, ratio = 256/64 = 4.
//   - A non-front output dim jk when the front dim j0 has size 1 (split inner with outer=1):
//     Because inputOffset = 0 * innerSize + innerOffset = innerOffset, the ratio formula
//     still applies and produces ratio = inputSize / outputSize (which equals 1 for 2-way splits).
//     Example: dimMapping = [[0, 1], [2], [3]], input 320x64x4 -> output 1x320x64x4.
//             Output dim 1 (C=320): ratio = 320/320 = 1, tile transfers directly.
bool isSplitDim(ArrayRef<SmallVector<int64_t>> dimMapping, ArrayRef<SmallVector<int64_t>> reverseMap,
                ShapeRef outputShape, int64_t outDimIdx) {
    // Must come from exactly one input dim
    if (reverseMap[outDimIdx].size() != 1) {
        return false;
    }
    const auto inputDimIdx = reverseMap[outDimIdx].front();
    // That input dim must split to multiple output dims
    if (dimMapping[inputDimIdx].size() <= 1) {
        return false;
    }
    // This output dim must be the front (outermost) in the split, OR the front dim has size 1
    const auto frontDim = dimMapping[inputDimIdx].front();
    return frontDim == outDimIdx || outputShape[Dim(frontDim)] == 1;
}

// Check if the output dim is a merge of multiple input dims.
// Pure merge: multiple input dims each map exclusively to this single output dim.
// Merge with trailing ones: some contributing input dims also fan out to other output
// dims, but all those extra output dims have size 1 (trailing unsqueeze).
// Examples:
//   Pure merge: dimMapping = [[0], [0], [1], [2,3]], input 8x40 -> output 320.
//   Merge + trailing 1: dimMapping = [[0], [0, 1]], input 2x4 -> output 8x1.
//     in_d1 fans out to out_d0 (merge) and out_d1 (size 1), which is allowed.
bool isMergeDim(ArrayRef<SmallVector<int64_t>> dimMapping, ArrayRef<SmallVector<int64_t>> reverseMap,
                ShapeRef outputShape, int64_t outDimIdx) {
    // Must come from multiple input dims
    if (reverseMap[outDimIdx].size() <= 1) {
        return false;
    }
    // Each contributing input dim must map only to this output dim (pure merge),
    // or also to other output dims that are all size 1 (merge + trailing unsqueeze).
    return llvm::all_of(reverseMap[outDimIdx], [&](auto inDim) {
        if (dimMapping[inDim].size() == 1) {
            return true;
        }
        return llvm::all_of(dimMapping[inDim], [&](auto outDim) {
            return outDim == outDimIdx || outputShape[Dim(outDim)] == 1;
        });
    });
}

// For merge: find the first merged input dim with size > 1 (the tileable dim).
// Falls back to front() if all merged dims have size 1.
// Example: merge of [1, 256] -> 256: returns index 1 (size 256).
//          merge of [8, 40]  -> 320: returns index 0 (size 8).
int64_t getMergeTilingDimIdx(ShapeRef inputShape, ArrayRef<SmallVector<int64_t>> reverseMap, int64_t outDimIdx) {
    for (auto inDim : reverseMap[outDimIdx]) {
        if (inputShape[Dim(inDim)] > 1) {
            return inDim;
        }
    }
    return reverseMap[outDimIdx].front();
}

// For merge: compute the product of all merged input dims EXCEPT the tiling target.
// Example: merge of [1, 256] -> 256, target=1: otherProduct = 1.
//          merge of [8, 40]  -> 320, target=0: otherProduct = 40.
int64_t computeMergeOtherProduct(ShapeRef inputShape, ArrayRef<SmallVector<int64_t>> reverseMap, int64_t outDimIdx,
                                 int64_t targetDimIdx) {
    int64_t product = 1;
    for (auto inDim : reverseMap[outDimIdx]) {
        if (inDim != targetDimIdx) {
            product *= inputShape[Dim(inDim)];
        }
    }
    return product;
}

}  // anonymous namespace

// Helper to parse dim mapping and build reverse mapping from an AffineReshapeOp.
// Avoids repeating the same 4 lines of boilerplate in every interface method.
namespace {
struct AffineReshapeDimMappingInfo {
    ShapeRef inputShape;
    ShapeRef outputShape;
    SmallVector<SmallVector<int64_t>> dimMapping;
    SmallVector<SmallVector<int64_t>> reverseMap;

    explicit AffineReshapeDimMappingInfo(VPU::AffineReshapeOp op)
            : inputShape(getShape(op.getInput())),
              outputShape(getShape(op.getOutput())),
              dimMapping(parseIntArrayOfArrayAttr<int64_t>(op.getDimMapping())),
              reverseMap(buildReverseDimMapping(dimMapping, checked_cast<int64_t>(outputShape.size()))) {
    }
};

bool isDimMappingSupported(const AffineReshapeDimMappingInfo& info, int64_t outDimIdx) {
    return isSimpleDimMapping(info.dimMapping, info.reverseMap, info.inputShape, info.outputShape, outDimIdx) ||
           isSplitDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx) ||
           isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx);
}

}  // anonymous namespace

vpux::InputTiling vpux::VPU::AffineReshapeOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const AffineReshapeDimMappingInfo info(*this);

    TileInfo inputTile(info.inputShape);

    // For each output dimension that is tiled (axis != 1), map it back to the input
    for (auto dim : irange(info.outputShape.size())) {
        if (outputTile.axis[Dim(dim)] == 1) {
            continue;
        }

        const auto outDimIdx = checked_cast<int64_t>(dim);

        if (isSimpleDimMapping(info.dimMapping, info.reverseMap, info.inputShape, info.outputShape, outDimIdx)) {
            // Simple 1:1: directly transfer tile info
            const auto inputDimIdx = info.reverseMap[dim].front();
            inputTile.shape[Dim(inputDimIdx)] = outputTile.shape[Dim(dim)];
            inputTile.offsets[Dim(inputDimIdx)] = outputTile.offsets[Dim(dim)];
            inputTile.axis[Dim(inputDimIdx)] = outputTile.axis[Dim(dim)];
        } else if (isSplitDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
            // Split: scale by ratio = inputSize / outputSize (ratio=1 when outer dim has size 1)
            const auto inputDimIdx = info.reverseMap[dim].front();
            VPUX_THROW_WHEN(info.outputShape[Dim(dim)] == 0, "Invalid zero output shape at dim {0}", dim);
            const auto ratio = info.inputShape[Dim(inputDimIdx)] / info.outputShape[Dim(dim)];
            inputTile.shape[Dim(inputDimIdx)] = outputTile.shape[Dim(dim)] * ratio;
            inputTile.offsets[Dim(inputDimIdx)] = outputTile.offsets[Dim(dim)] * ratio;
            inputTile.axis[Dim(inputDimIdx)] = outputTile.axis[Dim(dim)];
        } else if (isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
            // Merge: find the first non-unit input dim and divide by the product of the rest
            const auto targetDimIdx = getMergeTilingDimIdx(info.inputShape, info.reverseMap, outDimIdx);
            const auto otherProduct =
                    computeMergeOtherProduct(info.inputShape, info.reverseMap, outDimIdx, targetDimIdx);
            VPUX_THROW_WHEN(otherProduct == 0, "Invalid zero other product for merge at dim {0}", dim);
            inputTile.shape[Dim(targetDimIdx)] = outputTile.shape[Dim(dim)] / otherProduct;
            inputTile.offsets[Dim(targetDimIdx)] = outputTile.offsets[Dim(dim)] / otherProduct;
            inputTile.axis[Dim(targetDimIdx)] = outputTile.axis[Dim(dim)];
            // Other merged dims retain full size (already set from inputShape initialization)
        } else {
            VPUX_THROW("AffineReshapeOp at '{0}': tiling on dim {1} is not supported", getLoc(), dim);
        }
    }

    return InputTiling{inputTile};
}

mlir::SmallVector<int64_t> vpux::VPU::AffineReshapeOp::backInferTilingStrategy(
        mlir::ArrayRef<int64_t> outputTilingStrategy) {
    const AffineReshapeDimMappingInfo info(*this);

    SmallVector<int64_t> inputTilingStrategy(info.inputShape.size(), 1);

    for (auto dim : irange(outputTilingStrategy.size())) {
        if (outputTilingStrategy[dim] <= 1) {
            continue;
        }
        const auto outDimIdx = checked_cast<int64_t>(dim);
        if (isDimMappingSupported(info, outDimIdx)) {
            const auto targetDim = isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)
                                           ? getMergeTilingDimIdx(info.inputShape, info.reverseMap, outDimIdx)
                                           : info.reverseMap[dim].front();
            inputTilingStrategy[targetDim] = outputTilingStrategy[dim];
        }
    }

    return inputTilingStrategy;
}

vpux::Dim vpux::VPU::AffineReshapeOp::backInferTilingDim(vpux::Dim outputDim) {
    const AffineReshapeDimMappingInfo info(*this);

    const auto outDimIdx = checked_cast<int64_t>(outputDim.ind());

    if (isDimMappingSupported(info, outDimIdx)) {
        if (isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
            return Dim(getMergeTilingDimIdx(info.inputShape, info.reverseMap, outDimIdx));
        }
        return Dim(info.reverseMap[outDimIdx].front());
    }

    // Fallback: return the same dim (should not be reached if isSupportedTilingDim is correct)
    return outputDim;
}

mlir::SmallVector<vpux::Dim> vpux::VPU::AffineReshapeOp::inferTilingDim(vpux::Dim inputDim) {
    const AffineReshapeDimMappingInfo info(*this);

    const auto inDimIdx = checked_cast<size_t>(inputDim.ind());
    SmallVector<Dim> outDims;
    // dimMapping[inputDim] gives the output dims this input dim maps to.
    // Filter to only those that are supported for tiling.
    for (auto outIdx : info.dimMapping[inDimIdx]) {
        if (isDimMappingSupported(info, outIdx)) {
            outDims.push_back(Dim(outIdx));
        }
    }
    return outDims;
}

mlir::FailureOr<mlir::SmallVector<int64_t>> vpux::VPU::AffineReshapeOp::inferTilingStrategy(
        mlir::ArrayRef<int64_t> inputTilingStrategy) {
    const AffineReshapeDimMappingInfo info(*this);

    if (inputTilingStrategy.size() != info.inputShape.size()) {
        return mlir::failure();
    }

    SmallVector<int64_t> outputTilingStrategy(info.outputShape.size(), 1);

    for (auto inDim : irange(inputTilingStrategy.size())) {
        if (inputTilingStrategy[inDim] <= 1) {
            continue;
        }

        auto outDims = inferTilingDim(Dim(checked_cast<int64_t>(inDim)));
        if (outDims.empty()) {
            return mlir::failure();
        }

        // For merge patterns, only the target input dim (first non-unit) is tileable.
        // Other merged input dims cannot be independently tiled through AffineReshape.
        for (const auto& outDim : outDims) {
            const auto outDimIdx = checked_cast<int64_t>(outDim.ind());
            if (isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
                const auto targetDimIdx = getMergeTilingDimIdx(info.inputShape, info.reverseMap, outDimIdx);
                if (checked_cast<int64_t>(inDim) != targetDimIdx) {
                    return mlir::failure();
                }
            }
        }

        // Prefer non-unit dims for strategy propagation when multiple output dims are mapped
        // from the same input dim (e.g. split-inner with outer=1 patterns).
        SmallVector<Dim> preferredOutDims;
        for (const auto& outDim : outDims) {
            if (info.outputShape[outDim] != 1) {
                preferredOutDims.push_back(outDim);
            }
        }

        auto& targetOutDims = preferredOutDims.empty() ? outDims : preferredOutDims;
        for (const auto& outDim : targetOutDims) {
            outputTilingStrategy[outDim.ind()] = inputTilingStrategy[inDim];
        }
    }

    return outputTilingStrategy;
}

void vpux::VPU::AffineReshapeOp::adjustAttrs(const TilingInfo&, const TileInfo& outputTile, ShapeRef) {
    // Update the shape_value attribute to reflect the tiled output shape
    auto newShape = getIntArrayAttr(getContext(), outputTile.shape);
    setShapeValueAttr(newShape);
}

bool vpux::VPU::AffineReshapeOp::isSupportedTilingDim(DimArrRef tilingDims) {
    if (tilingDims.empty()) {
        return true;
    }

    const AffineReshapeDimMappingInfo info(*this);

    // Check that each tiling dim is a supported pattern
    for (const auto& tilingDim : tilingDims) {
        const auto outDimIdx = checked_cast<int64_t>(tilingDim.ind());
        if (!isDimMappingSupported(info, outDimIdx)) {
            return false;
        }
    }
    return true;
}

bool vpux::VPU::AffineReshapeOp::isSupportedTilingDimWithRestrictions(Dim tilingDim) {
    VPUX_THROW_UNLESS(isSupportedTilingDim({tilingDim}), "AffineReshapeOp does not support tiling on dim {0}",
                      tilingDim);
    // Split outer and merge patterns have tiling restrictions (shape scaling/division).
    // Simple 1:1 and split inner with outer=1 have no restrictions (direct transfer, ratio=1).
    const AffineReshapeDimMappingInfo info(*this);
    const auto outDimIdx = checked_cast<int64_t>(tilingDim.ind());
    if (isSplitDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
        // Only true split-outer (front dim == tiling dim) has restrictions.
        // Split-inner-with-outer-1 has ratio=1, so divisibility is always satisfied.
        const auto inputDimIdx = info.reverseMap[outDimIdx].front();
        return info.dimMapping[inputDimIdx].front() == outDimIdx;
    }
    return isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx);
}

bool vpux::VPU::AffineReshapeOp::isSupportedOutTile(const TileInfo& outTile) {
    auto tilingDims = getNonOneDim(ShapeRef(outTile.axis));
    if (!isSupportedTilingDim(tilingDims)) {
        return false;
    }
    if (tilingDims.empty()) {
        return true;
    }

    const AffineReshapeDimMappingInfo info(*this);

    // Check each tiling dim for divisibility constraints
    for (const auto& tilingDim : tilingDims) {
        const auto outDimIdx = checked_cast<int64_t>(tilingDim.ind());

        if (isSimpleDimMapping(info.dimMapping, info.reverseMap, info.inputShape, info.outputShape, outDimIdx)) {
            // Simple dims have no constraints
            continue;
        }

        if (isSplitDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
            // Split: check that ratio produces integer input tile shape and offset
            const auto inputDimIdx = info.reverseMap[outDimIdx].front();
            if (info.outputShape[Dim(outDimIdx)] == 0) {
                return false;
            }
            if ((outTile.shape[tilingDim] * info.inputShape[Dim(inputDimIdx)]) % info.outputShape[Dim(outDimIdx)] !=
                0) {
                return false;
            }
            if ((outTile.offsets[tilingDim] * info.inputShape[Dim(inputDimIdx)]) % info.outputShape[Dim(outDimIdx)] !=
                0) {
                return false;
            }
            continue;
        }

        if (isMergeDim(info.dimMapping, info.reverseMap, info.outputShape, outDimIdx)) {
            // Merge: tile size and offset must be divisible by the product of non-target dims
            const auto targetDimIdx = getMergeTilingDimIdx(info.inputShape, info.reverseMap, outDimIdx);
            const auto otherProduct =
                    computeMergeOtherProduct(info.inputShape, info.reverseMap, outDimIdx, targetDimIdx);
            if (otherProduct == 0) {
                return false;
            }
            if (outTile.shape[tilingDim] % otherProduct != 0) {
                return false;
            }
            if (outTile.offsets[tilingDim] % otherProduct != 0) {
                return false;
            }
            continue;
        }

        // Unsupported pattern (should not reach here if isSupportedTilingDim is correct)
        return false;
    }
    return true;
}

mlir::OpFoldResult vpux::VPU::AffineReshapeOp::fold(FoldAdaptor adaptor) {
    // This op is view-like, which means that if input and output type are equal, it's a no-op.
    auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    if (inputType == outputType) {
        return getInput();
    }

    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput()); attr != nullptr) {
        return attr.transform().affineReshape(getDimMappingAttr(), getShapeValue()).get();
    }

    return nullptr;
}
