//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/reshape_utils.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"

#include <numeric>

using namespace vpux;

SmallVector<MemDimArr> getOutMemDimsCandidates(MemShapeRef inMemShape, MemShapeRef outMemShape, MemDim inMemDim) {
    const size_t targetDimSize = checked_cast<size_t>(inMemShape[inMemDim]);
    SmallVector<MemDimArr> outMemDims;
    // For Example: inMemShape: 1x512x512x16, outMemShape: 16x32x512x16, inMemDim: 'H'
    // The 'targetDimSize' will have two candidates: [[0, 1],[2]]
    // Candidate 1: [0, 1] input 'H' split into output 'N' and 'C'
    // Candidate 2: [2] input 'H' split into output 'H'
    for (size_t dimIdx = 0; dimIdx < outMemShape.size(); dimIdx++) {
        size_t accumulateSize = 1;
        size_t beginIdx = dimIdx;
        MemDimArr currMemDims;
        while (beginIdx < outMemShape.size()) {
            accumulateSize = accumulateSize * outMemShape[MemDim(beginIdx)];
            currMemDims.push_back(MemDim(beginIdx));
            if (accumulateSize == targetDimSize) {
                outMemDims.push_back(currMemDims);
            } else if (accumulateSize > targetDimSize) {
                break;
            }
            beginIdx++;
        }
    }
    return outMemDims;
}

std::optional<MemDimArr> vpux::deduceLegalOutputMemDims(MemShapeRef inMemShape, MemShapeRef outMemShape,
                                                        MemDim inMemDim) {
    const auto outMemDimsCandidates = getOutMemDimsCandidates(inMemShape, outMemShape, inMemDim);
    if (outMemDimsCandidates.empty()) {
        return std::nullopt;
    }

    auto getAccumulateSize = [](MemShapeRef memShape, auto beginIdx, auto endIdx) {
        VPUX_THROW_UNLESS(checked_cast<int32_t>(beginIdx) <= checked_cast<int32_t>(endIdx) &&
                                  memShape.begin() + endIdx <= memShape.end(),
                          "Got unexpect memShape");
        return std::accumulate(memShape.begin() + beginIdx, memShape.begin() + endIdx, int64_t(1),
                               std::multiplies<int64_t>());
    };

    // For Example: inMemShape: 1x512x512x16, outMemShape: 16x32x512x16, inMemDim: H
    // The 'outMemDims' will have two candidates: [[0, 1],[2]]
    // Candidate 1: outMemDims is [0, 1]
    // inTotalLeftSize(1x512) != outTotalLeftSize(1) && inTotalRightSize(16) != outTotalRightSize(512x16)
    // Candidate 2: outMemDims is [2]
    // inTotalLeftSize(1x512) == outTotalLeftSize(16x32) && inTotalRightSize(16) == outTotalRightSize(16)
    // The candidate 2 is legal candidate
    for (auto& outMemDims : outMemDimsCandidates) {
        const auto inTotalLeftShapeSize = getAccumulateSize(inMemShape, 0, inMemDim.ind());
        const auto inTotalRightShapeSize = getAccumulateSize(inMemShape, inMemDim.ind() + 1, inMemShape.size());
        const auto outTotalLeftShapeSize = getAccumulateSize(outMemShape, 0, outMemDims.front().ind());
        const auto outTotalRightShapeSize =
                getAccumulateSize(outMemShape, outMemDims.back().ind() + 1, outMemShape.size());
        if (inTotalLeftShapeSize == outTotalLeftShapeSize && inTotalRightShapeSize == outTotalRightShapeSize) {
            return outMemDims;
        }
    }
    return std::nullopt;
}

// If inputType has strides, infer the corresponding output type according to the output shape.
// If return 'std::nullopt', the output type cannot be inferred.
// This function only infer input stride only exist on one axis.
// Assume that there is a input MemShape [a, b, c, d] and MemStrides [2bcd, 2cd, 2d, 1], stridesDim: d2
// If output shape splits d2 into d3, cannot infer the output MemStride for d3 because it's not contiguous.
// If output shape splits d2 into d1, can infer the output MemStride for d1 because it's contiguous.
// This can be extended to the general case that the left product and right product split by stridesDim of input and
// output should be equal. The algorithm is as follows:
// 1. Find the 'stridesDim' in input type.
// 2. Split by 'stridesDim', input and output memory shape can be divided into two parts:
//    stridesMemDimLeftProduct: the product from highest dim to strided memDim
//    [inStridesMemDimLeftProduct,  inStridesMemDimRightProduct]
//    [outStridesMemDimLeftProduct, outStridesMemDimRightProduct]
//    inStridesMemDimLeftProduct should equal to outStridesMemDimLeftProduct
// 3. If stridesMemDimLeftProduct is equal to 1:
//    1x256x32x16, strides = [262144, 512, 16, 1] -> 256x32x16, strides = [512, 16, 1]
//    Although it seems that the corresponding strides info is lost, in NNDMA it will change its offset.
std::optional<Strides> inferReshapeOutputStrides(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    if (inType.getShape().totalSize() != outType.getShape().totalSize()) {
        return outType.getStrides();
    }

    const auto inStridesMemDims = VPUIP::getStridesMemDims(inType);
    if (inStridesMemDims.size() > 1) {
        return std::nullopt;
    }

    const auto outMemShape = outType.getMemShape();
    const auto outOrder = outType.getDimsOrder();
    const auto outElemSize = outType.getElemTypeSize();

    auto outMemStrides = StrideReqs::compact(outOrder.numDims()).calcStrides(outElemSize, outMemShape);

    if (inStridesMemDims.empty()) {
        return outOrder.toLogicalOrder(outMemStrides);
    }

    const auto inMemShape = inType.getMemShape();
    const auto inMemStrides = inType.getMemStrides();
    const auto inStridesMemDim = inStridesMemDims.front().ind();

    int64_t inStridesMemDimProduct = 1;
    for (size_t i = 0; i <= static_cast<size_t>(inStridesMemDim); i++) {
        inStridesMemDimProduct *= inMemShape.raw()[i];
    }

    int64_t outStridesMemDimProduct = 1;
    size_t outStridesDimRightBoundary = 0;
    const size_t outMemShapeSize = outMemShape.size();
    while (outStridesDimRightBoundary < outMemShapeSize) {
        outStridesMemDimProduct *= outMemShape.raw()[outStridesDimRightBoundary];
        if (outStridesMemDimProduct >= inStridesMemDimProduct) {
            break;
        }
        ++outStridesDimRightBoundary;
    }

    if (inStridesMemDimProduct != 1 && inStridesMemDimProduct != outStridesMemDimProduct) {
        return std::nullopt;
    }

    if (inStridesMemDimProduct == outStridesMemDimProduct) {
        outMemStrides.raw()[outStridesDimRightBoundary] = inMemStrides.raw()[inStridesMemDim];
        for (int64_t ind = static_cast<int64_t>(outStridesDimRightBoundary) - 1; ind >= 0; --ind) {
            const size_t currentMemDim = static_cast<size_t>(ind);
            const size_t prevMemDim = currentMemDim + 1;
            outMemStrides.raw()[currentMemDim] = outMemStrides.raw()[prevMemDim] * outMemShape.raw()[prevMemDim];
        }
    }

    return outOrder.toLogicalOrder(outMemStrides);
}

mlir::FailureOr<vpux::NDTypeInterface> vpux::updateStridesForReshape(const vpux::NDTypeInterface& inType,
                                                                     const vpux::NDTypeInterface& outType) {
    const auto outputStrides = inferReshapeOutputStrides(inType, outType);
    if (!outputStrides.has_value()) {
        return mlir::failure();
    }
    const auto outputStridesVal = outputStrides.value();
    return outType.getStrides() != outputStridesVal ? outType.changeStrides(outputStridesVal) : outType;
}

// This function checks if inType and outType strides are compatible for multi stride dims.
// If they have compatible strides, they should meet:
// Both memory strides [memStride0, memStrideForDim1, memStride1, ...]
// and memory shape [leftMemShapeProductForMemStride0, 1, leftMemShapeProductForMemStride1, ...]
// should be equal without dim size 1.
// The algorithm checks if the corresponding memStrides and leftMemShapeProduct are equal for stride dimension.
bool vpux::isInAndOutStridesCompatible(const vpux::NDTypeInterface& inType, const vpux::NDTypeInterface& outType) {
    const auto inStridesMemDims = VPUIP::getStridesMemDims(inType);
    const auto outStridesMemDims = VPUIP::getStridesMemDims(outType);
    const auto inMemShape = inType.getMemShape();
    const auto outMemShape = outType.getMemShape();
    const auto inMemStrides = inType.getMemStrides();
    const auto outMemStrides = outType.getMemStrides();

    // Store leftProduct and corresponding stride dim memStride
    struct ProductStrideInfo {
        int64_t product;
        Bit stride;

        bool operator==(const ProductStrideInfo& other) const {
            return product == other.product && stride == other.stride;
        }
    };

    SmallVector<ProductStrideInfo> leftProductsIn;
    SmallVector<ProductStrideInfo> leftProductsOut;

    const auto getLeftProducts = [](const auto& memShape, const auto& memStrides, const auto& stridesMemDims,
                                    auto& leftProducts) {
        int64_t leftProduct = 1;
        if (stridesMemDims.empty()) {
            return;
        }
        size_t cnt = 0;
        for (size_t i = 0; i < memShape.size(); ++i) {
            leftProduct *= memShape[MemDim(i)];
            if (cnt < stridesMemDims.size() && llvm::is_contained(stridesMemDims, MemDim(i))) {
                leftProducts.push_back({leftProduct, memStrides[MemDim(i)]});
                leftProduct = 1;
                ++cnt;
                continue;
            }
            if (cnt == stridesMemDims.size()) {
                break;
            }
        }
    };

    getLeftProducts(inMemShape, inMemStrides, inStridesMemDims, leftProductsIn);
    getLeftProducts(outMemShape, outMemStrides, outStridesMemDims, leftProductsOut);

    // Remove all elements with product equal to 1
    llvm::erase_if(leftProductsIn, [](const ProductStrideInfo& info) {
        return info.product == 1;
    });
    llvm::erase_if(leftProductsOut, [](const ProductStrideInfo& info) {
        return info.product == 1;
    });

    return leftProductsIn == leftProductsOut;
}
