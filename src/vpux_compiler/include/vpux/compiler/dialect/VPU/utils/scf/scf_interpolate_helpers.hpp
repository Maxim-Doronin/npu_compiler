//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#include <tuple>

namespace vpux::VPU {

// Transform output coordinate to input coordinate (affine expression equivalent)
// Mirrors: inferInCoord() in tiling.cpp
// Returns a tuple of affine expressions: {floor, ceil, round_prefer_ceil, round_prefer_floor} versions of
// inCoord = f(outCoord) based on coordMode
//
// Integer approximations for each mode (scale = inSize / outSize):
// - ASYMMETRIC: inCoord = outCoord * scale = outCoord * inSize / outSize
// - HALF_PIXEL: inCoord = scale * (outCoord + 0.5) - 0.5
//             = (outCoord * inSize + inSize/2 - outSize/2) / outSize
//             ~ (2 * outCoord * inSize + inSize - outSize) / (2 * outSize)
// - ALIGN_CORNERS: inCoord = outCoord * (inSize - 1) / (outSize - 1)
//
// Rounding modes:
// - round_prefer_ceil (round half up):   round(a/b) = floor((2*a + b) / (2*b))     - ties go up
// - round_prefer_floor (round half down): round(a/b) = floor((2*a + b - 1) / (2*b)) - ties go down
inline std::tuple<mlir::AffineExpr, mlir::AffineExpr, mlir::AffineExpr, mlir::AffineExpr> inferInCoordExprs(
        mlir::AffineExpr outCoord, IE::InterpolateCoordMode coordMode, int64_t initialInSize, int64_t initialOutSize) {
    switch (coordMode) {
    case IE::InterpolateCoordMode::ASYMMETRIC: {
        // inCoord = outCoord * inSize / outSize
        auto floorExpr = (outCoord * initialInSize).floorDiv(initialOutSize);
        auto ceilExpr = (outCoord * initialInSize).ceilDiv(initialOutSize);
        // round_prefer_ceil (round half up): round(a/b) = floor((2*a + b) / (2*b))
        auto roundPreferCeilExpr = (2 * outCoord * initialInSize + initialOutSize).floorDiv(2 * initialOutSize);
        // round_prefer_floor (round half down): round(a/b) = floor((2*a + b - 1) / (2*b))
        auto roundPreferFloorExpr = (2 * outCoord * initialInSize + initialOutSize - 1).floorDiv(2 * initialOutSize);
        return {floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr};
    }

    case IE::InterpolateCoordMode::HALF_PIXEL: {
        // inCoord = scale * (outCoord + 0.5) - 0.5
        auto floorExpr = (2 * outCoord * initialInSize + initialInSize - initialOutSize).floorDiv(2 * initialOutSize);
        auto ceilExpr = (2 * outCoord * initialInSize + initialInSize - initialOutSize).ceilDiv(2 * initialOutSize);
        // round_prefer_ceil: round(a/b) = floor((2*a + b) / (2*b))
        // where a = (2*inSize*outCoord + inSize - outSize), b = (2*outSize)
        // = floor((2*(2*inSize*outCoord + inSize - outSize) + 2*outSize) / (4*outSize))
        // = floor((4*inSize*outCoord + 2*inSize) / (4*outSize))
        // = floor((2*inSize*outCoord + inSize) / (2*outSize))
        auto roundPreferCeilExpr = (2 * outCoord * initialInSize + initialInSize).floorDiv(2 * initialOutSize);
        // round_prefer_floor: round(a/b) = floor((2*a + b - 1) / (2*b))
        // = floor((4*inSize*outCoord + 2*inSize - 1) / (4*outSize))
        auto roundPreferFloorExpr = (4 * outCoord * initialInSize + 2 * initialInSize - 1).floorDiv(4 * initialOutSize);
        return {floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr};
    }

    case IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL: {
        // Same as HALF_PIXEL, but returns 0 when outSize == 1
        if (initialOutSize == 1) {
            auto zeroExpr = getAffineConstantExpr(0, outCoord.getContext());
            return {zeroExpr, zeroExpr, zeroExpr, zeroExpr};
        }
        auto floorExpr = (2 * outCoord * initialInSize + initialInSize - initialOutSize).floorDiv(2 * initialOutSize);
        auto ceilExpr = (2 * outCoord * initialInSize + initialInSize - initialOutSize).ceilDiv(2 * initialOutSize);
        auto roundPreferCeilExpr = (2 * outCoord * initialInSize + initialInSize).floorDiv(2 * initialOutSize);
        auto roundPreferFloorExpr = (4 * outCoord * initialInSize + 2 * initialInSize - 1).floorDiv(4 * initialOutSize);
        return {floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr};
    }

    case IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN: {
        // inCoord = (outCoord + 0.5) * scale
        // Multiply by 2 to avoid fraction:
        //         = (2 * outCoord * inSize + inSize) / (2 * outSize)
        auto floorExpr = (2 * outCoord * initialInSize + initialInSize).floorDiv(2 * initialOutSize);
        auto ceilExpr = (2 * outCoord * initialInSize + initialInSize).ceilDiv(2 * initialOutSize);
        // round_prefer_ceil: round(a/b) = floor((2*a + b) / (2*b))
        // where a = (2*outCoord*inSize + inSize), b = (2*outSize)
        // = floor((4*outCoord*inSize + 2*inSize + 2*outSize) / (4*outSize))
        auto roundPreferCeilExpr =
                (4 * outCoord * initialInSize + 2 * initialInSize + 2 * initialOutSize).floorDiv(4 * initialOutSize);
        // round_prefer_floor: round(a/b) = floor((2*a + b - 1) / (2*b))
        // = floor((4*outCoord*inSize + 2*inSize + 2*outSize - 1) / (4*outSize))
        auto roundPreferFloorExpr = (4 * outCoord * initialInSize + 2 * initialInSize + 2 * initialOutSize - 1)
                                            .floorDiv(4 * initialOutSize);
        return {floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr};
    }

    case IE::InterpolateCoordMode::ALIGN_CORNERS: {
        // inCoord = outCoord * (inSize - 1) / (outSize - 1)
        if (initialOutSize > 1) {
            auto floorExpr = (outCoord * (initialInSize - 1)).floorDiv(initialOutSize - 1);
            auto ceilExpr = (outCoord * (initialInSize - 1)).ceilDiv(initialOutSize - 1);
            // round_prefer_ceil: round(a/b) = floor((2*a + b) / (2*b))
            auto roundPreferCeilExpr =
                    (2 * outCoord * (initialInSize - 1) + initialOutSize - 1).floorDiv(2 * (initialOutSize - 1));
            // round_prefer_floor: round(a/b) = floor((2*a + b - 1) / (2*b))
            auto roundPreferFloorExpr =
                    (2 * outCoord * (initialInSize - 1) + initialOutSize - 2).floorDiv(2 * (initialOutSize - 1));
            return {floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr};
        }
        auto zeroExpr = getAffineConstantExpr(0, outCoord.getContext());
        return {zeroExpr, zeroExpr, zeroExpr, zeroExpr};
    }

    default: {
        // Default: simple scaling (same as ASYMMETRIC)
        auto floorExpr = (outCoord * initialInSize).floorDiv(initialOutSize);
        auto ceilExpr = (outCoord * initialInSize).ceilDiv(initialOutSize);
        auto roundPreferCeilExpr = (2 * outCoord * initialInSize + initialOutSize).floorDiv(2 * initialOutSize);
        auto roundPreferFloorExpr = (2 * outCoord * initialInSize + initialOutSize - 1).floorDiv(2 * initialOutSize);
        return {floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr};
    }
    }
}

// Get nearest integer coordinate for the input (floor or ceil based on roundUp)
// Mirrors: getNearestCoord() in tiling.cpp
// Returns the input coordinate for a given output coordinate
inline mlir::AffineExpr getNearestCoordExpr(mlir::AffineExpr outCoord, IE::InterpolateMode interpolateMode,
                                            IE::InterpolateCoordMode coordMode, IE::InterpolateNearestMode nearestMode,
                                            int64_t initialInSize, int64_t initialOutSize, bool roundUp) {
    // Get floor, ceil, round_prefer_ceil, and round_prefer_floor expressions for the input coordinate
    auto [floorExpr, ceilExpr, roundPreferCeilExpr, roundPreferFloorExpr] =
            inferInCoordExprs(outCoord, coordMode, initialInSize, initialOutSize);

    if (interpolateMode == IE::InterpolateMode::LINEAR || interpolateMode == IE::InterpolateMode::LINEAR_ONNX) {
        // LINEAR/LINEAR_ONNX: floor for start, ceil for end
        if (roundUp) {
            return ceilExpr;
        }
        return floorExpr;
    } else if (interpolateMode == IE::InterpolateMode::CUBIC) {
        // CUBIC needs extra padding: floor(inCoord) - 1 for start, floor(inCoord) + 2 for end
        if (roundUp) {
            return floorExpr + 2;
        }
        return floorExpr - 1;
    } else if (interpolateMode == IE::InterpolateMode::NEAREST) {
        // NEAREST mode - apply rounding based on nearestMode
        // Reference: getNearestCoord() in tiling.cpp
        switch (nearestMode) {
        case IE::InterpolateNearestMode::ROUND_PREFER_CEIL:
            return roundPreferCeilExpr;

        case IE::InterpolateNearestMode::ROUND_PREFER_FLOOR:
            return roundPreferFloorExpr;

        case IE::InterpolateNearestMode::FLOOR:
            return floorExpr;

        case IE::InterpolateNearestMode::CEIL:
            return ceilExpr;

        case IE::InterpolateNearestMode::SIMPLE:
            // SIMPLE mode: scale > 1 (downscaling) uses ceil; scale <= 1 (upscaling) uses floor
            // scale = initialInSize / initialOutSize (backward scale)
            if (initialInSize > initialOutSize) {
                // Downscaling (backward scale > 1): uses ceil
                return ceilExpr;
            }

            // Upscaling (backward scale <= 1): uses floor
            return floorExpr;

        default:
            // Default: use floor (most common case)
            return floorExpr;
        }
    } else {
        if (roundUp) {
            return ceilExpr;
        }
        return floorExpr;
    }
}

// Propagate offset for a single dimension
// Mirrors: propagateOffsetForInterpolate() in tiling.cpp
// Returns the input coordinate for a given output coordinate
inline mlir::OpFoldResult propagateSCFOffsetForInterpolate(
        mlir::OpBuilder& builder, mlir::Location loc, mlir::OpFoldResult outOffset, IE::InterpolateMode interpolateMode,
        IE::InterpolateCoordMode coordMode, IE::InterpolateNearestMode nearestMode, int64_t initialInSize,
        int64_t initialOutSize, int64_t currentInSize, bool roundUp) {
    if (initialInSize == initialOutSize) {
        return outOffset;
    }

    mlir::AffineExpr d0;
    bindDims(builder.getContext(), d0);

    // Get the affine expression for coordinate transformation
    auto inCoordExpr =
            getNearestCoordExpr(d0, interpolateMode, coordMode, nearestMode, initialInSize, initialOutSize, roundUp);

    auto coordMap = mlir::AffineMap::get(1, 0, {inCoordExpr}, builder.getContext());
    auto inCoord = mlir::affine::makeComposedFoldedAffineApply(
            builder, appendLoc(loc, roundUp ? "inCoordEnd" : "inCoordStart"), coordMap, {outOffset});

    // Clamp to valid range [0, currentInSize - 1]
    mlir::AffineExpr s0;
    bindSymbols(builder.getContext(), s0);
    auto maxWithZeroMap = mlir::AffineMap::get(0, 1, {s0, builder.getAffineConstantExpr(0)}, builder.getContext());
    auto clampedMin =
            mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "clampMin"), maxWithZeroMap, {inCoord});

    auto maxVal = currentInSize - 1;
    auto minWithMaxMap = mlir::AffineMap::get(0, 1, {s0, builder.getAffineConstantExpr(maxVal)}, builder.getContext());
    auto clampedResult =
            mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "clampMax"), minWithMaxMap, {clampedMin});

    return clampedResult;
}

// Back-infer offset for all dimensions
// Mirrors: backInferOffsetForInterpolate() in tiling.cpp
inline SmallVector<mlir::OpFoldResult> backInferSCFOffsetForInterpolate(
        mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<mlir::OpFoldResult> outOffsets,
        IE::InterpolateMode interpolateMode, IE::InterpolateCoordMode coordMode, IE::InterpolateNearestMode nearestMode,
        ArrayRef<int64_t> initialInputDims, ArrayRef<int64_t> initialOutputDims, ArrayRef<int64_t> currentInputDims,
        bool roundUp) {
    SmallVector<mlir::OpFoldResult> inOffsets;
    inOffsets.reserve(outOffsets.size());

    for (size_t i = 0; i < outOffsets.size(); ++i) {
        auto inOffset = propagateSCFOffsetForInterpolate(builder, loc, outOffsets[i], interpolateMode, coordMode,
                                                         nearestMode, initialInputDims[i], initialOutputDims[i],
                                                         currentInputDims[i], roundUp);
        inOffsets.push_back(inOffset);
    }

    return inOffsets;
}

}  // namespace vpux::VPU
