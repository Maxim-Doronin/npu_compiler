//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"

#include <gtest/gtest.h>

using namespace vpux;

#define EXPECT_SHAPE_EQ(act, ref)                                                                                    \
    const auto act##Ref = ref;                                                                                       \
    EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(act)>, std::decay_t<decltype(act##Ref)>>))                     \
            << "Mismatch between actual and reference types";                                                        \
    ASSERT_EQ(act.size(), act##Ref.size());                                                                          \
    for (size_t i = 0; i < act##Ref.size(); ++i) {                                                                   \
        EXPECT_EQ(act[Dim(i)], act##Ref[Dim(i)])                                                                     \
                << llvm::formatv("Value mismatch at dimension: {0}, (act={1} vs. ref={2})", i, act, act##Ref).str(); \
    }

//
// Dim Types Tests
//

TEST(MLIR_DynamicShape_BoundedDim, OperatorsDynamic) {
    const BoundedDim dimX{mlir::ShapedType::kDynamic, 64};
    const BoundedDim dimY{32};
    const auto addResult = dimX + dimY;
    const auto subResult = dimX - dimY;
    const auto mulResult = dimX * dimY;
    const MaskedDim maskedDim = addResult;

    EXPECT_EQ(addResult.reifiedSize(), 96);
    EXPECT_EQ(subResult.reifiedSize(), 32);
    EXPECT_EQ(mulResult.reifiedSize(), 2048);
    EXPECT_EQ(addResult.isDynamic(), true);
    EXPECT_EQ(subResult.isDynamic(), true);
    EXPECT_EQ(mulResult.isDynamic(), true);
    EXPECT_EQ(addResult.dimValue(), mlir::ShapedType::kDynamic);
    EXPECT_EQ(subResult.dimValue(), mlir::ShapedType::kDynamic);
    EXPECT_EQ(mulResult.dimValue(), mlir::ShapedType::kDynamic);
    EXPECT_EQ(addResult.representation(), 96);
    EXPECT_EQ(subResult.representation(), 32);
    EXPECT_EQ(mulResult.representation(), 2048);
    EXPECT_EQ(maskedDim.reifiedSize(), 96);
    EXPECT_EQ(maskedDim.isDynamic(), true);
}

TEST(MLIR_DynamicShape_BoundedDim, OperatorsStatic) {
    const BoundedDim dimX{64, 64};
    const BoundedDim dimY{32};
    const auto addResult = dimX + dimY;
    const auto subResult = dimX - dimY;
    const auto mulResult = dimX * dimY;
    const MaskedDim maskedDim = addResult;

    EXPECT_EQ(addResult.reifiedSize(), 96);
    EXPECT_EQ(subResult.reifiedSize(), 32);
    EXPECT_EQ(mulResult.reifiedSize(), 2048);
    EXPECT_EQ(addResult.isDynamic(), false);
    EXPECT_EQ(subResult.isDynamic(), false);
    EXPECT_EQ(mulResult.isDynamic(), false);
    EXPECT_EQ(addResult.dimValue(), 96);
    EXPECT_EQ(subResult.dimValue(), 32);
    EXPECT_EQ(mulResult.dimValue(), 2048);
    EXPECT_EQ(addResult.representation(), 96);
    EXPECT_EQ(subResult.representation(), 32);
    EXPECT_EQ(mulResult.representation(), 2048);
    EXPECT_EQ(maskedDim.reifiedSize(), 96);
    EXPECT_EQ(maskedDim.isDynamic(), false);
}

TEST(MLIR_DynamicShape_BoundedDim, Invalid) {
    EXPECT_THROW((BoundedDim{32, 64}), std::exception);
    EXPECT_THROW((BoundedDim{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}), std::exception);
    EXPECT_THROW((BoundedDim{mlir::ShapedType::kDynamic, 32} - BoundedDim{64}), std::exception);
}

TEST(MLIR_DynamicShape_MaskedDim, OperatorsDynamic) {
    const MaskedDim dimX{64, true};
    const MaskedDim dimY{32};
    const auto addResult = dimX + dimY;
    const auto subResult = dimX - dimY;
    const auto mulResult = dimX * dimY;
    const BoundedDim boundedDim = addResult;

    EXPECT_EQ(addResult.reifiedSize(), 96);
    EXPECT_EQ(subResult.reifiedSize(), 32);
    EXPECT_EQ(mulResult.reifiedSize(), 2048);
    EXPECT_EQ(addResult.isDynamic(), true);
    EXPECT_EQ(subResult.isDynamic(), true);
    EXPECT_EQ(mulResult.isDynamic(), true);
    EXPECT_EQ(addResult.dimValue(), mlir::ShapedType::kDynamic);
    EXPECT_EQ(subResult.dimValue(), mlir::ShapedType::kDynamic);
    EXPECT_EQ(mulResult.dimValue(), mlir::ShapedType::kDynamic);
    EXPECT_EQ(addResult.representation(), true);
    EXPECT_EQ(subResult.representation(), true);
    EXPECT_EQ(mulResult.representation(), true);
    EXPECT_EQ(boundedDim.reifiedSize(), 96);
    EXPECT_EQ(boundedDim.isDynamic(), true);
}

TEST(MLIR_DynamicShape_MaskedDim, OperatorsStatic) {
    const MaskedDim dimX{64, false};
    const MaskedDim dimY{32};
    const auto addResult = dimX + dimY;
    const auto subResult = dimX - dimY;
    const auto mulResult = dimX * dimY;
    const BoundedDim boundedDim = addResult;

    EXPECT_EQ(addResult.reifiedSize(), 96);
    EXPECT_EQ(subResult.reifiedSize(), 32);
    EXPECT_EQ(mulResult.reifiedSize(), 2048);
    EXPECT_EQ(addResult.isDynamic(), false);
    EXPECT_EQ(subResult.isDynamic(), false);
    EXPECT_EQ(mulResult.isDynamic(), false);
    EXPECT_EQ(addResult.dimValue(), 96);
    EXPECT_EQ(subResult.dimValue(), 32);
    EXPECT_EQ(mulResult.dimValue(), 2048);
    EXPECT_EQ(addResult.representation(), false);
    EXPECT_EQ(subResult.representation(), false);
    EXPECT_EQ(mulResult.representation(), false);
    EXPECT_EQ(boundedDim.reifiedSize(), 96);
    EXPECT_EQ(boundedDim.isDynamic(), false);
}

TEST(MLIR_DynamicShape_MaskedDim, Invalid) {
    EXPECT_THROW((MaskedDim{mlir::ShapedType::kDynamic}), std::exception);
    EXPECT_THROW((MaskedDim{32} - MaskedDim{64, true}), std::exception);
}

//
// Shapes Tests
//

TEST(MLIR_DynamicShape_BoundedShape, Conversions) {
    const BoundedShape boundedShape{{1}, {mlir::ShapedType::kDynamic, 64}, {256}, {mlir::ShapedType::kDynamic, 512}};

    const auto shape = boundedShape.toShape();
    const auto reifiedShape = boundedShape.toReifiedShape();
    const auto bounds = boundedShape.toRepresentation();
    const auto dimsMask = boundedShape.toRepresentationOf<DimsMaskedShape>();
    const auto totalSize = boundedShape.totalSize();
    const auto dimsMaskedShape = makeShape<DimsMaskedShape>(reifiedShape, dimsMask);

    EXPECT_SHAPE_EQ(shape, (Shape{1, mlir::ShapedType::kDynamic, 256, mlir::ShapedType::kDynamic}));
    EXPECT_SHAPE_EQ(reifiedShape, (Shape{1, 64, 256, 512}));
    EXPECT_SHAPE_EQ(bounds, (Bounds{1, 64, 256, 512}));
    EXPECT_SHAPE_EQ(dimsMask, (DynamicDimsMask{false, true, false, true}));
    EXPECT_EQ(totalSize, 64 * 256 * 512);
    EXPECT_SHAPE_EQ(dimsMaskedShape, (DimsMaskedShape{{1}, {64, true}, {256}, {512, true}}));
}

TEST(MLIR_DynamicShape_DimsMaskedShape, Conversions) {
    const DimsMaskedShape dimsMaskedShape{{1}, {64, true}, {256}, {512, true}};

    const auto shape = dimsMaskedShape.toShape();
    const auto reifiedShape = dimsMaskedShape.toReifiedShape();
    const auto dimsMask = dimsMaskedShape.toRepresentation();
    const auto bounds = dimsMaskedShape.toRepresentationOf<BoundedShape>();
    const auto totalSize = dimsMaskedShape.totalSize();
    const auto boundedShape = makeShape<BoundedShape>(shape, bounds);

    EXPECT_SHAPE_EQ(shape, (Shape{1, mlir::ShapedType::kDynamic, 256, mlir::ShapedType::kDynamic}));
    EXPECT_SHAPE_EQ(reifiedShape, (Shape{1, 64, 256, 512}));
    EXPECT_SHAPE_EQ(bounds, (Bounds{1, 64, 256, 512}));
    EXPECT_SHAPE_EQ(dimsMask, (DynamicDimsMask{false, true, false, true}));
    EXPECT_EQ(totalSize, 64 * 256 * 512);
    EXPECT_SHAPE_EQ(boundedShape,
                    (BoundedShape{{1}, {mlir::ShapedType::kDynamic, 64}, {256}, {mlir::ShapedType::kDynamic, 512}}));
}
