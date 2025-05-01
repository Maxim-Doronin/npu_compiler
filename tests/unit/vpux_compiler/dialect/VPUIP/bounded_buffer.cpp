//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include "common/utils.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr vpux::StringRef CMX_NAME = "CMX_NN";
constexpr vpux::StringRef DDR_NAME = "DDR";

}  // namespace

using MLIR_NDTypeInterface = MLIR_UnitBase;

TEST_F(MLIR_NDTypeInterface, BoundedBufferType) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    auto dataShape = Shape({1, 64, 13, 16});
    auto dataElementType = mlir::Float16Type::get(&ctx);
    auto dimsSpace = vpux::IndexedSymbolAttr::get(&ctx, CMX_NAME);
    auto dataBuffer = mlir::MemRefType::get(dataShape.raw(),  // shape
                                            dataElementType,  // elementType
                                            nullptr,          // orderAttr
                                            dimsSpace         // memorySpace
    );

    auto dynamicShape = Shape({4});
    auto dynamicShapeType = mlir::IntegerType::get(&ctx, 32);
    auto dynamicShapeBuffer = mlir::MemRefType::get(dynamicShape.raw(),  // shape
                                                    dynamicShapeType     // elementType
    );
    auto boundedBuffer = VPUIP::BoundedBufferType::get(dataBuffer, dynamicShapeBuffer);

    const auto ndTypeBoundedBuffer = mlir::dyn_cast<vpux::NDTypeInterface>(boundedBuffer);
    ASSERT_TRUE(ndTypeBoundedBuffer != nullptr) << "BoundedBuffer is not of vpux::NDTypeInterface type";

    const auto ndTypeData = mlir::dyn_cast<vpux::NDTypeInterface>(boundedBuffer.getData());
    ASSERT_TRUE(ndTypeData != nullptr) << "BoundedBuffer.getData() is not of vpux::NDTypeInterface type";
    const auto ndTypeDynamicShape = mlir::dyn_cast<vpux::NDTypeInterface>(boundedBuffer.getDynamicShape());
    ASSERT_TRUE(ndTypeDynamicShape != nullptr)
            << "BoundedBuffer.getDynamicShape() is not of vpux::NDTypeInterface type";

    EXPECT_EQ(ndTypeBoundedBuffer.getShape(), ndTypeData.getShape());
    EXPECT_EQ(ndTypeBoundedBuffer.getMemShape(), ndTypeData.getMemShape());

    EXPECT_EQ(ndTypeBoundedBuffer.hasRank(), ndTypeData.hasRank());
    EXPECT_EQ(ndTypeBoundedBuffer.getRank(), ndTypeData.getRank());
    EXPECT_EQ(ndTypeBoundedBuffer.getNumElements(), ndTypeData.getNumElements());
    EXPECT_EQ(ndTypeBoundedBuffer.getElementType(), ndTypeData.getElementType());
    EXPECT_EQ(ndTypeBoundedBuffer.getDimsOrder(), ndTypeData.getDimsOrder());
    EXPECT_EQ(ndTypeBoundedBuffer.getMemSpace(), ndTypeData.getMemSpace());
    EXPECT_EQ(ndTypeBoundedBuffer.getMemoryKind(), ndTypeData.getMemoryKind());

    const SmallVector<vpux::Bit> strides({212992_Bit, 3328_Bit, 256_Bit, 16_Bit});
    EXPECT_EQ(ndTypeBoundedBuffer.getStrides().raw(), strides);
    EXPECT_EQ(ndTypeBoundedBuffer.getMemStrides().raw(), strides);

    EXPECT_EQ(ndTypeBoundedBuffer.getTotalAllocSize(),
              ndTypeData.getTotalAllocSize() + ndTypeDynamicShape.getTotalAllocSize());
    EXPECT_EQ(ndTypeBoundedBuffer.getCompactAllocSize(),
              ndTypeData.getCompactAllocSize() + ndTypeDynamicShape.getCompactAllocSize());

    const SmallVector<int64_t> newShape({1, 32, 13});
    const auto changedShape = ndTypeBoundedBuffer.changeShape(vpux::ShapeRef(newShape));
    EXPECT_EQ(changedShape.getShape(), vpux::ShapeRef(newShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(mlir::cast<vpux::VPUIP::BoundedBufferType>(changedShape).getData())
                      .getShape(),
              vpux::ShapeRef(newShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(
                      mlir::cast<vpux::VPUIP::BoundedBufferType>(changedShape).getDynamicShape())
                      .getShape(),
              vpux::ShapeRef({3}));

    const auto changedElementType = ndTypeBoundedBuffer.changeElemType(mlir::Float32Type::get(&ctx));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedElementType.getElementType()));
    EXPECT_TRUE(mlir::isa<mlir::IntegerType>(
            mlir::cast<vpux::NDTypeInterface>(
                    mlir::cast<vpux::VPUIP::BoundedBufferType>(changedElementType).getDynamicShape())
                    .getElementType()));

    const auto changedShapeAndElementType =
            ndTypeBoundedBuffer.changeShapeElemType(vpux::ShapeRef(newShape), mlir::Float32Type::get(&ctx));
    EXPECT_EQ(changedShapeAndElementType.getShape(), vpux::ShapeRef(newShape));
    EXPECT_TRUE(mlir::isa<mlir::Float32Type>(changedShapeAndElementType.getElementType()));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(
                      mlir::cast<vpux::VPUIP::BoundedBufferType>(changedShapeAndElementType).getData())
                      .getShape(),
              vpux::ShapeRef(newShape));
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(
                      mlir::cast<vpux::VPUIP::BoundedBufferType>(changedShapeAndElementType).getDynamicShape())
                      .getShape(),
              vpux::ShapeRef({3}));

    const auto changedDimsOrder = ndTypeBoundedBuffer.changeDimsOrder(DimsOrder::NCHW);
    EXPECT_EQ(changedDimsOrder.getDimsOrder(), vpux::DimsOrder::NCHW);

    const auto changedMemoryKind = ndTypeBoundedBuffer.changeMemSpace(vpux::IndexedSymbolAttr::get(&ctx, DDR_NAME));
    EXPECT_EQ(changedMemoryKind.getMemoryKind(), vpux::VPU::MemoryKind::DDR);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(mlir::cast<vpux::VPUIP::BoundedBufferType>(changedMemoryKind).getData())
                      .getMemoryKind(),
              vpux::VPU::MemoryKind::DDR);
    EXPECT_EQ(mlir::cast<vpux::NDTypeInterface>(
                      mlir::cast<vpux::VPUIP::BoundedBufferType>(changedMemoryKind).getDynamicShape())
                      .getMemoryKind(),
              vpux::VPU::MemoryKind::DDR);

    const SmallVector<Bit> newStrides({851968_Bit, 13312_Bit, 1024_Bit, 16_Bit});
    const auto changedStrides = ndTypeBoundedBuffer.changeStrides(StridesRef(newStrides));
    EXPECT_EQ(changedStrides.getStrides().raw(), newStrides);
}
