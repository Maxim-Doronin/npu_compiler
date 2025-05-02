//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/dialect.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <gtest/gtest.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <exception>

using namespace vpux;

namespace {

class MLIR_BoundedTensorTypeTest : public MLIR_UnitBase {  // NOLINT case style
public:
    MLIR_BoundedTensorTypeTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<mlir::tensor::TensorDialect, vpux::Core::CoreDialect, vpux::Const::ConstDialect>();
        listener = std::make_unique<mlir::OpBuilder::Listener>();
        builder = std::make_unique<mlir::OpBuilder>(&ctx, listener.get());
    }

    MLIR_BoundedTensorTypeTest(const MLIR_BoundedTensorTypeTest&) = delete;
    MLIR_BoundedTensorTypeTest& operator=(const MLIR_BoundedTensorTypeTest&) = delete;
    MLIR_BoundedTensorTypeTest(MLIR_BoundedTensorTypeTest&&) = delete;
    MLIR_BoundedTensorTypeTest& operator=(MLIR_BoundedTensorTypeTest&&) = delete;

    mlir::Type getStaticTensorType() {
        const auto shape = SmallVector<int64_t>{1, 2, 10, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr);

        return mlir::RankedTensorType::get(shape, elemType, tensorAttr);
    }

    std::pair<mlir::Type, SmallVector<int64_t>> getTensorWithBoundsType() {
        const auto shape = SmallVector<int64_t>{1, 2, mlir::ShapedType::kDynamic, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto bounds = SmallVector<int64_t>{1, 2, 10, 20};
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr, Bounds(bounds));
        const auto type = mlir::RankedTensorType::get(shape, elemType, tensorAttr);

        return std::make_pair(type, bounds);
    };

    mlir::MLIRContext ctx;
    std::unique_ptr<mlir::OpBuilder::Listener> listener;
    std::unique_ptr<mlir::OpBuilder> builder;
};

}  // namespace

TEST_F(MLIR_BoundedTensorTypeTest, NoBounds) {
    auto type = getStaticTensorType();

    const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    ASSERT_TRUE(boundedType == nullptr);
}

TEST_F(MLIR_BoundedTensorTypeTest, ThrowWithStaticShape) {
    auto type = getStaticTensorType();

    const auto bounds = SmallVector<int64_t>{1, 2, 10, 20};
    EXPECT_THROW(Core::BoundedTensorType::get(type, bounds), std::exception);
}

TEST_F(MLIR_BoundedTensorTypeTest, Get) {
    auto type = getStaticTensorType();
    auto ndType = mlir::cast<NDTypeInterface>(type);
    auto dynShapeType = ndType.changeShape(Shape(ndType.getShape().size(), mlir::ShapedType::kDynamic));

    const auto bounds = SmallVector<int64_t>{1, 2, 10, 20};
    const auto boundedType = Core::BoundedTensorType::get(dynShapeType, bounds);
    ASSERT_TRUE(boundedType != nullptr);
    ASSERT_TRUE(llvm::equal(boundedType.getBounds(), bounds));
}

TEST_F(MLIR_BoundedTensorTypeTest, GetBounds) {
    auto [type, bounds] = getTensorWithBoundsType();

    const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    ASSERT_TRUE(boundedType != nullptr);

    const auto typeBounds = boundedType.getBounds();
    ASSERT_TRUE(llvm::equal(typeBounds, bounds));
}

TEST_F(MLIR_BoundedTensorTypeTest, ChangeBounds) {
    auto [type, bounds] = getTensorWithBoundsType();

    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type);
    ASSERT_TRUE(boundedType != nullptr);

    const auto newBounds = mlir::SmallVector<int64_t>{1, 2, 20, 20};
    ASSERT_TRUE(bounds != newBounds);

    const auto newType = boundedType.changeBounds(newBounds);
    ASSERT_TRUE(llvm::equal(newType.getBounds(), newBounds));
}
