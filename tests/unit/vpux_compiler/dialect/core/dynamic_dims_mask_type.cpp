//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
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

class MLIR_DynamicDimsMaskTensorTypeTest : public MLIR_UnitBase {  // NOLINT case style
public:
    MLIR_DynamicDimsMaskTensorTypeTest(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<mlir::tensor::TensorDialect, vpux::Core::CoreDialect, vpux::Const::ConstDialect>();
        listener = std::make_unique<mlir::OpBuilder::Listener>();
        builder = std::make_unique<mlir::OpBuilder>(&ctx, listener.get());
    }

    MLIR_DynamicDimsMaskTensorTypeTest(const MLIR_DynamicDimsMaskTensorTypeTest&) = delete;
    MLIR_DynamicDimsMaskTensorTypeTest& operator=(const MLIR_DynamicDimsMaskTensorTypeTest&) = delete;
    MLIR_DynamicDimsMaskTensorTypeTest(MLIR_DynamicDimsMaskTensorTypeTest&&) = delete;
    MLIR_DynamicDimsMaskTensorTypeTest& operator=(MLIR_DynamicDimsMaskTensorTypeTest&&) = delete;

    mlir::Type getDynamicTensorType() {
        const auto shape = SmallVector<int64_t>{1, 2, mlir::ShapedType::kDynamic, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr);

        return mlir::RankedTensorType::get(shape, elemType, tensorAttr);
    }

    mlir::Type getStaticTensorType() {
        const auto shape = SmallVector<int64_t>{1, 2, 10, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr);

        return mlir::RankedTensorType::get(shape, elemType, tensorAttr);
    }

    std::pair<mlir::Type, SmallVector<int64_t>> getTensorWithDynamicDimsMaskType() {
        const auto shape = SmallVector<int64_t>{1, 2, 10, 20};
        const auto elemType = mlir::Float32Type::get(&ctx);
        const auto order = DimsOrder::NCHW;
        const auto dimsMask = SmallVector<int64_t>{0, 0, 1, 1};
        const auto tensorAttr = getTensorAttr(&ctx, order, nullptr, Bounds(), DynamicDimsMask(dimsMask));
        const auto type = mlir::RankedTensorType::get(shape, elemType, tensorAttr);

        return std::make_pair(type, dimsMask);
    };

    mlir::MLIRContext ctx;
    std::unique_ptr<mlir::OpBuilder::Listener> listener;
    std::unique_ptr<mlir::OpBuilder> builder;
};

}  // namespace

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, NoMask) {
    auto type = getDynamicTensorType();

    const auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type);
    ASSERT_TRUE(dynamicDimsMaskType == nullptr);
}

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, ThrowWithDynamicShape) {
    auto type = getDynamicTensorType();

    const auto dimsMask = SmallVector<int64_t>{0, 1, 1, 0};
    EXPECT_THROW(Core::DynamicDimsMaskTensorType::get(type, dimsMask), std::exception);
}

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, GetDynamicDimsMask) {
    auto [type, dimsMask] = getTensorWithDynamicDimsMaskType();

    const auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type);
    ASSERT_TRUE(dynamicDimsMaskType != nullptr);

    const auto typeDimsMask = dynamicDimsMaskType.getDynamicDimsMask();
    ASSERT_TRUE(llvm::equal(typeDimsMask, dimsMask));
}

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, Get) {
    auto type = getStaticTensorType();
    const auto dimsMask = SmallVector<int64_t>{1, 1, 0, 0};
    const auto dynamicDimsMaskType = Core::DynamicDimsMaskTensorType::get(type, dimsMask);
    ASSERT_TRUE(dynamicDimsMaskType != nullptr);
    ASSERT_TRUE(llvm::equal(dynamicDimsMaskType.getDynamicDimsMask(), dimsMask));
}

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, ChangeDynamicDimsMask) {
    auto [type, dimsMask] = getTensorWithDynamicDimsMaskType();

    auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type);
    ASSERT_TRUE(dynamicDimsMaskType != nullptr);

    const auto newDimsMask = mlir::SmallVector<int64_t>{1, 1, 1, 1};
    ASSERT_TRUE(dimsMask != newDimsMask);

    const auto newType = dynamicDimsMaskType.changeDynamicDimsMask(newDimsMask);
    ASSERT_TRUE(llvm::equal(newType.getDynamicDimsMask(), newDimsMask));
}

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, ThrowChangeIncorrectMask) {
    auto [type, _] = getTensorWithDynamicDimsMaskType();

    auto dynMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type);
    ASSERT_TRUE(dynMaskType != nullptr);

    EXPECT_THROW([[maybe_unused]] auto _ = dynMaskType.changeDynamicDimsMask({10, 1, 0, 0}), std::exception);
    EXPECT_THROW([[maybe_unused]] auto _ = dynMaskType.changeDynamicDimsMask({1, 0}), std::exception);
    EXPECT_THROW([[maybe_unused]] auto _ = dynMaskType.changeDynamicDimsMask({0, 0, 0, 0}), std::exception);
}

TEST_F(MLIR_DynamicDimsMaskTensorTypeTest, ThrowGetWithIncorrectMask) {
    auto type = getStaticTensorType();
    EXPECT_THROW(Core::DynamicDimsMaskTensorType::get(type, {10, 1, 0, 0}), std::exception);
    EXPECT_THROW(Core::DynamicDimsMaskTensorType::get(type, {1, 1, 0, 0, 0}), std::exception);
    EXPECT_THROW(Core::DynamicDimsMaskTensorType::get(type, {0, 0, 0, 0}), std::exception);
}
