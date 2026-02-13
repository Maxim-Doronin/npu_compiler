//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace vpux::VPU {
bool operator==(const TransformationsSplit& x, const TransformationsSplit& y) {
    return x.getContentAttr() == y.getContentAttr();
}
}  // namespace vpux::VPU

using namespace vpux;

struct MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo : MLIR_UnitBase {
public:
    MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Const::ConstDialect>();
    }

    template <typename Pred>
    std::vector<VPU::TransformationsSplit> extractSpecificSplits(mlir::ModuleOp moduleOp, Pred pred) {
        std::vector<VPU::TransformationsSplit> splits;
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            auto moveWorthy = VPU::collectMoveWorthyConstants(log, funcOp);
            std::copy_if(moveWorthy.begin(), moveWorthy.end(), std::back_inserter(splits), pred);
        });
        return splits;
    }

    std::vector<VPU::TransformationsSplit> extractSplits(mlir::ModuleOp moduleOp) {
        return extractSpecificSplits(moduleOp, [](const VPU::TransformationsSplit&) {
            return true;
        });
    }

    mlir::MLIRContext ctx;
    Logger log = Logger::global();
};

// Note: comparing transformations splits is complicated, so sort both ranges to
// simplify it
#define ASSERT_EQ_SPLITS(x, y)     \
    std::sort(x.begin(), x.end()); \
    std::sort(y.begin(), y.end()); \
    ASSERT_EQ(x, y)

constexpr llvm::StringLiteral INPUT_IR = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_3: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_4: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<102x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.PadWithZero<[0, 0], [100, 0]>]
        // ov1 = 2 * 2 * f16 + 102 * 2 * f16

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        %ov3_1 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_3> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        %ov3_2 = const.Declare tensor<52x2xf16> = dense_resource<vpux_ow_3> : tensor<2x2xf16>,
            [#const.PadWithZero<[0, 0], [50, 0]>]
        // ov3 = 2 * 2 * f16 + 52 * 2 * f16

        %ov4 = const.Declare tensor<2x2xf64> = dense_resource<vpux_ow_4> : tensor<2x2xf16>,
            [#const.CastElemType<f64>]
        // ov4 = 2 * 2 * f64

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, vpux::Byte(std::numeric_limits<int64_t>::max()));
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, CompletelySlicedInit) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    // Note: Memory limit = 0 means that every constant would be in its own
    // init. The nature of the procedure is such that even though the limit is
    // exceeded, *all* constants that share *the same dense_resource<>* are
    // guaranteed to appear in *the same* init. This is the intended behavior of
    // the algorithm.
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, vpux::Byte(0));
    ASSERT_EQ(allSlices.size(), 4);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, PartialSlicing) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    // Note: allow ov2 and ov4 to be stored together
    const auto justEnoughMemory =
            /* ov2 + ov4 inputs */ vpux::Byte(2 * 2 * sizeof(vpux::type::float16)) +
            vpux::Byte(2 * 2 * sizeof(vpux::type::float16)) +
            /* transformed(ov2) + transformed(ov4) outputs  */ vpux::Byte(2 * 2 * sizeof(vpux::type::float16)) +
            vpux::Byte(2 * 2 * sizeof(double));
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, vpux::Byte(justEnoughMemory));
    ASSERT_EQ(allSlices.size(), 3);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        std::vector<VPU::TransformationsSplit> expected;
        // Note: ov2 and ov4 are assumed to be stored together
        if (resourceName == "vpux_ow_2" || resourceName == "vpux_ow_4") {
            expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
                const auto xName = getResourceName(x.getContentAttr().getBaseContent()).str();
                return xName == "vpux_ow_2" || xName == "vpux_ow_4";
            });
        } else {
            expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
                return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
            });
        }

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_SUBVIEWS = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<1x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 0], [1, 2]>]
        %ov1_1 = const.Declare tensor<1x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[1, 0], [1, 2]>]
        %ov1_2 = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 0], [2, 1]>]
        %ov1_3 = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 1], [2, 1]>]
        // ov1 = 2 * 2 * f16 (subviews do not count)

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit_SubView) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_SUBVIEWS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 output (single) */ tensorSize2x2xf16 +
            /* ov2 output */ tensorSize2x2xf16;
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, justEnoughMemory);
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SlicedInit_SubView) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_SUBVIEWS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 output (single) */ tensorSize2x2xf16 +
            /* ov2 output */ tensorSize2x2xf16;
    // subtract 1 to disallow single init
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, justEnoughMemory - vpux::Byte(1));
    ASSERT_EQ(allSlices.size(), 2);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_REORDERS = R"(
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000ABABABABCDCDCDCD",
            vpux_ow_2: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<1x2xf16, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[0, 0], [1, 2]>]
        %ov1_1 = const.Declare tensor<1x2xf16, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[1, 0], [1, 2]>]
        %ov1_2 = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.PadWithZero<[0, 0], [0, 1]>, #const.SubView<[0, 0], [2, 1]>]
        // ov1 with reorder = 2 * 2 * f16 + 2 * 3 * f16 (subviews do not count)

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        return %arg0 : tensor<2x2xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SingleInit_ReorderAndPad) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_REORDERS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto expected = extractSplits(module.get());
    ASSERT_FALSE(expected.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 outputs */ tensorSize2x2xf16 + vpux::Byte(2 * 3 * sizeof(vpux::type::float16)) +
            /* ov2 output */ tensorSize2x2xf16;
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, expected, justEnoughMemory);
    ASSERT_EQ(allSlices.size(), 1);

    auto& actual = allSlices.front();
    ASSERT_EQ_SPLITS(actual, expected);
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_SplitInitAlgo, SlicedInit_ReorderAndPad) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_REORDERS, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto splits = extractSplits(module.get());
    ASSERT_FALSE(splits.empty());

    const auto tensorSize2x2xf16 = vpux::Byte(2 * 2 * sizeof(vpux::type::float16));
    const auto justEnoughMemory =
            /* ov1 input */ tensorSize2x2xf16 +
            /* ov2 input */ tensorSize2x2xf16 +
            /* ov1 outputs */ tensorSize2x2xf16 + vpux::Byte(2 * 3 * sizeof(vpux::type::float16)) +
            /* ov2 output */ tensorSize2x2xf16;
    auto allSlices = VPU::sliceAccordingToMemoryLimit(log, splits, justEnoughMemory - vpux::Byte(1));
    ASSERT_EQ(allSlices.size(), 2);

    for (auto& actual : allSlices) {
        ASSERT_FALSE(actual.empty());

        const auto resourceName = getResourceName(actual.front().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

struct MLIR_VPU_WeightsSeparationUtils_Obfuscation : MLIR_UnitBase {
    MLIR_VPU_WeightsSeparationUtils_Obfuscation(): MLIR_UnitBase() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Core::CoreDialect>();
        ctx.loadDialect<Const::ConstDialect>();
        ctx.loadDialect<IE::IEDialect>();
    }

    mlir::MLIRContext ctx;
    Logger log = Logger::global();

    static mlir::Operation* createSlice(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                        ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
        return builder.create<IE::SliceOp>(loc, input, offsets, sizes);
    }

    static mlir::Operation* createConcat(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<mlir::Value> inputs,
                                         int64_t axis) {
        return builder.create<IE::ConcatOp>(loc, inputs, axis);
    }
};

constexpr llvm::StringLiteral INPUT_IR_OBFUSCATION = R"(
module @main {
    func.func private @dummy(%a0: tensor<2xf32>) -> tensor<2xf32> {
        return %a0 : tensor<2xf32>
    }

    func.func @foo(%a0: tensor<1x1x1xf32>, %a1: tensor<1x1x2xf16>, %a2: tensor<4x2x1xsi8>,
                   %extra: tensor<2x3x4xf16>)
                   -> (tensor<1x1x1xsi8>, tensor<1x1x2xf32>, tensor<4x2x1xf16>, tensor<2x3x4xf16>) {
        %0 = IE.Convert(%a0) {dstElemType = si8} : tensor<1x1x1xf32> -> tensor<1x1x1xsi8>
        %1 = IE.Convert(%a1) {dstElemType = f32} : tensor<1x1x2xf16> -> tensor<1x1x2xf32>
        %2 = IE.Convert(%a2) {dstElemType = f16} : tensor<4x2x1xsi8> -> tensor<4x2x1xf16>
        return %0, %1, %2, %extra : tensor<1x1x1xsi8>, tensor<1x1x2xf32>, tensor<4x2x1xf16>, tensor<2x3x4xf16>
    }
})";

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Noop) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {0}, createSlice);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Partial) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {1, 2}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 1}, mlir::Float32Type::get(&ctx)),
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            // Note: new input is added as the last argument
            mlir::RankedTensorType::get({12}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateInputs_Full) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateInputs(log, appendLoc(op.getLoc(), "test"), op, {0, 1, 2, 3}, createSlice);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({64}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getInputs(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateOutputs_Noop) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *ops.begin();
    ASSERT_EQ(op.getSymName(), "dummy");

    VPU::obfuscateOutputs(log, appendLoc(op.getLoc(), "test"), op, {0}, createConcat);

    // test that nothing was changed
    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({2}, mlir::Float32Type::get(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getResults(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateOutputs_Partial) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateOutputs(log, appendLoc(op.getLoc(), "test"), op, {1, 2}, createConcat);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({1, 1, 1}, getSInt8Type(&ctx)),
            mlir::RankedTensorType::get({2, 3, 4}, mlir::Float16Type::get(&ctx)),
            // Note: new output is added as the last argument
            mlir::RankedTensorType::get({24}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getResults(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}

TEST_F(MLIR_VPU_WeightsSeparationUtils_Obfuscation, ObfuscateOutputs_Full) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(INPUT_IR_OBFUSCATION, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto ops = module->getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(ops.empty());
    auto op = *std::next(ops.begin());
    ASSERT_EQ(op.getSymName(), "foo");

    VPU::obfuscateOutputs(log, appendLoc(op.getLoc(), "test"), op, {0, 1, 2, 3}, createConcat);

    SmallVector<mlir::Type> expected = {
            mlir::RankedTensorType::get({73}, getInt8Type(&ctx)),
    };
    ASSERT_EQ(op.getFunctionType().getResults(), ArrayRef(expected));

    ASSERT_TRUE(mlir::succeeded(mlir::verify(op))) << "IR must be valid";
}
