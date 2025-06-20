//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

#include <algorithm>

namespace vpux::VPU {
bool operator==(const TransformationsSplit& x, const TransformationsSplit& y) {
    return x.declareOp().getContentAttr() == y.declareOp().getContentAttr();
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
    SmallVector<VPU::TransformationsSplit> extractSpecificSplits(mlir::ModuleOp moduleOp, Pred pred) {
        SmallVector<VPU::TransformationsSplit> splits;
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            auto moveWorthy = VPU::collectMoveWorthyTransformationSplits(log, funcOp);
            std::copy_if(moveWorthy.begin(), moveWorthy.end(), std::back_inserter(splits), pred);
        });
        return splits;
    }

    SmallVector<VPU::TransformationsSplit> extractSplits(mlir::ModuleOp moduleOp) {
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
            ov1: "0x10000000ABABABABCDCDCDCD",
            ov2: "0x10000000ABABABABCDCDCDCD",
            ov3: "0x10000000ABABABABCDCDCDCD",
            ov4: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<2x2xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<102x2xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.PadWithZero<[0, 0], [100, 0]>]
        // ov1 = 2 * 2 * f16 + 102 * 2 * f16

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<ov2> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        // ov2 = 2 * 2 * f16

        %ov3_1 = const.Declare tensor<2x2xf16> = dense_resource<ov3> : tensor<2x2xf16>,
            [#const.Rescale<5.0>]
        %ov3_2 = const.Declare tensor<52x2xf16> = dense_resource<ov3> : tensor<2x2xf16>,
            [#const.PadWithZero<[0, 0], [50, 0]>]
        // ov3 = 2 * 2 * f16 + 52 * 2 * f16

        %ov4 = const.Declare tensor<2x2xf64> = dense_resource<ov4> : tensor<2x2xf16>,
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

        const auto resourceName = getResourceName(actual.front().declareOp().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.declareOp().getContentAttr().getBaseContent()).str() == resourceName;
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

        const auto resourceName = getResourceName(actual.front().declareOp().getContentAttr().getBaseContent()).str();
        SmallVector<VPU::TransformationsSplit> expected;
        // Note: ov2 and ov4 are assumed to be stored together
        if (resourceName == "ov2" || resourceName == "ov4") {
            expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
                const auto xName = getResourceName(x.declareOp().getContentAttr().getBaseContent()).str();
                return xName == "ov2" || xName == "ov4";
            });
        } else {
            expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
                return getResourceName(x.declareOp().getContentAttr().getBaseContent()).str() == resourceName;
            });
        }

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_SUBVIEWS = R"(
{-#
    dialect_resources: {
        builtin: {
            ov1: "0x10000000ABABABABCDCDCDCD",
            ov2: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<1x2xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 0], [1, 2]>]
        %ov1_1 = const.Declare tensor<1x2xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[1, 0], [1, 2]>]
        %ov1_2 = const.Declare tensor<2x1xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 0], [2, 1]>]
        %ov1_3 = const.Declare tensor<2x1xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.SubView<[0, 1], [2, 1]>]
        // ov1 = 2 * 2 * f16 (subviews do not count)

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<ov2> : tensor<2x2xf16>,
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

        const auto resourceName = getResourceName(actual.front().declareOp().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.declareOp().getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}

constexpr llvm::StringLiteral INPUT_IR_REORDERS = R"(
{-#
    dialect_resources: {
        builtin: {
            ov1: "0x10000000ABABABABCDCDCDCD",
            ov2: "0x10000000ABABABABCDCDCDCD"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %ov1_0 = const.Declare tensor<1x2xf16, {order = #CN}> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[0, 0], [1, 2]>]
        %ov1_1 = const.Declare tensor<1x2xf16, {order = #CN}> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[1, 0], [1, 2]>]
        %ov1_2 = const.Declare tensor<2x1xf16> = dense_resource<ov1> : tensor<2x2xf16>,
            [#const.Add<1.0>, #const.PadWithZero<[0, 0], [0, 1]>, #const.SubView<[0, 0], [2, 1]>]
        // ov1 with reorder = 2 * 2 * f16 + 2 * 3 * f16 (subviews do not count)

        %ov2 = const.Declare tensor<2x2xf16> = dense_resource<ov2> : tensor<2x2xf16>,
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

        const auto resourceName = getResourceName(actual.front().declareOp().getContentAttr().getBaseContent()).str();
        auto expected = extractSpecificSplits(module.get(), [&](const VPU::TransformationsSplit& x) {
            return getResourceName(x.declareOp().getContentAttr().getBaseContent()).str() == resourceName;
        });

        ASSERT_EQ_SPLITS(actual, expected);
    }
}
