//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/init/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using MLIR_TilingViewLikeOpInterfaceTest = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_TilingViewLikeOpInterfaceTest, ShapeCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x64x8xf16, {order=#NHWC}>) ->  tensor<1x1x64x128xf16, {order=#NHWC}>{
                %0 = VPU.ShapeCast {shape = [1, 1, 64, 128]} inputs(%arg0 : tensor<1x16x64x8xf16, {order=#NHWC}>) -> tensor<1x1x64x128xf16, {order=#NHWC}>
                return %0 : tensor<1x1x64x128xf16, {order=#NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::H));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::W));
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::C));
    {
        const auto divisor = SmallVector<int64_t>{1, 1, 1, 8};

        // roundtrip infer/back-infer tiling strategy
        const auto backInferStrategy = tilingViewLikeOp.backInferTilingStrategy(divisor);
        const auto expectedInputTilingStrategy = SmallVector<int64_t>{1, 1, 1, 8};
        EXPECT_EQ(backInferStrategy, expectedInputTilingStrategy);

        const auto inferredOutputDivisors = tilingViewLikeOp.inferTilingStrategy(backInferStrategy);
        ASSERT_TRUE(mlir::succeeded(inferredOutputDivisors));
        EXPECT_EQ(inferredOutputDivisors.value(), divisor);

        // Tile output on W into 8 pieces, and check if it is supported or not
        const auto divisorAsShape = Shape(divisor);
        auto tiles = fillDividedTiles(tilingViewLikeOp, divisorAsShape, getShape(tilingViewLikeOp->getResult(0)));
        ASSERT_TRUE(mlir::succeeded(tiles));
        for (const auto& tile : tiles.value()) {
            EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(tile));
        }
    }

    {
        // Tile output on W into 10 pieces, and check if it is supported or not
        Shape divisor = {1, 1, 1, 10};
        auto tiles = fillDividedTiles(tilingViewLikeOp, divisor, getShape(tilingViewLikeOp->getResult(0)));
        ASSERT_TRUE(mlir::succeeded(tiles));
        for (const auto& tile : tiles.value()) {
            EXPECT_FALSE(tilingViewLikeOp.isSupportedOutTile(tile));
        }
    }
}

TEST_F(MLIR_TilingViewLikeOpInterfaceTest, PermuteCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
        module @test {
            func.func @main(%arg0: tensor<1x256x40x1xf16, {order = #NHWC}>) -> tensor<1x40x1x256xf16, {order = #NCHW}> {
                %0 = VPU.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW}
                        : tensor<1x256x40x1xf16, {order = #NHWC}> -> tensor<1x40x1x256xf16, {order = #NCHW}>
                return %0 : tensor<1x40x1x256xf16, {order = #NCHW}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::H));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::W));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::C));

    auto log = Logger::global();
    {
        auto backInferTilingDim = tilingViewLikeOp.backInferTilingDim(Dims4D::Act::W);
        EXPECT_EQ(backInferTilingDim, Dims4D::Act::C);

        // Tile output on C into 8 pieces, and check the input tile
        const auto divisor = SmallVector<int64_t>{1, 8, 1, 1};

        const auto backInferStrategy = tilingViewLikeOp.backInferTilingStrategy(divisor);
        const auto expectedInputTilingStrategy = SmallVector<int64_t>{1, 1, 8, 1};
        EXPECT_EQ(backInferStrategy, expectedInputTilingStrategy);

        const auto inferredOutputDivisors = tilingViewLikeOp.inferTilingStrategy(backInferStrategy);
        ASSERT_TRUE(mlir::succeeded(inferredOutputDivisors));
        EXPECT_EQ(inferredOutputDivisors.value(), divisor);

        const auto divisorAsShape = Shape(divisor);
        auto tiles = fillDividedTiles(tilingViewLikeOp, divisorAsShape, getShape(tilingViewLikeOp->getResult(0)));
        ASSERT_TRUE(mlir::succeeded(tiles));

        const auto secondTile = tiles.value()[1];  // check the second tile to make sure the offset is correct
        auto inputTile = tilingViewLikeOp.backInferTileInfo(secondTile, log);
        const auto expectedInputTilingOffsets = Shape{0, 0, 5, 0};
        EXPECT_EQ(inputTile.tiles[0].offsets, expectedInputTilingOffsets);
        const auto expectedInputTilingSizes = Shape{1, 256, 5, 1};
        EXPECT_EQ(inputTile.tiles[0].shape, expectedInputTilingSizes);
    }
}

// AffineReshape with Simple 1:1 mapping on C dimension
// Input: 1x128x256x1, Output: 1x128x64x4, dim_mapping=[[0],[1],[2,3],[3]]
// C (dim 1): 128 -> 128, simple 1:1 mapping
TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_Simple) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x128x256x1xf16, {order=#NHWC}>) -> tensor<1x128x64x4xf16, {order=#NHWC}> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
                    : tensor<1x128x256x1xf16, {order=#NHWC}> -> tensor<1x128x64x4xf16, {order=#NHWC}>
                return %0 : tensor<1x128x64x4xf16, {order=#NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // C (dim 1) is simple 1:1 -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::C));
    // N (dim 0) is simple 1:1 -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::N));
    // H (dim 2) is split outer -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::H));
    // W (dim 3) is merge with fan-out -> NOT supported
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::W));

    // Test backInferTileInfo for simple dim C tiling
    // Output tile: shape=[1,64,64,4], offset=[0,0,0,0] (tiling C into 2)
    vpux::TileInfo outputTile({1, 64, 64, 4});
    outputTile.offsets = Shape({0, 0, 0, 0});
    outputTile.axis = Shape({1, 2, 1, 1});

    auto inputTiling = tilingViewLikeOp.backInferTileInfo(outputTile, Logger::global());
    ASSERT_EQ(inputTiling.tiles.size(), 1);
    const auto& inputTile = inputTiling.tiles.front();

    // Simple 1:1: input C tile = output C tile = 64
    // Input shape should be [1, 64, 256, 1]
    EXPECT_EQ(inputTile.shape[Dims4D::Act::N], 1);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::C], 64);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::H], 256);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::W], 1);

    // Second tile: offset=[0,64,0,0]
    vpux::TileInfo outputTile2({1, 64, 64, 4});
    outputTile2.offsets = Shape({0, 64, 0, 0});
    outputTile2.axis = Shape({1, 2, 1, 1});

    auto inputTiling2 = tilingViewLikeOp.backInferTileInfo(outputTile2, Logger::global());
    ASSERT_EQ(inputTiling2.tiles.size(), 1);
    const auto& inputTile2 = inputTiling2.tiles.front();

    EXPECT_EQ(inputTile2.offsets[Dims4D::Act::C], 64);
}

// AffineReshape with Split Outer dim on H dimension
// Input: 1x128x256x1, Output: 1x128x64x4, dim_mapping=[[0],[1],[2,3],[3]]
// H (dim 2): 256 splits to 64x4, output dim 2 is split outer
TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_SplitOuter) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x128x256x1xf16, {order=#NHWC}>) -> tensor<1x128x64x4xf16, {order=#NHWC}> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
                    : tensor<1x128x256x1xf16, {order=#NHWC}> -> tensor<1x128x64x4xf16, {order=#NHWC}>
                return %0 : tensor<1x128x64x4xf16, {order=#NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // H (dim 2) is split outer -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::H));

    // Test backInferTileInfo for split outer dim H tiling
    // Output tile: shape=[1,128,32,4], offset=[0,0,0,0] (tiling H into 2)
    // ratio = 256/64 = 4, so input H = 32 * 4 = 128
    vpux::TileInfo outputTile({1, 128, 32, 4});
    outputTile.offsets = Shape({0, 0, 0, 0});
    outputTile.axis = Shape({1, 1, 2, 1});

    auto inputTiling = tilingViewLikeOp.backInferTileInfo(outputTile, Logger::global());
    ASSERT_EQ(inputTiling.tiles.size(), 1);
    const auto& inputTile = inputTiling.tiles.front();

    // Split outer: input H = output H * ratio = 32 * 4 = 128
    EXPECT_EQ(inputTile.shape[Dims4D::Act::N], 1);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::C], 128);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::H], 128);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::W], 1);

    // Second tile: offset=[0,0,32,0] -> input offset = 32 * 4 = 128
    vpux::TileInfo outputTile2({1, 128, 32, 4});
    outputTile2.offsets = Shape({0, 0, 32, 0});
    outputTile2.axis = Shape({1, 1, 2, 1});

    auto inputTiling2 = tilingViewLikeOp.backInferTileInfo(outputTile2, Logger::global());
    ASSERT_EQ(inputTiling2.tiles.size(), 1);
    const auto& inputTile2 = inputTiling2.tiles.front();

    EXPECT_EQ(inputTile2.offsets[Dims4D::Act::H], 128);
    EXPECT_EQ(inputTile2.shape[Dims4D::Act::H], 128);

    // Test isSupportedOutTile - tile shape must produce integer input tile
    // Valid: 32 * 256 / 64 = 128 (integer)
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(outputTile));

    // Test with isSupportedTilingDimWithRestrictions
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDimWithRestrictions(Dims4D::Act::H));
}

// AffineReshape with Pure Merge on output dim 1
// Input: 1x8x40x1, Output: 1x320x1x1, dim_mapping=[[0],[1],[1],[2,3]]
// C,H merge: 8*40 = 320
TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_Merge) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x8x40x1xf16>) -> tensor<1x320x1x1xf16> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 320, 1, 1]}
                    : tensor<1x8x40x1xf16> -> tensor<1x320x1x1xf16>
                return %0 : tensor<1x320x1x1xf16>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // C (dim 1) is pure merge -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::C));
    // N (dim 0) is simple 1:1 -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::N));

    // Test backInferTileInfo for merge dim C tiling
    // Output tile: shape=[1,160,1,1], offset=[0,0,0,0] (tiling C=320 into 2)
    // innerProduct = inputShape[H] = 40
    // input C = 160 / 40 = 4
    vpux::TileInfo outputTile({1, 160, 1, 1});
    outputTile.offsets = Shape({0, 0, 0, 0});
    outputTile.axis = Shape({1, 2, 1, 1});

    auto inputTiling = tilingViewLikeOp.backInferTileInfo(outputTile, Logger::global());
    ASSERT_EQ(inputTiling.tiles.size(), 1);
    const auto& inputTile = inputTiling.tiles.front();

    // Merge: outerInputDim (C) = 160 / 40 = 4, innerDim (H) stays 40
    EXPECT_EQ(inputTile.shape[Dims4D::Act::N], 1);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::C], 4);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::H], 40);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::W], 1);

    // Second tile: offset=[0,160,0,0] -> input offset C = 160 / 40 = 4
    vpux::TileInfo outputTile2({1, 160, 1, 1});
    outputTile2.offsets = Shape({0, 160, 0, 0});
    outputTile2.axis = Shape({1, 2, 1, 1});

    auto inputTiling2 = tilingViewLikeOp.backInferTileInfo(outputTile2, Logger::global());
    ASSERT_EQ(inputTiling2.tiles.size(), 1);
    const auto& inputTile2 = inputTiling2.tiles.front();

    EXPECT_EQ(inputTile2.offsets[Dims4D::Act::C], 4);
    EXPECT_EQ(inputTile2.shape[Dims4D::Act::C], 4);

    // Test isSupportedOutTile - tile must be divisible by innerProduct=40
    // Valid: 160 % 40 = 0
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(outputTile));

    // Invalid tile: 100 % 40 != 0
    vpux::TileInfo invalidTile({1, 100, 1, 1});
    invalidTile.offsets = Shape({0, 0, 0, 0});
    invalidTile.axis = Shape({1, 2, 1, 1});
    EXPECT_FALSE(tilingViewLikeOp.isSupportedOutTile(invalidTile));
}

// AffineReshape with Split Inner dim where outer=1
// Input: 320x64x4 (3D), Output: 1x320x64x4 (4D), dim_mapping=[[0,1],[2],[3]]
// C (dim 1) = 320 is split inner, but outer N (dim 0) = 1
// This is valid because inputOffset = 0 * 320 + innerOffset = innerOffset
TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_SplitInnerWithOuterOne) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<320x64x4xf16>) -> tensor<1x320x64x4xf16> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 320, 64, 4]}
                    : tensor<320x64x4xf16> -> tensor<1x320x64x4xf16>
                return %0 : tensor<1x320x64x4xf16>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // C (dim 1) is split inner with outer=1 -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::C));
    // H (dim 2) is simple 1:1 -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::H));
    // W (dim 3) is simple 1:1 -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::W));
    // N (dim 0) is split outer -> supported
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::N));

    // Test backInferTileInfo for split inner dim C tiling (outer=1)
    // Output tile: shape=[1,160,64,4], offset=[0,0,0,0] (tiling C=320 into 2)
    // Since outer N=1, input tile transfers directly: input dim 0 = 160
    vpux::TileInfo outputTile({1, 160, 64, 4});
    outputTile.offsets = Shape({0, 0, 0, 0});
    outputTile.axis = Shape({1, 2, 1, 1});

    auto inputTiling = tilingViewLikeOp.backInferTileInfo(outputTile, Logger::global());
    ASSERT_EQ(inputTiling.tiles.size(), 1);
    const auto& inputTile = inputTiling.tiles.front();

    // Split inner with outer=1: input dim 0 = output C = 160
    EXPECT_EQ(inputTile.shape[Dim(0)], 160);
    EXPECT_EQ(inputTile.shape[Dim(1)], 64);
    EXPECT_EQ(inputTile.shape[Dim(2)], 4);

    // Second tile: offset=[0,160,0,0] -> input offset dim 0 = 160
    vpux::TileInfo outputTile2({1, 160, 64, 4});
    outputTile2.offsets = Shape({0, 160, 0, 0});
    outputTile2.axis = Shape({1, 2, 1, 1});

    auto inputTiling2 = tilingViewLikeOp.backInferTileInfo(outputTile2, Logger::global());
    ASSERT_EQ(inputTiling2.tiles.size(), 1);
    const auto& inputTile2 = inputTiling2.tiles.front();

    EXPECT_EQ(inputTile2.offsets[Dim(0)], 160);
    EXPECT_EQ(inputTile2.shape[Dim(0)], 160);

    // Test isSupportedOutTile - split inner with outer=1 has no constraints (direct transfer)
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(outputTile));

    // Test isSupportedTilingDimWithRestrictions - split inner with outer=1 has no restrictions
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDimWithRestrictions(Dims4D::Act::C));

    // Test backInferTilingDim
    // Output dim C -> Input dim 0 (the source dim of the split)
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dims4D::Act::C), Dim(0));
}

// AffineReshape with multi-dim tiling (C simple + H split outer)
TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_MultiDim) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x128x256x1xf16, {order=#NHWC}>) -> tensor<1x128x64x4xf16, {order=#NHWC}> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
                    : tensor<1x128x256x1xf16, {order=#NHWC}> -> tensor<1x128x64x4xf16, {order=#NHWC}>
                return %0 : tensor<1x128x64x4xf16, {order=#NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // Multi-dim tiling: C and H together should be supported
    SmallVector<vpux::Dim> multiDims = {Dims4D::Act::C, Dims4D::Act::H};
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(multiDims));

    // Multi-dim with unsupported W should fail
    SmallVector<vpux::Dim> invalidMultiDims = {Dims4D::Act::C, Dims4D::Act::W};
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDim(invalidMultiDims));

    // Test backInferTileInfo for multi-dim tiling (C=2, H=2)
    // Output tile: shape=[1,64,32,4], offset=[0,0,0,0]
    // C: simple, input C = 64
    // H: split outer, ratio=4, input H = 32 * 4 = 128
    vpux::TileInfo outputTile({1, 64, 32, 4});
    outputTile.offsets = Shape({0, 0, 0, 0});
    outputTile.axis = Shape({1, 2, 2, 1});

    auto inputTiling = tilingViewLikeOp.backInferTileInfo(outputTile, Logger::global());
    ASSERT_EQ(inputTiling.tiles.size(), 1);
    const auto& inputTile = inputTiling.tiles.front();

    EXPECT_EQ(inputTile.shape[Dims4D::Act::N], 1);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::C], 64);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::H], 128);
    EXPECT_EQ(inputTile.shape[Dims4D::Act::W], 1);

    // Test second tile: offset=[0,64,32,0]
    vpux::TileInfo outputTile2({1, 64, 32, 4});
    outputTile2.offsets = Shape({0, 64, 32, 0});
    outputTile2.axis = Shape({1, 2, 2, 1});

    auto inputTiling2 = tilingViewLikeOp.backInferTileInfo(outputTile2, Logger::global());
    ASSERT_EQ(inputTiling2.tiles.size(), 1);
    const auto& inputTile2 = inputTiling2.tiles.front();

    EXPECT_EQ(inputTile2.offsets[Dims4D::Act::C], 64);
    EXPECT_EQ(inputTile2.offsets[Dims4D::Act::H], 128);
}

// Test backInferTilingStrategy and backInferTilingDim
TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_BackInferStrategy) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x128x256x1xf16, {order=#NHWC}>) -> tensor<1x128x64x4xf16, {order=#NHWC}> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
                    : tensor<1x128x256x1xf16, {order=#NHWC}> -> tensor<1x128x64x4xf16, {order=#NHWC}>
                return %0 : tensor<1x128x64x4xf16, {order=#NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // Test backInferTilingStrategy
    // Output strategy [1, 2, 2, 1] -> Input strategy should be [1, 2, 2, 1]
    // C: simple, maps to input C
    // H: split outer, maps to input H
    SmallVector<int64_t> outputStrategy = {1, 2, 2, 1};
    auto inputStrategy = tilingViewLikeOp.backInferTilingStrategy(outputStrategy);

    EXPECT_EQ(inputStrategy.size(), 4);
    EXPECT_EQ(inputStrategy[0], 1);  // N
    EXPECT_EQ(inputStrategy[1], 2);  // C
    EXPECT_EQ(inputStrategy[2], 2);  // H
    EXPECT_EQ(inputStrategy[3], 1);  // W

    // Test backInferTilingDim
    // Output dim C -> Input dim C (simple)
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dims4D::Act::C), Dims4D::Act::C);
    // Output dim H -> Input dim H (split outer)
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dims4D::Act::H), Dims4D::Act::H);
}

TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_RankChangingInferTilingStrategy) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<256x2048x16x1x1xf16>) -> tensor<1x256x2048x16xf16> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 256, 2048, 16]}
                    : tensor<256x2048x16x1x1xf16> -> tensor<1x256x2048x16xf16>
                return %0 : tensor<1x256x2048x16xf16>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(0)}));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(1)}));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(2)}));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(3)}));

    auto inferredOutDims = tilingViewLikeOp.inferTilingDim(Dim(0));
    SmallVector<Dim> expectedOutDims = {Dim(0), Dim(1)};
    EXPECT_EQ(inferredOutDims, expectedOutDims);

    auto inferredOutputStrategy = tilingViewLikeOp.inferTilingStrategy(SmallVector<int64_t>{2, 1, 1, 1, 1});
    ASSERT_TRUE(mlir::succeeded(inferredOutputStrategy));
    SmallVector<int64_t> expectedOutputStrategy = {1, 2, 1, 1};
    EXPECT_EQ(inferredOutputStrategy.value(), expectedOutputStrategy);

    auto invalidOutputStrategy = tilingViewLikeOp.inferTilingStrategy(SmallVector<int64_t>{8, 1, 1, 1});
    EXPECT_TRUE(mlir::failed(invalidOutputStrategy));

    auto inputStrategyFromOutputC = tilingViewLikeOp.backInferTilingStrategy(SmallVector<int64_t>{1, 2, 1, 1});
    SmallVector<int64_t> expectedInputStrategyFromOutputC = {2, 1, 1, 1, 1};
    EXPECT_EQ(inputStrategyFromOutputC, expectedInputStrategyFromOutputC);

    auto inputStrategyFromOutputW = tilingViewLikeOp.backInferTilingStrategy(SmallVector<int64_t>{1, 1, 1, 2});
    SmallVector<int64_t> expectedInputStrategyFromOutputW = {1, 1, 2, 1, 1};
    EXPECT_EQ(inputStrategyFromOutputW, expectedInputStrategyFromOutputW);

    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dim(1)), Dim(0));
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dim(3)), Dim(2));

    vpux::TileInfo splitTile({1, 128, 2048, 16});
    splitTile.offsets = Shape({0, 128, 0, 0});
    splitTile.axis = Shape({1, 2, 1, 1});
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(splitTile));

    vpux::TileInfo mergeTile({1, 256, 2048, 8});
    mergeTile.offsets = Shape({0, 0, 0, 8});
    mergeTile.axis = Shape({1, 1, 1, 2});
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(mergeTile));
}

TEST_F(MLIR_TilingViewLikeOpInterfaceTest, AffineReshape_RankChanging4Dto5D_InferTilingStrategy) {
    // 4D -> 5D: tensor<1x256x2048x16xf16> -> tensor<256x2048x16x1x1xf16>
    // dim_mapping [[0], [0], [1], [2, 3, 4]]:
    //   Output dim 0 (256):  merge of input dims 0 (1) and 1 (256)
    //   Output dim 1 (2048): simple 1:1 from input dim 2
    //   Output dim 2 (16):   split front from input dim 3 (16 -> [16, 1, 1])
    //   Output dims 3,4 (1): split non-front, size=1, never tiled
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x256x2048x16xf16>) -> tensor<256x2048x16x1x1xf16> {
                %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [256, 2048, 16, 1, 1]}
                    : tensor<1x256x2048x16xf16> -> tensor<256x2048x16x1x1xf16>
                return %0 : tensor<256x2048x16x1x1xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    // isSupportedTilingDim: merge, simple, split-front are supported; split non-front with outer!=1 are not
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(0)}));   // merge
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(1)}));   // simple
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim({Dim(2)}));   // split front
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDim({Dim(3)}));  // split non-front, outer=16!=1
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDim({Dim(4)}));  // split non-front, outer=16!=1

    // backInferTilingDim: output dim -> input dim
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dim(0)), Dim(1));  // merge -> innermost non-unit input (C=256)
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dim(1)), Dim(2));  // simple -> input dim 2
    EXPECT_EQ(tilingViewLikeOp.backInferTilingDim(Dim(2)), Dim(3));  // split front -> input dim 3

    // inferTilingDim: input dim -> output dims
    auto outDimsFromDim1 = tilingViewLikeOp.inferTilingDim(Dim(1));
    SmallVector<Dim> expectedOutDimsFromDim1 = {Dim(0)};  // C=256 maps to output dim 0 (merge)
    EXPECT_EQ(outDimsFromDim1, expectedOutDimsFromDim1);

    auto outDimsFromDim3 = tilingViewLikeOp.inferTilingDim(Dim(3));
    SmallVector<Dim> expectedOutDimsFromDim3 = {Dim(2)};  // W=16 maps to output dim 2 (split front only)
    EXPECT_EQ(outDimsFromDim3, expectedOutDimsFromDim3);

    // inferTilingStrategy: 4D input strategy -> 5D output strategy
    // Tile on input C (dim 1) -> output merge dim 0
    auto inferredFromC = tilingViewLikeOp.inferTilingStrategy(SmallVector<int64_t>{1, 2, 1, 1});
    ASSERT_TRUE(mlir::succeeded(inferredFromC));
    SmallVector<int64_t> expectedFromC = {2, 1, 1, 1, 1};
    EXPECT_EQ(inferredFromC.value(), expectedFromC);

    // Tile on input W (dim 3) -> output split-front dim 2
    auto inferredFromW = tilingViewLikeOp.inferTilingStrategy(SmallVector<int64_t>{1, 1, 1, 2});
    ASSERT_TRUE(mlir::succeeded(inferredFromW));
    SmallVector<int64_t> expectedFromW = {1, 1, 2, 1, 1};
    EXPECT_EQ(inferredFromW.value(), expectedFromW);

    // Wrong-rank input strategy must fail
    auto invalidStrategy = tilingViewLikeOp.inferTilingStrategy(SmallVector<int64_t>{2, 1, 1, 1, 1});
    EXPECT_TRUE(mlir::failed(invalidStrategy));

    // backInferTilingStrategy: 5D output strategy -> 4D input strategy
    auto inputStrategyFromMerge = tilingViewLikeOp.backInferTilingStrategy(SmallVector<int64_t>{2, 1, 1, 1, 1});
    SmallVector<int64_t> expectedInputFromMerge = {1, 2, 1, 1};
    EXPECT_EQ(inputStrategyFromMerge, expectedInputFromMerge);

    auto inputStrategyFromSplit = tilingViewLikeOp.backInferTilingStrategy(SmallVector<int64_t>{1, 1, 2, 1, 1});
    SmallVector<int64_t> expectedInputFromSplit = {1, 1, 1, 2};
    EXPECT_EQ(inputStrategyFromSplit, expectedInputFromSplit);

    // isSupportedOutTile: merge tile on dim 0
    vpux::TileInfo mergeTile({128, 2048, 16, 1, 1});
    mergeTile.offsets = Shape({128, 0, 0, 0, 0});
    mergeTile.axis = Shape({2, 1, 1, 1, 1});
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(mergeTile));

    // isSupportedOutTile: split tile on dim 2
    vpux::TileInfo splitTile({256, 2048, 8, 1, 1});
    splitTile.offsets = Shape({0, 0, 8, 0, 0});
    splitTile.axis = Shape({1, 1, 2, 1, 1});
    EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(splitTile));
}
