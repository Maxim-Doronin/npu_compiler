//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/nce_utils.hpp"
#include "common/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>
#include <mlir/Support/LLVM.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using namespace vpux;

class MLIR_SchedulerLoopCreationTest : public testing::TestWithParam<config::ArchKind> {
protected:
    void SetUp() override {
        registry = vpux::createDialectRegistry();
        auto interfacesRegistry = vpux::createInterfacesRegistry(GetParam());
        interfacesRegistry->registerInterfaces(registry);
        VPU::initializeSingletons(registry, VPU::DeviceVersion{std::nullopt, GetParam()});

        ctx = std::make_unique<mlir::MLIRContext>(registry);
        ctx->appendDialectRegistry(registry);
        ctx->loadDialect<VPUIP::VPUIPDialect>();
        ctx->loadDialect<vpux::VPU::VPUDialect>();
    }

    mlir::MLIRContext* getCtx() {
        return ctx.get();
    }

private:
    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;
};

VPU::MPEEngineAttr createMPEEngineAttr(mlir::MLIRContext* ctx, [[maybe_unused]] config::ArchKind arch) {
    return VPU::MPEEngine37XXAttr::get(ctx, VPU::MPEEngine37XXModeAttr::get(ctx, VPU::MPEEngine37XXMode::SCL));
}

VPUIP::NCEClusterTaskOp createNCEClusterTaskOp(mlir::OpBuilder& builder, mlir::MLIRContext* ctx, mlir::Location loc,
                                               int64_t kernel, int64_t padding, int64_t stride, mlir::Value inputTile,
                                               mlir::Value weightOp, mlir::Value weightTableOp,
                                               mlir::memref::AllocOp outputTile, mlir::Type outputTileType,
                                               VPU::MPEEngineAttr mpeEngineAttr) {
    auto paddingAttr =
            vpux::VPU::PaddingAttr::get(ctx, builder.getI64IntegerAttr(padding), builder.getI64IntegerAttr(padding),
                                        builder.getI64IntegerAttr(padding), builder.getI64IntegerAttr(padding));
    auto kernelSize = builder.getI64ArrayAttr({kernel, kernel});
    auto kernelStrides = builder.getI64ArrayAttr({stride, stride});
    auto nceOp = builder.create<VPUIP::NCEClusterTaskOp>(
            loc, outputTileType, nullptr, nullptr, inputTile, nullptr, nullptr, weightOp, nullptr, weightTableOp,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, inputTile, nullptr, nullptr,
            outputTile.getResult(), nullptr, mlir::ValueRange(), outputTile.getResult(), nullptr, nullptr, nullptr,
            nullptr, nullptr, mlir::ValueRange(), VPUIP::NCETaskType::CONV, kernelSize, kernelStrides, paddingAttr,
            false, nullptr, false, nullptr, false, false, false, nullptr, nullptr, nullptr, false, false, mpeEngineAttr,
            nullptr, nullptr, nullptr, nullptr);
    return nceOp;
}

mlir::Value createWeightsTable(mlir::OpBuilder& builder, mlir::Location loc, int64_t tileC,
                               const vpux::IndexedSymbolAttr& ddrSpace, const vpux::IndexedSymbolAttr& cmxSpace) {
    auto weightTableTypeDDR =
            vpux::getMemRefType({tileC, 1, 1, 4}, builder.getIntegerType(32, true), DimsOrder::OIYX, ddrSpace);
    auto weightTableTypeCMX =
            vpux::getMemRefType({tileC, 1, 1, 4}, builder.getIntegerType(32, true), DimsOrder::OIYX, cmxSpace);
    auto weightTableTensorType = mlir::RankedTensorType::get({tileC, 1, 1, 4}, builder.getIntegerType(32, true));

    auto weightTableAttr = mlir::DenseElementsAttr::get(weightTableTensorType, mlir::APInt(32, 1, true));
    auto weightTableDDR = builder.create<vpux::Const::DeclareOp>(loc, weightTableTypeDDR,
                                                                 vpux::Const::ContentAttr::get(weightTableAttr));
    auto weightTableCMX = builder.create<mlir::memref::AllocOp>(loc, weightTableTypeCMX);
    auto copyWeightsTable = builder.create<VPUIP::NNDMAOp>(loc, weightTableDDR, weightTableCMX);

    return copyWeightsTable.getOutput();
}

mlir::Value createWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemType, ShapeRef weightsShape,
                          const vpux::IndexedSymbolAttr& ddrSpace, const vpux::IndexedSymbolAttr& cmxSpace) {
    auto weightsTypeDDR = vpux::getMemRefType(weightsShape, elemType, DimsOrder::NHWC, ddrSpace);
    auto weightsTypeCMX = vpux::getMemRefType(weightsShape, elemType, DimsOrder::NHWC, cmxSpace);

    auto weightsTensorType = mlir::RankedTensorType::get(weightsShape, elemType);

    Const::ContentSetup contentAttrSetup(weightsTensorType);
    contentAttrSetup = contentAttrSetup.castElemType(elemType);
    contentAttrSetup = contentAttrSetup.reorder(DimsOrder::NHWC);
    auto weightsAttr = mlir::DenseElementsAttr::get(weightsTensorType, llvm::APFloat(mlir::APFloat::IEEEhalf(), "1.0"));
    auto contentAttr = Const::ContentAttr::get(weightsAttr, contentAttrSetup);

    auto weightsDDR = builder.create<Const::DeclareOp>(loc, weightsTypeDDR, contentAttr).getOutput();
    auto weightsCMX = builder.create<mlir::memref::AllocOp>(loc, weightsTypeCMX);
    auto copyWeights = builder.create<VPUIP::NNDMAOp>(loc, weightsDDR, weightsCMX);

    return copyWeights.getOutput();
}

// Helper function to create a tiled convolution test
mlir::OwningOpRef<mlir::ModuleOp> createTiledConvolutionModule(mlir::MLIRContext* ctx, int numTilesH, int numTilesC,
                                                               config::ArchKind arch) {
    auto loc = mlir::UnknownLoc::get(ctx);
    auto module = mlir::ModuleOp::create(loc);
    auto builder = mlir::OpBuilder(module.getBody(), module.getBody()->begin());

    // setup
    const DimsOrder orderNHWC = DimsOrder::NHWC;
    const auto cmxSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    const auto ddrSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::DDR), 0);
    const auto f16Type = mlir::Float16Type::get(ctx);

    // Hardcoded test parameters for simplicity. Move to parameters later if needed.
    // They should be divisible by numTilesH and numTilesC, otherwise test logic will be more complicated.
    const int64_t inputSizeH = 128;
    const int64_t inputSizeW = 64;
    const int64_t inputChannels = 16;
    const int64_t outputChannels = 128;
    const int64_t kernelSize = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;

    // Actual test parameters are number of tiles in H and C dimensions
    const int64_t tileH = inputSizeH / numTilesH;
    const int64_t remH = inputSizeH % numTilesH;
    VPUX_THROW_UNLESS(remH == 0, "Input height {0} is not divisible by numTilesH {1}", inputSizeH, numTilesH);
    const int64_t tileC = outputChannels / numTilesC;
    const int64_t remC = outputChannels % numTilesC;
    VPUX_THROW_UNLESS(remC == 0, "Output channels {0} is not divisible by numTilesC {1}", outputChannels, numTilesC);

    // Create function
    auto inputTypeDDR = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto outputTypeDDR = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, ddrSpace);
    auto inputTypeCMX = vpux::getMemRefType({1, inputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
    auto outputTypeCMX = vpux::getMemRefType({1, outputChannels, inputSizeH, inputSizeW}, f16Type, orderNHWC, cmxSpace);

    auto funcType = builder.getFunctionType({inputTypeDDR, outputTypeDDR}, {outputTypeDDR});
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    func.setPublic();

    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto inputArg = entryBlock->getArgument(0);
    auto outputArg = entryBlock->getArgument(1);

    // Allocate CMX buffers for input and output
    auto input = builder.create<mlir::memref::AllocOp>(loc, inputTypeCMX);
    auto output = builder.create<mlir::memref::AllocOp>(loc, outputTypeCMX);

    // Copy input argument from DDR to CMX
    auto copyIn = builder.create<VPUIP::NNDMAOp>(loc, inputArg, input);

    // Create weights and weight tables
    llvm::SmallVector<mlir::Value> weightOps;
    llvm::SmallVector<mlir::Value> weightTableOps;
    for (int c = 0; c < numTilesC; c++) {
        Shape weightsShape = {tileC, inputChannels, kernelSize, kernelSize};
        auto copyWeights = createWeights(builder, loc, f16Type, weightsShape, ddrSpace, cmxSpace);
        auto copyWeightsTable = createWeightsTable(builder, loc, tileC, ddrSpace, cmxSpace);
        weightOps.push_back(copyWeights);
        weightTableOps.push_back(copyWeightsTable);
    }

    llvm::SmallVector<mlir::Value> allTiles;

    for (int h = 0; h < numTilesH; h++) {
        for (int c = 0; c < numTilesC; c++) {
            // Create input tile view
            auto inputTile = builder.create<vpux::VPUIP::SubViewOp>(
                    loc, copyIn.getOutput(), mlir::ArrayRef<int64_t>{0, 0, h * tileH, 0},
                    mlir::ArrayRef<int64_t>{1, inputChannels, tileH, inputSizeW});

            // Allocate output tile
            auto outputTileType = vpux::getMemRefType({1, tileC, tileH, inputSizeW}, f16Type, orderNHWC, cmxSpace);
            auto outputTile = builder.create<mlir::memref::AllocOp>(loc, outputTileType);

            // Create NCE task
            auto mpeEngineAttr = createMPEEngineAttr(ctx, arch);
            auto nceOp = createNCEClusterTaskOp(builder, ctx, loc, kernelSize, padding, stride, inputTile, weightOps[c],
                                                weightTableOps[c], outputTile, outputTileType, mpeEngineAttr);

            // Set tilingIndex attribute
            auto tilingIndexAttr = builder.getI64IntegerAttr(0);
            nceOp->setAttr(TILING_LOOP_INDEX_ATTR_NAME, tilingIndexAttr);

            // Add DPU task
            auto& dpuTaskRegion = nceOp.getVariants();
            builder.setInsertionPointToStart(&dpuTaskRegion.front());

            createDPUTaskOp(builder, {0, c * tileC, h * tileH}, {1, (c + 1) * tileC, (h + 1) * tileH});
            builder.setInsertionPointAfter(nceOp);

            allTiles.push_back(nceOp.getOutput());
        }
    }

    // Concatenate all tiles into the output buffer
    auto concatOp = builder.create<vpux::VPUIP::ConcatViewOp>(loc, allTiles, output);

    // Copy result back from CMX to DDR output argument
    auto copyOut = builder.create<VPUIP::NNDMAOp>(loc, concatOp.getOutput(), outputArg);

    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{copyOut.getOutput()});

    // Initialize architecture
    module->setAttr("config.arch", config::ArchKindAttr::get(ctx, arch));

    // Wrap into async regions
    mlir::PassManager pm(ctx);
    VPUIP::buildAsyncSchedulingPipeline(pm);
    EXPECT_TRUE(mlir::succeeded(pm.run(func)));

    // Assign dummy cost
    func.walk([&](mlir::async::ExecuteOp execOp) {
        execOp->setAttr(cycleCostAttrName, builder.getI64IntegerAttr(1000));
    });

    return module;
}

std::string testParamName(const testing::TestParamInfo<config::ArchKind>& info) {
    switch (info.param) {
    case config::ArchKind::NPU40XX:
        return "NPU40XX";
    case config::ArchKind::NPU50XX:
        return "NPU50XX";
    default:
        return "UnknownArch";
    }
}

TEST_P(MLIR_SchedulerLoopCreationTest, ConvolutionTiledOnC) {
    // create module
    const int tilesH = 1;
    const int tilesC = 4;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());
    // Function contains:
    // 4 tiles over C, each tile has 2 DMAs: weights in, weights table in and 1 DPU CONV
    // Also there are copies to/from DDR for input and output

    // get alias info, live range info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure:
    ComputeRegionVec
    |---ComputeRegion 0:
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 0, DATA_IN <-- input copy from DDR
    |
    |---ComputeRegion 1:
    |   \---SchedulingLoop: type: Tiling, iterations: 4
    |       \---loopBody:
    |           |---Iteration 0:
    |           |        Alloc: opIdx_: 1, DATA_IN <-- weights in
    |           |        Alloc: opIdx_: 2, DATA_IN <-- weights table in
    |           |        Alloc: opIdx_: 9, COMPUTE
    |           |---Iteration 1:
    |           |        Alloc: opIdx_: 3, DATA_IN
    |           |        Alloc: opIdx_: 4, DATA_IN
    |           |        Alloc: opIdx_: 10, COMPUTE
    |           |---Iteration 2:
    |           |        Alloc: opIdx_: 5, DATA_IN
    |           |        Alloc: opIdx_: 6, DATA_IN
    |           |        Alloc: opIdx_: 11, COMPUTE
    |           \---Iteration 3:
    |                   Alloc: opIdx_: 7, DATA_IN
    |                   Alloc: opIdx_: 8, DATA_IN
    |                   Alloc: opIdx_: 12, COMPUTE
    |
    \---ComputeRegion 2:
        \---SchedulingLoop: type: None, iterations: 1
            \---loopBody:
                \---Alloc: opIdx_: 13, DATA_OUT --> output copy to DDR
    */

    // 3 compute regions expected
    EXPECT_EQ(regions.size(), 3);
    // Region with index 1 is compute op tiled on C
    const size_t tiledRegionIndex = 1;
    const auto loopSize = regions[tiledRegionIndex].schedulingLoop->loopBodies.size();
    // Each iteration corresponds to 1 tile, so expect 4 tiles
    EXPECT_EQ(loopSize, 4);
    // check that the first iteration contains expected ops
    const auto& firstIteration = regions[tiledRegionIndex].schedulingLoop->loopBodies[0];
    EXPECT_EQ(firstIteration.size(), 3);  // expect 3 ops: weights in, weights table in, DPU CONV
    // check that the first op is weights data in
    const auto firstOp = depsInfo.getExecuteOpAtIndex(firstIteration[0].opIdx);
    const auto firstExecKind = VPUIP::VPUIPDialect::getExecutorKind(firstOp);
    EXPECT_EQ(firstExecKind, config::ExecutorKind::DMA_NN);
    // check that the second op is weights table data in
    const auto secondOp = depsInfo.getExecuteOpAtIndex(firstIteration[1].opIdx);
    const auto secondExecKind = VPUIP::VPUIPDialect::getExecutorKind(secondOp);
    EXPECT_EQ(secondExecKind, config::ExecutorKind::DMA_NN);
    // check that the third op is DPU
    const auto dpuOp = depsInfo.getExecuteOpAtIndex(firstIteration[2].opIdx);
    const auto dpuExecKind = VPUIP::VPUIPDialect::getExecutorKind(dpuOp);
    EXPECT_EQ(dpuExecKind, config::ExecutorKind::DPU);
}

TEST_P(MLIR_SchedulerLoopCreationTest, ConvolutionTiledOnH) {
    // create module
    const int tilesH = 4;
    const int tilesC = 1;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    // get alias info, live range info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure:
    ComputeRegionVec
    |---ComputeRegion 0
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 0, DATA_IN
    |
    |---ComputeRegion 1
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 1, DATA_IN
    |
    |---ComputeRegion 2
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 2, DATA_IN
    |
    |---ComputeRegion 3
    |   \---SchedulingLoop: type: Tiling, iterations: 4
    |       \---loopBody:
    |           |---Iteration 0:
    |           |   \---Alloc: opIdx_: 3, COMPUTE <- no data ins since subviews are used for inputs
    |           |---Iteration 1:
    |           |   \---Alloc: opIdx_: 4, COMPUTE
    |           |---Iteration 2:
    |           |   \---Alloc: opIdx_: 5, COMPUTE
    |           \---Iteration 3:
    |               \---Alloc: opIdx_: 6, COMPUTE
    |
    \---ComputeRegion 4
        \---SchedulingLoop: type: None, iterations: 1
            \---loopBody:
                \---Alloc: opIdx_: 7, DATA_OUT
    */
    // 5 compute regions expected
    EXPECT_EQ(regions.size(), 5);
    // Region with index 3 is compute op tiled on H
    const size_t tiledRegionIndex = 3;
    const auto loopSize = regions[tiledRegionIndex].schedulingLoop->loopBodies.size();
    // Each iteration corresponds to 1 tile, so expect 4 tiles
    EXPECT_EQ(loopSize, 4);
    // check that the first iteration contains expected ops
    const auto& firstIteration = regions[tiledRegionIndex].schedulingLoop->loopBodies[0];
    EXPECT_EQ(firstIteration.size(), 1);  // expect only 1 op: DPU CONV
    // check that the first op is DPU
    const auto conv = depsInfo.getExecuteOpAtIndex(firstIteration[0].opIdx);
    const auto convExecKind = VPUIP::VPUIPDialect::getExecutorKind(conv);
    EXPECT_EQ(convExecKind, config::ExecutorKind::DPU);
}

TEST_P(MLIR_SchedulerLoopCreationTest, ConvolutionTiledOnCAndH) {
    // create module
    const int tilesH = 4;
    const int tilesC = 4;
    auto module = createTiledConvolutionModule(getCtx(), tilesH, tilesC, GetParam());
    ASSERT_TRUE(module);
    EXPECT_TRUE(module->verify().succeeded());

    // get alias info, live range info and async deps info
    AliasesInfo aliasInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};
    AsyncDepsInfo depsInfo{module->lookupSymbol<mlir::func::FuncOp>("main")};

    // test start
    auto regions = vpux::getComputeRegionsFromAsyncExec(aliasInfo, depsInfo);
    /*
    Expected structure:
    ComputeRegionVec
    |---ComputeRegion 0
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 0, DATA_IN
    |
    .
    . All data ins, op idxes 0-8
    .
    .
    |---ComputeRegion 8
    |   \---SchedulingLoop: type: None, iterations: 1
    |       \---loopBody:
    |           \---Alloc: opIdx_: 8, DATA_IN
    |
    |---ComputeRegion 9 <- Tiled ops grouped by tile index over H
    |   \---SchedulingLoop: type: Tiling, iterations: 4
    |       \---loopBody:
    |           |---Iteration 0:
    |           |   \---Alloc: opIdx_: 9, COMPUTE <- no data ins since subviews are used for inputs
    |           |---Iteration 1:
    |           |   \---Alloc: opIdx_: 13, COMPUTE
    |           |---Iteration 2:
    |           |   \---Alloc: opIdx_: 17, COMPUTE
    |           \---Iteration 3:
    |               \---Alloc: opIdx_: 21, COMPUTE
    |---ComputeRegion 10
    |   \---SchedulingLoop: type: Tiling, iterations: 4
    |       \---loopBody:
    |           |---Iteration 0:
    |           |   \---Alloc: opIdx_: 10, COMPUTE <- no data ins since subviews are used for inputs
    |           |---Iteration 1:
    |           |   \---Alloc: opIdx_: 14, COMPUTE
    |           |---Iteration 2:
    |           |   \---Alloc: opIdx_: 18, COMPUTE
    |           \---Iteration 3:
    |               \---Alloc: opIdx_: 22, COMPUTE
    |---ComputeRegion 11
    |   \---SchedulingLoop: type: Tiling, iterations: 4
    |       \---loopBody:
    |           |---Iteration 0:
    |           |   \---Alloc: opIdx_: 11, COMPUTE <- no data ins since subviews are used for inputs
    |           |---Iteration 1:
    |           |   \---Alloc: opIdx_: 15, COMPUTE
    |           |---Iteration 2:
    |           |   \---Alloc: opIdx_: 19, COMPUTE
    |           \---Iteration 3:
    |               \---Alloc: opIdx_: 23, COMPUTE
    |---ComputeRegion 12
    |   \---SchedulingLoop: type: Tiling, iterations: 4
    |       \---loopBody:
    |           |---Iteration 0:
    |           |   \---Alloc: opIdx_: 12, COMPUTE <- no data ins since subviews are used for inputs
    |           |---Iteration 1:
    |           |   \---Alloc: opIdx_: 16, COMPUTE
    |           |---Iteration 2:
    |           |   \---Alloc: opIdx_: 20, COMPUTE
    |           \---Iteration 3:
    |               \---Alloc: opIdx_: 24, COMPUTE
    \---ComputeRegion 13
        \---SchedulingLoop: type: None, iterations: 1
            \---loopBody:
                \---Alloc: opIdx_: 25, DATA_OUT
    */

    // 14 compute regions expected
    EXPECT_EQ(regions.size(), 14);
    size_t tiledRegionIndex = 9;
    auto loopSize = regions[tiledRegionIndex].schedulingLoop->loopBodies.size();
    EXPECT_EQ(loopSize, 4);

    // Compute region with index 9 is tiled over on H, check that the first iteration contains expected ops
    const auto& firstIteration = regions[tiledRegionIndex].schedulingLoop->loopBodies[0];
    EXPECT_EQ(firstIteration.size(), 1);  // expect only 1 op: DPU CONV
    // check that the first op is DPU
    const auto conv = depsInfo.getExecuteOpAtIndex(firstIteration[0].opIdx);
    const auto convExecKind = VPUIP::VPUIPDialect::getExecutorKind(conv);
    EXPECT_EQ(convExecKind, config::ExecutorKind::DPU);

    // calculate number of tiled regions
    size_t numTiledRegions = llvm::count_if(regions, [](const auto& region) {
        return region.schedulingLoop->type == LoopType::Tiling;
    });
    // expect 4 tiled regions, all tiled over H
    // nested tiling loops are not supported yet therefore tiling over C does not form a tiled region
    EXPECT_EQ(numTiledRegions, 4);
}

INSTANTIATE_TEST_SUITE_P(SchedulerLoopCreation, MLIR_SchedulerLoopCreationTest,
                         testing::Values(config::ArchKind::NPU40XX, config::ArchKind::NPU50XX), testParamName);
