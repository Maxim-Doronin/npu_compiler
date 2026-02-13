//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_vpu_nce_ops_interface.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

void addppeAttr(const Logger& log, mlir::OpBuilder& builder, VPUIP::NCEClusterTaskOp& nceOp, VPU::PPEAttr ppeAttr) {
    log.nest().trace("Adding PPE attribute '{0}'", ppeAttr);
    nceOp.addPPETask(builder, ppeAttr);
}

void addDPUTasks(const Logger& log, VPUIP::NCEClusterTaskOp nceOp, mlir::OpBuilder& rewriter, mlir::Region& workloads,
                 bool isNCEPermute) {
    log.nest().trace("Adding DPU tasks");

    for (auto dpuTaskOp : workloads.getOps<VPU::DPUWorkloadOp>()) {
        SmallVector<int64_t> ends;
        const auto offsets = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutOffsets());
        const auto sizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getOutSizes());
        ends.reserve(sizes.size());

        llvm::transform(llvm::seq<size_t>(0, sizes.size()), std::back_inserter(ends), [&](size_t index) {
            return offsets[index] + sizes[index] - 1;
        });

        mlir::ArrayAttr inStartAttr = nullptr;
        mlir::ArrayAttr inEndAttr = nullptr;
        const auto isGroupedMatMul = offsets.size() == DimsGroups5D::Act::numDims;
        // Update workloads padding, offsets and sizes
        // after reshape and layout changes.
        if (isNCEPermute) {
            // Reshape Offsets and Sizes from CHW to HCW layout
            const SmallVector<int64_t> outDpuStart{offsets[Dims4D::Act::H.ind()], offsets[Dims4D::Act::C.ind()],
                                                   offsets[Dims4D::Act::W.ind()]};
            const SmallVector<int64_t> outDpuEnds{ends[Dims4D::Act::H.ind()], ends[Dims4D::Act::C.ind()],
                                                  ends[Dims4D::Act::W.ind()]};
            if (dpuTaskOp.getInOffsetsAttr() != nullptr && dpuTaskOp.getInSizesAttr() != nullptr) {
                const auto inOffset = parseIntArrayAttr<int64_t>(dpuTaskOp.getInOffsetsAttr());
                const auto inSizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getInSizesAttr());
                const SmallVector<int64_t> inDpuStart{inOffset[Dims4D::Act::H.ind()], inOffset[Dims4D::Act::C.ind()],
                                                      inOffset[Dims4D::Act::W.ind()]};
                const SmallVector<int64_t> inDpuEnds{
                        inOffset[Dims4D::Act::H.ind()] + inSizes[Dims4D::Act::H.ind()] - 1,
                        inOffset[Dims4D::Act::C.ind()] + inSizes[Dims4D::Act::C.ind()] - 1,
                        inOffset[Dims4D::Act::W.ind()] + inSizes[Dims4D::Act::W.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnds);
            }
            nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                             inStartAttr, inEndAttr, dpuTaskOp.getPadAttr(), dpuTaskOp.getMpeMode(),
                             dpuTaskOp.getClusterIdAttr());
        } else if (isGroupedMatMul) {
            // This part is for grouped Matmul which has 5D input/output
            // Logic is same only dimensions are adjusted for 5D
            const auto dimC = DimsGroups5D::Act::C;
            const auto dimH = DimsGroups5D::Act::H;
            const auto dimW = DimsGroups5D::Act::W;
            const SmallVector<int64_t> outDpuStart{offsets[dimW.ind()], offsets[dimH.ind()], offsets[dimC.ind()]};
            const SmallVector<int64_t> outDpuEnds{ends[dimW.ind()], ends[dimH.ind()], ends[dimC.ind()]};

            if (dpuTaskOp.getInOffsetsAttr() != nullptr && dpuTaskOp.getInSizesAttr() != nullptr) {
                const auto inOffset = parseIntArrayAttr<int64_t>(dpuTaskOp.getInOffsetsAttr());
                const auto inSizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getInSizesAttr());

                const SmallVector<int64_t> inDpuStart{inOffset[dimW.ind()], inOffset[dimH.ind()], inOffset[dimC.ind()]};
                const SmallVector<int64_t> inDpuEnds{inOffset[dimW.ind()] + inSizes[dimW.ind()] - 1,
                                                     inOffset[dimH.ind()] + inSizes[dimH.ind()] - 1,
                                                     inOffset[dimC.ind()] + inSizes[dimC.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnds);
            }

            nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                             inStartAttr, inEndAttr, dpuTaskOp.getPad(), dpuTaskOp.getMpeMode(),
                             dpuTaskOp.getClusterIdAttr());
        } else {
            // as soon as we need workload_x, workload_y, workload_z coords
            const SmallVector<int64_t> outDpuStart{offsets[Dims4D::Act::W.ind()], offsets[Dims4D::Act::H.ind()],
                                                   offsets[Dims4D::Act::C.ind()]};
            const SmallVector<int64_t> outDpuEnds{ends[Dims4D::Act::W.ind()], ends[Dims4D::Act::H.ind()],
                                                  ends[Dims4D::Act::C.ind()]};

            if (dpuTaskOp.getInOffsetsAttr() != nullptr && dpuTaskOp.getInSizesAttr() != nullptr) {
                const auto inOffset = parseIntArrayAttr<int64_t>(dpuTaskOp.getInOffsetsAttr());
                const auto inSizes = parseIntArrayAttr<int64_t>(dpuTaskOp.getInSizesAttr());

                const SmallVector<int64_t> inDpuStart{inOffset[Dims4D::Act::W.ind()], inOffset[Dims4D::Act::H.ind()],
                                                      inOffset[Dims4D::Act::C.ind()]};
                const SmallVector<int64_t> inDpuEnds{
                        inOffset[Dims4D::Act::W.ind()] + inSizes[Dims4D::Act::W.ind()] - 1,
                        inOffset[Dims4D::Act::H.ind()] + inSizes[Dims4D::Act::H.ind()] - 1,
                        inOffset[Dims4D::Act::C.ind()] + inSizes[Dims4D::Act::C.ind()] - 1};

                inStartAttr = getIntArrayAttr(rewriter, inDpuStart);
                inEndAttr = getIntArrayAttr(rewriter, inDpuEnds);
            }

            nceOp.addDPUTask(rewriter, getIntArrayAttr(rewriter, outDpuStart), getIntArrayAttr(rewriter, outDpuEnds),
                             inStartAttr, inEndAttr, dpuTaskOp.getPad(), dpuTaskOp.getMpeMode(),
                             dpuTaskOp.getClusterIdAttr());
        }
    }
}

//
// Create VPUIP.NCEClusterTask and ensure sparse types interact with the operation as individual buffers
//

struct NCEClusterTaskParams {
    struct Weights {
        mlir::Value weights;
        mlir::Value weightsTable;
        mlir::Value weightTableScale;
        mlir::Value weightTableBias;
        mlir::Value weightTableZeroPoints;
    };

    struct Kernel {
        mlir::ArrayAttr kernelSizeAttr;
        mlir::ArrayAttr kernelStridesAttr;
        vpux::VPU::PaddingAttr kernelPaddingAttr;
    };

    // Required attributes: They have to explicitly be set by the user.
    mlir::Value input;
    Weights weights;
    ArrayRef<mlir::Value> outputBuffs;
    vpux::VPUIP::NCETaskType taskType;
    Kernel kernel;
    mlir::Region& workloads;

    NCEClusterTaskParams(mlir::Value input, const Weights& weights, ArrayRef<mlir::Value> outputBuffs,
                         vpux::VPUIP::NCETaskType taskType, const Kernel& kernel, mlir::Region& workloads)
            : input(input),
              weights(weights),
              outputBuffs(outputBuffs),
              taskType(taskType),
              kernel(kernel),
              workloads(workloads) {
    }

    // Optional attributes
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    VPU::PPEAttr ppeAttr = nullptr;
    mlir::Attribute dpuCostAttr = nullptr;
    mlir::BoolAttr isInplace = nullptr;
    mlir::UnitAttr isPermuteQuantize = nullptr;
    mlir::IntegerAttr cmSpPattern = nullptr;
    mlir::UnitAttr inputChannelsCompression = nullptr;
    bool isNCEPermute = false;
    mlir::UnitAttr smallKernelOptimization = nullptr;
    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    VPU::EltwiseTypeAttr eltwiseType = nullptr;
    TilingLoopIndexAttr tilingLoopIndex = nullptr;
    VFLoopIndexAttr vfLoopIndex = nullptr;
    VFLoopLayerIndexAttr vfLoopLayerIndex = nullptr;
};

mlir::Value createNCEClusterTask(mlir::OpBuilder& rewriter, mlir::Location loc, const NCEClusterTaskParams& params,
                                 Logger log = Logger::global()) {
    const auto getIndividualBuffers = [&](mlir::Value value) {
        mlir::Value data = value;
        mlir::Value sparsityMap = nullptr;
        mlir::Value seTable = nullptr;
        if (value != nullptr && mlir::isa<vpux::VPUIP::SparseBufferType>(value.getType())) {
            auto ungroupedOp = rewriter.create<VPUIP::UngroupSparseBufferOp>(loc, value);
            data = ungroupedOp.getData();
            sparsityMap = ungroupedOp.getSparsityMap();
            seTable = ungroupedOp.getStorageElementTable();
        }
        return std::make_tuple(data, sparsityMap, seTable);
    };

    mlir::Value inputData, inputSparsityMap, inputSETable;
    std::tie(inputData, inputSparsityMap, inputSETable) = getIndividualBuffers(params.input);

    mlir::Value weightsData, weightsSparsityMap;
    std::tie(weightsData, weightsSparsityMap, std::ignore) = getIndividualBuffers(params.weights.weights);

    mlir::Value outputBuffData = params.outputBuffs[0];
    mlir::Value outputBuffSparsityMap = (params.outputBuffs.size() > 1) ? params.outputBuffs[1] : nullptr;

    auto nceClusterTask = rewriter.create<VPUIP::NCEClusterTaskOp>(
            loc, inputData, inputSparsityMap, inputSETable, weightsData, weightsSparsityMap,
            params.weights.weightsTable,
            /*weight_table_data_ptr=*/nullptr, /*weight_table_sp_ptr=*/nullptr, params.weights.weightTableScale,
            params.weights.weightTableBias,
            /*weight_zero_points=*/params.weights.weightTableZeroPoints,
            /*sprLookupTable=*/nullptr, /*palletLookupTable=*/nullptr, inputData, inputSparsityMap, inputSETable,
            outputBuffData, outputBuffSparsityMap, outputBuffData, outputBuffSparsityMap, /*profiling_data=*/nullptr,
            /*dynamic_sequence_length*/ nullptr,
            /*max_per_xy=*/nullptr, /*min_per_xy=*/nullptr, /*min_max_per_tensor=*/mlir::ValueRange(), params.taskType,
            params.kernel.kernelSizeAttr, params.kernel.kernelStridesAttr, params.kernel.kernelPaddingAttr,
            /*is_continued=*/nullptr, params.cmSpPattern,
            /*is_segmented=*/nullptr,
            /*out_channel_offset=*/nullptr, params.inputChannelsCompression, /*isZeroOffsetWeightsTable=*/nullptr,
            params.isSuperdenseAttr, params.isInplace,
            /*input_se_size=*/nullptr,
            /*output_se_size=*/nullptr, params.isPermuteQuantize, params.smallKernelOptimization, params.mpeEngineAttr,
            params.eltwiseType,
            /*dynamicScaleConfig=*/nullptr);

    addDPUTasks(log, nceClusterTask, rewriter, params.workloads, params.isNCEPermute);
    addppeAttr(log, rewriter, nceClusterTask, params.ppeAttr);

    if (params.dpuCostAttr != nullptr) {
        nceClusterTask->setAttr(DPUCost, params.dpuCostAttr);
    }

    if (nceClusterTask.getOutputSparsityMap() != nullptr) {
        auto groupedOp = rewriter.create<VPUIP::GroupSparseBufferOp>(loc, nceClusterTask.getOutput(),
                                                                     nceClusterTask.getOutputSparsityMap());
        return groupedOp.getOutput();
    }

    if (params.tilingLoopIndex != nullptr) {
        nceClusterTask->setAttr(TILING_LOOP_INDEX_ATTR_NAME, params.tilingLoopIndex);
    }
    if (params.vfLoopIndex != nullptr) {
        nceClusterTask->setAttr(VF_LOOP_INDEX_ATTR_NAME, params.vfLoopIndex);
    }
    if (params.vfLoopLayerIndex != nullptr) {
        nceClusterTask->setAttr(VF_LOOP_LAYER_INDEX_ATTR_NAME, params.vfLoopLayerIndex);
    }

    return nceClusterTask.getOutput();
}

bool isSuperdenseOp(mlir::Operation* nceOp) {
    auto outType = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType());
    const auto outputOrder = outType.getDimsOrder();
    const auto outputShape = outType.getShape();
    const auto outElemType = outType.getElementType();

    // Check output shape for each cluster
    if (auto distributedTensorType = mlir::dyn_cast<VPU::DistributedTensorType>(outType)) {
        auto tiledComputeShapes = distributedTensorType.getPerClusterComputeShapes();
        for (auto& computeShape : tiledComputeShapes) {
            if (VPU::NCESparsity::isSuperdenseRequired(outputOrder, computeShape, outElemType)) {
                return true;
            }
        }
        return false;
    }

    return VPU::NCESparsity::isSuperdenseRequired(outputOrder, outputShape, outElemType);
}

SmallVector<int64_t> calculateWCHShape(ArrayRef<int64_t> shape) {
    const int64_t tensorSizeZ = shape[Dims4D::Act::W.ind()];
    const int64_t tensorSizeY = shape[Dims4D::Act::C.ind()];
    const int64_t tensorSizeX = shape[Dims4D::Act::H.ind()];
    return {shape[Dims4D::Act::N.ind()], tensorSizeZ, tensorSizeY, tensorSizeX};
}

}  // namespace

//
// bufferize VPU::NCEConvolutionO
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEConvolutionOp origOp,
                                      VPU::NCEConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    const auto taskType = VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    mlir::IntegerAttr cmSpPattern = nullptr;
    auto inputShape = mlir::cast<NDTypeInterface>(newArgs.getInput().getType()).getShape();
    if (inputShape.size() == 4 && inputShape[Dims4D::Act::C] < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
        const auto pattern = (static_cast<int64_t>(1) << inputShape[Dims4D::Act::C]) - 1;
        cmSpPattern = getIntAttr(ctx, pattern);
    }

    const auto loopAttributes = getLoopAttributes(origOp);
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getFilter(), newArgs.getWeightsTable(), newArgs.getWeightTableScale(),
                                          newArgs.getWeightTableBias(), newArgs.getWeightZeroPoints()},
            outputBuffers, taskType,
            NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.cmSpPattern = cmSpPattern;
    params.mpeEngineAttr = origOp.getMpeEngineAttr();
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEMaxPoolOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMaxPoolOp origOp,
                                      VPU::NCEMaxPoolOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEMaxPoolOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{nullptr, newArgs.getWeightsTable(), nullptr, nullptr, nullptr}, outputBuffers,
            VPUIP::NCETaskType::MAXPOOL,
            NCEClusterTaskParams::Kernel{origOp.getKernelSize(), origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;

    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEAveragePoolOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEAveragePoolOp origOp,
                                      VPU::NCEAveragePoolOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEAveragePoolOp", 0);
    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    mlir::UnitAttr isSmallKernelOptimizationAttr = nullptr;
    if (VPU::NCEInvariant::isSmallKernelOptimizationSupported(config::getArch(origOp), origOp)) {
        isSmallKernelOptimizationAttr = mlir::UnitAttr::get(ctx);
    }

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(
            newArgs.getInput(), NCEClusterTaskParams::Weights{nullptr, nullptr, nullptr, nullptr, nullptr},
            outputBuffers, VPUIP::NCETaskType::AVEPOOL,
            NCEClusterTaskParams::Kernel{origOp.getKernelSize(), origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.smallKernelOptimization = isSmallKernelOptimizationAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEDepthConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEDepthConvolutionOp origOp,
                                      VPU::NCEDepthConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEDepthConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    auto ppeAttr = origOp.getPpeAttr();

    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    auto arch = config::getArch(origOp);
    mlir::UnitAttr isSmallKernelOptimizationAttr = nullptr;
    if (VPU::NCEInvariant::isSmallKernelOptimizationSupported(arch, origOp)) {
        isSmallKernelOptimizationAttr = mlir::UnitAttr::get(ctx);
    }

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getFilter(), newArgs.getWeightsTable(), newArgs.getWeightTableScale(),
                                          newArgs.getWeightTableBias(), nullptr},
            outputBuffers, VPUIP::NCETaskType::DWCONV,
            NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.smallKernelOptimization = isSmallKernelOptimizationAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEInterpolateOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEInterpolateOp origOp,
                                      VPU::NCEInterpolateOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEInterpolateOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));

    log.nest().trace("Allocating output buffer");

    auto newLoc = appendLoc(origOp.getLoc(), "interpolate");

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, newLoc, rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);

    auto nceOpInterface = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getWeights(), newArgs.getWeightsTable(),
                                          newArgs.getWeightTableScale(), newArgs.getWeightTableBias(), nullptr},
            outputBuffers, VPUIP::NCETaskType::CONV,
            NCEClusterTaskParams::Kernel{kernelSizeAttr, getIntArrayAttr(ctx, nceOpInterface.getStridesVal()),
                                         nceOpInterface.getPad()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, newLoc, params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEEltwiseOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEEltwiseOp origOp,
                                      VPU::NCEEltwiseOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEEltwiseOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto ppeAttr = origOp.getPpeAttr();

    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(newArgs.getInput1(),
                                NCEClusterTaskParams::Weights{newArgs.getInput2(), nullptr, nullptr, nullptr, nullptr},
                                outputBuffers, VPUIP::NCETaskType::ELTWISE,
                                NCEClusterTaskParams::Kernel{nullptr, nullptr, nullptr}, origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.isInplace = origOp.getIsInplaceAttr();
    params.mpeEngineAttr = mpeEngineAttr;
    params.eltwiseType = origOp.getOpTypeAttr();
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCEReduceOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEReduceOp origOp,
                                      VPU::NCEReduceOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEReduceOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    auto nceTaskType = VPU::configureNCEReduceTaskType(origOp);
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);
    auto nceOpInterface = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());

    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{nceOpInterface.getWeightsOperand(), nceOpInterface.getWeightsTableOperand(),
                                          nullptr, nullptr, nullptr},
            outputBuffers, nceTaskType,
            NCEClusterTaskParams::Kernel{getIntArrayAttr(ctx, nceOpInterface.getKernelSizeVal()),
                                         getIntArrayAttr(ctx, nceOpInterface.getStridesVal()), nceOpInterface.getPad()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = origOp.getPpeAttr();
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

//
// bufferize VPU::NCECompressConvolutionOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCECompressConvolutionOp origOp,
                                      VPU::NCECompressConvolutionOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCECompressConvolutionOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    //
    // Get dimensions
    //

    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto channelAlignValue = VPU::NCEInvariant::getAlignment(
            mlir::cast<vpux::NDTypeInterface>(newArgs.getFilter().getType()).getElementType());

    const auto finalShape = SmallVector<int64_t>({filterShape[Dims4D::Filter::OC], channelAlignValue, KY, KX});
    auto shapeCastWeightsOp = rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getFilter(),
                                                                  getIntArrayAttr(origOp.getContext(), finalShape));
    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //
    auto inputType = newArgs.getInput().getType();
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(inputType).getShape();
    const auto finalInputShape = vpux::Shape(
            {inputShape[Dims4D::Act::N], channelAlignValue, inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]});
    auto finalInputShapeAttr = getIntArrayAttr(origOp.getContext(), finalInputShape);

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }
    auto inputShapeCastOp =
            rewriter.create<VPUIP::ShapeCastOp>(origOp->getLoc(), newArgs.getInput(), finalInputShapeAttr);
    const auto inputChannelsCompression = mlir::UnitAttr::get(origOp->getContext());

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);

    NCEClusterTaskParams params(inputShapeCastOp.getResult(),
                                NCEClusterTaskParams::Weights{shapeCastWeightsOp.getResult(), newArgs.getWeightsTable(),
                                                              nullptr, nullptr, nullptr},
                                outputBuffers, VPUIP::NCETaskType::CONV,
                                NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
                                origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.cmSpPattern = origOp.getCmSpPatternAttr();
    params.inputChannelsCompression = inputChannelsCompression;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

VPU::DistributedTensorType createCustomDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                             NDTypeInterface targetType,
                                                             VPU::DistributionInfoAttr origDistTensorAttr,
                                                             mlir::UnitAttr equalMemoryAndComputeView, ShapeRef shape);

//
// bufferize VPU::NCEPermuteOp
//

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEPermuteOp origOp,
                                      VPU::NCEPermuteOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEPermuteOp", 0);

    auto copyDistTensorType = mlir::dyn_cast<VPU::DistributedTensorType>(origOp->getOperand(0).getType());
    if (copyDistTensorType != nullptr) {
        log.trace("Got '{0}' Multi Tile at '{1}'", origOp->getName(), origOp->getLoc());
        auto ctx = origOp->getContext();

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                          origOp);

        const auto loc = origOp.getLoc();
        const auto copyDistTensorAttr = copyDistTensorType.getDistribution();
        auto targetType = mlir::cast<NDTypeInterface>(origOp.getOperand().getType());
        targetType = targetType.changeDimsOrder(DimsOrder::NHWC);

        auto castToDistType = createCustomDistributedTensorType(clusteredOp, targetType, copyDistTensorAttr,
                                                                copyDistTensorAttr.getEqualMemoryAndComputeView(),
                                                                targetType.getShape());

        auto outBufferTypeInViewOp = vpux::getBufferType(castToDistType);
        const auto castLoc = appendLoc(loc, "cast number of input tiles");
        // ViewOp Input
        // Reshape to NxWxCxH
        // Layout change to NHWC
        auto inputViewOp = rewriter.create<VPUIP::ViewOp>(castLoc, outBufferTypeInViewOp, newArgs.getInput());

        // Manual update output type
        auto outType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
        auto outTypeShape = outType.getShape();
        targetType = targetType.changeElemType(outType.getElementType());
        auto origOutDistribution = mlir::cast<VPU::DistributedTensorType>(outType).getDistribution();
        auto newDistType =
                createCustomDistributedTensorType(clusteredOp, targetType, origOutDistribution,
                                                  origOutDistribution.getEqualMemoryAndComputeView(), outTypeShape);
        auto newOutputType = newDistType.changeDimsOrder(DimsOrder::NWCH);

        //
        // Prepare output buffer for DPU
        //
        auto bufferType = vpux::getBufferType(newOutputType);

        auto ppeAttr = origOp.getPpeAttr();
        const auto& modeAdapter = VPU::PpeVersionConfig::getFactoryAs<vpux::VPU::IPpeAdapterMode>();
        ppeAttr = modeAdapter.updateMode(ppeAttr, vpux::VPU::PPEMode::ADD);

        mlir::UnitAttr isSuperdenseAttr = nullptr;
        if (isSuperdenseOp(origOp)) {
            VPUX_THROW_WHEN(mlir::isa<VPU::SparseTensorType>(origOp->getResult(0).getType()),
                            "Output cannot be sparse and super-dense at the same time");
            isSuperdenseAttr = mlir::UnitAttr::get(ctx);
        }

        const auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;
        const auto isPermuteQuantizeAttr = mlir::UnitAttr::get(ctx);
        const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
        const auto loopAttributes = getLoopAttributes(origOp);

        log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
        const auto outputBuffers =
                VPUIP::allocateBuffersOfType(log.nest(), loc, rewriter, bufferType, /*individualBuffers=*/true);

        NCEClusterTaskParams params(
                inputViewOp.getResult(),
                NCEClusterTaskParams::Weights{inputViewOp.getResult(), nullptr, nullptr, nullptr, nullptr},
                outputBuffers, VPUIP::NCETaskType::ELTWISE, NCEClusterTaskParams::Kernel{nullptr, nullptr, nullptr},
                origOp.getWorkloads());
        params.isSuperdenseAttr = isSuperdenseAttr;
        params.ppeAttr = ppeAttr;
        params.dpuCostAttr = dpuCostAttr;
        params.isPermuteQuantize = isPermuteQuantizeAttr;
        params.isNCEPermute = true;
        params.mpeEngineAttr = mpeEngineAttr;
        params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
        params.vfLoopIndex = loopAttributes.vfLoopIndex;
        params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
        auto nceOpResult = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

        // ViewOp Output
        // Reshape to NxCxHxW
        // Layout change to NHWC

        auto outputViewOp = rewriter.create<VPUIP::ViewOp>(
                origOp.getLoc(), vpux::getBufferType(origOp.getResult().getType()), nceOpResult);

        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, outputViewOp.getResult());

        return mlir::success();
    }

    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());

    log.trace("Got '{0}' Single Tile '{1}'", origOp->getName(), origOp->getLoc());

    // ViewOp Input
    // Reshape to NxWxCxH
    // Layout change to NHWC
    const auto inputShape = getShape(newArgs.getInput());
    const auto targetShape = calculateWCHShape(inputShape.raw());

    auto inType = mlir::cast<vpux::NDTypeInterface>(newArgs.getInput().getType());
    const auto targetInOutOrder = DimsOrder::NHWC;
    inType = inType.changeShape(ShapeRef(targetShape));
    inType = inType.changeDimsOrder(targetInOutOrder);
    auto viewOpIn = rewriter.create<VPUIP::ViewOp>(origOp.getLoc(), inType, newArgs.getInput());

    auto ppeAttr = origOp.getPpeAttr();
    const auto& modeAdapter = VPU::PpeVersionConfig::getFactoryAs<vpux::VPU::IPpeAdapterMode>();
    ppeAttr = modeAdapter.updateMode(ppeAttr, vpux::VPU::PPEMode::ADD);

    // Manual update output type
    const auto outNCEPermuteShape = calculateWCHShape(outType.getShape().raw());
    outType = outType.changeShape(ShapeRef(outNCEPermuteShape));
    outType = outType.changeDimsOrder(DimsOrder::NWCH);

    //
    // Prepare output buffer for DPU
    //
    auto bufferType = vpux::getBufferType(outType);

    log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, outType);
    const auto outputBuffers =
            VPUIP::allocateBuffersOfType(log.nest(), origOp.getLoc(), rewriter, bufferType, /*individualBuffers=*/true);

    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;
    const auto isPermuteQuantizeAttr = mlir::UnitAttr::get(ctx);

    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);
    const auto loopAttributes = getLoopAttributes(origOp);

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");

    NCEClusterTaskParams params(viewOpIn.getResult(),
                                NCEClusterTaskParams::Weights{viewOpIn.getResult(), nullptr, nullptr, nullptr, nullptr},
                                outputBuffers, VPUIP::NCETaskType::ELTWISE,
                                NCEClusterTaskParams::Kernel{nullptr, nullptr, nullptr}, origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.isPermuteQuantize = isPermuteQuantizeAttr;
    params.isNCEPermute = true;
    params.mpeEngineAttr = mpeEngineAttr;
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    // ViewOp Output
    // Reshape to NxCxHxW
    // Layout change to NHWC
    auto viewOpOutType = mlir::cast<vpux::NDTypeInterface>(nceOp.getType()).changeDimsOrder(targetInOutOrder);
    viewOpOutType = viewOpOutType.changeShape(getShape(origOp.getOutput()));
    auto viewOpOut = rewriter.create<VPUIP::ViewOp>(origOp.getLoc(), viewOpOutType, nceOp);
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, viewOpOut.getResult());

    return mlir::success();
}

mlir::LogicalResult vpux::bufferizeOp(mlir::MLIRContext* ctx, VPU::NCEMatMulOp origOp,
                                      VPU::NCEMatMulOp::Adaptor& newArgs, mlir::RewriterBase& rewriter) {
    auto log = Logger::global().nest("one-shot-bufferize-NCEMatMulOp", 0);
    log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    //
    // Get dimensions
    //
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(origOp.getRawFilterShape()));

    const auto KY = filterShape[DimsGroups5D::Filter::KY];
    const auto KX = filterShape[DimsGroups5D::Filter::KX];

    //
    // Prepare output buffer for DPU
    //

    const auto outputBuffers =
            VPUIP::allocateBuffers(log, origOp.getLoc(), rewriter, {origOp.getOutput()}, /*individualBuffers=*/true);

    //
    // Create NCE per-cluster Operation
    //

    const auto kernelSizeAttr = getIntArrayAttr(ctx, ArrayRef({KY, KX}));
    const auto taskType = VPUIP::NCETaskType::CONV;
    auto ppeAttr = origOp.getPpeAttr();
    auto dpuCostAttr = origOp->hasAttr(DPUCost) ? origOp->getAttr(DPUCost) : nullptr;

    log.nest().trace("Creating VPUIP::NCEClusterTaskOp");
    mlir::UnitAttr isSuperdenseAttr = nullptr;
    if (isSuperdenseOp(origOp)) {
        VPUX_THROW_WHEN(mlir::isa<vpux::VPU::SparseTensorType>(origOp->getResult(0).getType()),
                        "Output cannot be sparse and super-dense at the same time");
        isSuperdenseAttr = mlir::UnitAttr::get(ctx);
    }

    const auto loopAttributes = getLoopAttributes(origOp);
    NCEClusterTaskParams params(
            newArgs.getInput(),
            NCEClusterTaskParams::Weights{newArgs.getWeights(), newArgs.getWeightsTable(),
                                          newArgs.getWeightTableScale(), newArgs.getWeightTableBias(), nullptr},
            outputBuffers, taskType,
            NCEClusterTaskParams::Kernel{kernelSizeAttr, origOp.getStrides(), origOp.getPadAttr()},
            origOp.getWorkloads());
    params.isSuperdenseAttr = isSuperdenseAttr;
    params.ppeAttr = ppeAttr;
    params.dpuCostAttr = dpuCostAttr;
    params.mpeEngineAttr = origOp.getMpeEngineAttr();
    params.tilingLoopIndex = loopAttributes.tilingLoopIndex;
    params.vfLoopIndex = loopAttributes.vfLoopIndex;
    params.vfLoopLayerIndex = loopAttributes.vfLoopLayerIndex;
    auto nceOp = createNCEClusterTask(rewriter, origOp->getLoc(), params, log);

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, origOp, nceOp);

    return mlir::success();
}

VPU::DistributedTensorType createCustomDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                             NDTypeInterface targetType,
                                                             VPU::DistributionInfoAttr origDistTensorAttr,
                                                             mlir::UnitAttr equalMemoryAndComputeView, ShapeRef shape) {
    auto* ctx = clusteredOp->getContext();

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto order = mlir::AffineMapAttr::get(targetType.getDimsOrder().toAffineMap(ctx));
    auto elemType = targetType.getElementType();

    const auto origDistTensorCtx = origDistTensorAttr.getContext();

    auto newNumTilesAttr = origDistTensorAttr.getNumTiles();
    if (newNumTilesAttr != nullptr) {
        auto numTiles = parseIntArrayAttr<int64_t>(newNumTilesAttr);
        newNumTilesAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(numTiles));
    }

    const auto activationTensorDistributionModeAttr =
            VPU::DistributionModeAttr::get(ctx, origDistTensorAttr.getMode().getValue());
    // Padding adaptions
    auto newPadAttr = origDistTensorAttr.getPads();
    if (newPadAttr != nullptr) {
        const auto fullInputChannels = mlir::cast<NDTypeInterface>(clusteredOp.getOperation()->getOperand(0).getType())
                                               .getShape()[Dims4D::Act::C];
        const auto fullOutputChannels = mlir::cast<NDTypeInterface>(clusteredOp.getOperation()->getResult(0).getType())
                                                .getShape()[Dims4D::Act::C];

        newPadAttr = VPU::getPaddingAttr(origDistTensorCtx, PadInfo(origDistTensorAttr.getPads().getTop().getInt(),
                                                                    origDistTensorAttr.getPads().getBottom().getInt(),
                                                                    0, fullOutputChannels - fullInputChannels));
    }
    auto newKernelAttr = origDistTensorAttr.getKernel();
    if (newKernelAttr != nullptr) {
        auto newKernel = parseIntArrayAttr<int64_t>(newKernelAttr);
        newKernelAttr = getIntArrayAttr(origDistTensorCtx,
                                        SmallVector<int64_t>{/*neutral val*/ 1, newKernel[Dims4D::Kernel::Y.ind()]});
    }
    auto newStridesAttr = origDistTensorAttr.getStrides();
    if (newStridesAttr != nullptr) {
        auto newStrides = parseIntArrayAttr<int64_t>(newStridesAttr);
        newStridesAttr = getIntArrayAttr(origDistTensorCtx,
                                         SmallVector<int64_t>{/*neutral val*/ 1, newStrides[Dims4D::Strides::Y.ind()]});
    }
    auto newAlignmentAttr = origDistTensorAttr.getAlignment();
    if (newAlignmentAttr != nullptr) {
        auto newAlignment = parseIntArrayAttr<int64_t>(newAlignmentAttr);
        newAlignmentAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(newAlignment));
    }

    auto calculateWCHShapeForArrayOfArray = [origDistTensorCtx](const mlir::ArrayAttr shape) -> mlir::ArrayAttr {
        if (shape != nullptr) {
            auto newIntShape = parseIntArrayOfArrayAttr<int64_t>(shape);
            for (size_t i = 0; i < newIntShape.size(); i++) {
                newIntShape[i] = calculateWCHShape(newIntShape[i]);
            }
            return getIntArrayOfArray(origDistTensorCtx, newIntShape);
        }
        return nullptr;
    };

    auto newMemoryNumTilesAttr = origDistTensorAttr.getMemoryNumTiles();
    if (newMemoryNumTilesAttr != nullptr) {
        auto memoryNumTiles = parseIntArrayAttr<int64_t>(newMemoryNumTilesAttr);
        newMemoryNumTilesAttr = getIntArrayAttr(origDistTensorCtx, calculateWCHShape(memoryNumTiles));
    }

    auto distributedTensorAttr = VPU::DistributionInfoAttr::get(
            ctx, activationTensorDistributionModeAttr, newNumTilesAttr, newKernelAttr, newPadAttr, newStridesAttr,
            origDistTensorAttr.getNumClusters(), newAlignmentAttr, origDistTensorAttr.getUniformDistributedSegments(),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getComputeShapes()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getComputeOffsets()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getMemoryShapes()),
            calculateWCHShapeForArrayOfArray(origDistTensorAttr.getMemoryOffsets()), equalMemoryAndComputeView,
            newMemoryNumTilesAttr);

    return VPU::DistributedTensorType::get(ctx, ArrayRef(calculateWCHShape(shape.raw())), elemType, order, memSpace,
                                           distributedTensorAttr);
}

//
// registerVpuNceBufferizableOpInterfaces
//

void vpux::registerVpuNceBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::NCEConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEConvolutionOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEAveragePoolOp>>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEDepthConvolutionOp>>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEInterpolateOp>>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCECompressConvolutionOp>>(
                *ctx);
        VPU::NCEPermuteOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEPermuteOp>>(*ctx);
        VPU::NCEMatMulOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEMatMulOp>>(*ctx);
        VPU::NCEReduceOp::attachInterface<VpuGenericOneShotBufferizeModel<VPU::NCEReduceOp>>(*ctx);
    });
}
