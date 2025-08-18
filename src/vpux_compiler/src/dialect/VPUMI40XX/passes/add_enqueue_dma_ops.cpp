//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"
namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDENQUEUEDMAOPS
#define GEN_PASS_DEF_ADDENQUEUEDMAOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {
class AddEnqueueDMAOps : public VPUMI40XX::impl::AddEnqueueDMAOpsBase<AddEnqueueDMAOps> {
public:
    explicit AddEnqueueDMAOps(Logger log): _dpuEnqueueDMACount(0), _shvEnqueueDMACount(0), _actShavePerTile(0) {
        Base::initLogger(log, Base::getArgumentName());
    }

    VPUMI40XX::NNDMAOp getInsertionDMAInPreviousPage(VPURegMapped::EnqueueOp enqueueOp, std::ostringstream& logStream);
    void createEnqueueNNDMAOp(mlir::OpBuilder& builder, mlir::Operation* dmaInsertionPoint,
                              mlir::Operation* bufferInsertionPoint, mlir::Operation* cstInsertionPoint,
                              VPUMI40XX::DeclareTaskBufferOp cmxTaskLocationBuf, uint32_t fifoAddr);
    void createDMAToPushTaskInFIFO(mlir::OpBuilder& builder, VPURegMapped::EnqueueOp enqueueOp,
                                   mlir::Operation* bufferInsertionPoint, mlir::Operation* cstInsertionPoint,
                                   std::ostringstream& logStream);
    mlir::Value getOrCreateRegisterBuffer(mlir::OpBuilder& builder, mlir::Operation* bufferInsertionPoint,
                                          mlir::MemRefType memType, uint32_t fifoAddr);

private:
    void safeRunOnFunc() final;
    llvm::DenseMap<int, VPUMI40XX::NNDMAOp> _lastDmaTile0List0ByPage;
    llvm::DenseMap<mlir::Value, VPUMI40XX::NNDMAOp> _opFetchDMAMap;
    llvm::DenseMap<std::pair<mlir::Type, uint32_t>, mlir::Value> _regBufferCache;

    size_t _dpuEnqueueDMACount;
    size_t _shvEnqueueDMACount;
    int64_t _actShavePerTile;
};

uint32_t getFifoAddr(VPURegMapped::TaskOpInterface taskOp, size_t tile, size_t list, int64_t actShavePerTile) {
    VPUX_THROW_WHEN(tile >= VPUMI40XX::NPU_MAX_TILES || list > 1, "Invalid tile index {0} or list index {1} for Op {3}",
                    tile, list, taskOp);
    const auto type = taskOp.getTaskType();
    const auto offset = type == VPURegMapped::TaskType::DPUVariant
                                ? VPUMI40XX::DPU_FIFO_OFFSETS[tile]
                                : VPUMI40XX::SHV_FIFO_OFFSETS[(actShavePerTile * tile) + list];
    const auto base = type == VPURegMapped::TaskType::DPUVariant ? VPUMI40XX::NNCMX_DPU_CMX_CTRL_BASE
                                                                 : VPUMI40XX::NNCMX_SHV_CMX_CTRL_BASE;
    return base + offset;
}

// If the `VPUMI40XX_ExecutableTaskOpInterface` had been assigned to `DPUVariant`,
// it would have been possible to avoid the use of switch statements in this function.
// However, the interface was not assigned to `DPUVariant`, nor was the `wlmPage` method
// defined for it, as barrier management is not yet implemented for `DPUVariant`.
// Although it is theoretically possible for each variant of the same invariant
// to have its own barriers in the future, this functionality is not part of the current
// design. Therefore, the interface was not used, and the switch statement remains as
// the current approach for handling different task types

// Returns the WLM page of the task associated with the given EnqueueOp.
int getWlmPageForTaskEnqueued(VPURegMapped::EnqueueOp enqueueOp) {
    auto taskType = enqueueOp.getTaskType();
    int wlmPage = -1;

    switch (taskType) {
    case VPURegMapped::TaskType::DPUVariant: {
        auto variantOp = mlir::cast<VPUMI40XX::DPUVariantOp>(enqueueOp.getStart().getDefiningOp());
        wlmPage = variantOp.getWlmPage().value();
        break;
    }
    case VPURegMapped::TaskType::ActKernelInvocation: {
        auto kernelRangeOp = mlir::cast<VPUMI40XX::ActKernelInvocationOp>(enqueueOp.getStart().getDefiningOp());
        wlmPage = kernelRangeOp.getWlmPage().value();
        break;
    }
    default:
        VPUX_THROW("Unsupported Type: {0}", stringifyTaskType(taskType));
    }
    return wlmPage;
}

VPUMI40XX::NNDMAOp getDMATypeUser(VPURegMapped::ViewTaskRangeOp taskRangeOp) {
    auto validDMAUser = [](mlir::Operation* op) {
        if (auto dmaOp = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(op)) {
            return true;
        }
        return false;
    };
    auto validTaskRangeUsers = taskRangeOp.getResult().getUsers() | vpux::filtered(std::move(validDMAUser));
    if (!validTaskRangeUsers.empty()) {
        return mlir::cast<VPUMI40XX::NNDMAOp>(*validTaskRangeUsers.begin());
    }
    VPUX_THROW("ViewTaskRangeOp is expected to have atleast 1 DMAType user, but has none!");
}

VPURegMapped::ViewTaskRangeOp getViewTaskRangeTypeUser(VPURegMapped::TaskOpInterface taskOp) {
    auto validTaskRangeUser = [](mlir::Operation* op) {
        return mlir::isa<VPURegMapped::ViewTaskRangeOp>(op);
    };
    auto validVariantUsers = taskOp.getResult().getUsers() | vpux::filtered(std::move(validTaskRangeUser));
    if (!validVariantUsers.empty()) {
        return mlir::cast<VPURegMapped::ViewTaskRangeOp>(*validVariantUsers.begin());
    }
    return nullptr;
}

// Function returns the associated Fetch DMAOp for the task enqueued by enqueueOp
//
// Fetch(%0-%61)
// ... More Ops
// Enqueue(%0-%10)
// Enqueue(%11-%20)
// Enqueue(%21-%61)
//
// Enqueue1 will find FetchOp using start() -> %0
// Enqueue2 will find FetchOp by traversing backwards leading to fetch op found by Enqueue1
// Enqueue3 will find FetchOp by traversing backwards leading to fetch op found by Enqueue2

// Since the enqueues are ordered it is safe to assume we will always find a fetch op by traversing backwards
VPUMI40XX::NNDMAOp getAssociatedFetchDMAOp(VPURegMapped::EnqueueOp enqueueOp,
                                           llvm::DenseMap<mlir::Value, VPUMI40XX::NNDMAOp>& opFetchDMAMap) {
    VPUMI40XX::NNDMAOp associatedFetchOp = nullptr;
    auto currentTask = enqueueOp.getStart();
    while (currentTask) {
        if (opFetchDMAMap.count(currentTask) > 0) {
            associatedFetchOp = opFetchDMAMap[currentTask];
            break;
        }

        auto taskOp = mlir::cast<VPURegMapped::TaskOpInterface>(currentTask.getDefiningOp());
        auto taskRangeOp = getViewTaskRangeTypeUser(taskOp);
        if (taskRangeOp != nullptr) {
            associatedFetchOp = getDMATypeUser(taskRangeOp);
            break;
        }
        currentTask = taskOp.getPreviousTask().getResult();
    }

    opFetchDMAMap[currentTask] = associatedFetchOp;
    return associatedFetchOp;
}

Const::DeclareOp createEnqueueConstant(mlir::OpBuilder& builder, mlir::Operation* insertionPoint, const uint32_t& val) {
    const Shape valShape = {1};
    const auto dataStorageType = mlir::RankedTensorType::get(valShape.raw(), getUInt32Type(builder.getContext()));
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, ArrayRef(val));

    auto memType = mlir::MemRefType::get(dataStorageType.getShape(), dataStorageType.getElementType());
    builder.setInsertionPoint(insertionPoint);
    auto configurationConstOp =
            builder.create<Const::DeclareOp>(builder.getUnknownLoc(), memType, Const::ContentAttr::get(dataAttr));

    return configurationConstOp;
}

mlir::Value AddEnqueueDMAOps::getOrCreateRegisterBuffer(mlir::OpBuilder& builder, mlir::Operation* bufferInsertionPoint,
                                                        mlir::MemRefType memType, uint32_t fifoAddr) {
    std::pair<mlir::Type, uint32_t> key = {memType, fifoAddr};

    auto it = _regBufferCache.find(key);
    if (it != _regBufferCache.end()) {
        return it->second;
    }

    builder.setInsertionPoint(bufferInsertionPoint);
    auto declBuf = builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), memType,
                                                          VPURT::BufferSection::Register, fifoAddr);

    mlir::Value buffer = declBuf.getBuffer();
    _regBufferCache[key] = buffer;
    return buffer;
}

void AddEnqueueDMAOps::createEnqueueNNDMAOp(mlir::OpBuilder& builder, mlir::Operation* dmaInsertionPoint,
                                            mlir::Operation* bufferInsertionPoint, mlir::Operation* cstInsertionPoint,
                                            VPUMI40XX::DeclareTaskBufferOp cmxTaskLocationBuf, uint32_t fifoAddr) {
    auto ctx = builder.getContext();
    auto insertionDmaOp = mlir::cast<VPUMI40XX::NNDMAOp>(dmaInsertionPoint);

    // ----------------------------------------
    // Step 1: Prepare DMA-specific Attributes
    // ----------------------------------------

    // Convert CMX offset to metadata-relative offset in 32-byte units.
    // 15360 = VPU_METADATA_OFFSET = start of metadata region in CMX.
    const auto descriptorOffsetInCMX = (cmxTaskLocationBuf.getOffset().value() + 15360) >> 5;
    auto enqueueConstOp = createEnqueueConstant(builder, cstInsertionPoint, descriptorOffsetInCMX);
    const auto constOutputType = mlir::cast<vpux::NDTypeInterface>(enqueueConstOp.getOutput().getType());

    const auto layout = mlir::MemRefLayoutAttrInterface{};
    const auto regMemSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::Register));
    auto outputType = constOutputType.changeMemSpace(regMemSpace);

    // ----------------------------------------
    // Step 2: Declare Buffers for Register
    // ----------------------------------------

    builder.setInsertionPoint(bufferInsertionPoint);
    auto memType = mlir::MemRefType::get(outputType.getShape().raw(), outputType.getElementType(), layout,
                                         outputType.getMemSpace());
    mlir::Value dstBuffer = getOrCreateRegisterBuffer(builder, bufferInsertionPoint, memType, fifoAddr);

    // ----------------------------------------
    // Step 3: Create New DMA Operation
    // ----------------------------------------

    builder.setInsertionPointAfter(insertionDmaOp);

    const auto wlmPage = insertionDmaOp.getWlmPage().value();
    const auto dmaDescriptorAttr = VPUIP::DMADescriptorAttr::get(ctx,
                                                                 /*numPlane*/ getIntAttr(ctx, 0),
                                                                 /*len*/ getIntAttr(ctx, 4),
                                                                 /*srcWidth*/ getIntAttr(ctx, 4),
                                                                 /*srcStride*/ getIntAttr(ctx, 0),
                                                                 /*srcPlaneStride*/ getIntAttr(ctx, 0),
                                                                 /*dstWidth*/ getIntAttr(ctx, 4),
                                                                 /*dstStride*/ getIntAttr(ctx, 0),
                                                                 /*dstPlaneStride*/ getIntAttr(ctx, 0));

    const auto dmaIndexAttr = VPURegMapped::IndexType::get(ctx, 0, 0, 0);
    auto newDMA = builder.create<VPUMI40XX::NNDMAOp>(
            builder.getUnknownLoc(), dmaIndexAttr, nullptr, enqueueConstOp.getOutput(), dstBuffer,
            insertionDmaOp.getResult(), mlir::ValueRange{}, mlir::ValueRange{}, 0, 0, false, false, false, 0,
            VPUIP::DMAAccMode::DISABLE, nullptr, nullptr, nullptr, dmaDescriptorAttr, nullptr, nullptr, 0, nullptr,
            nullptr, insertionDmaOp.getWlmPageAttr());

    // ----------------------------------------
    // Step 4: Update Internal State and IR
    // ----------------------------------------

    // Updating the last DMA ensures that any new enqueue task inserted into this page will be placed after all
    // previously inserted enqueue DMAs.
    _lastDmaTile0List0ByPage[wlmPage] = newDMA;
    insertionDmaOp.getResult().replaceAllUsesExcept(newDMA.getIndex(), newDMA.getOperation());

    auto newDMATaskOp = mlir::cast<VPURegMapped::TaskOpInterface>(newDMA.getOperation());
    newDMATaskOp.linkToPreviousTask();
}

// Returns the DMA op from the previous WLM page to use as the insertion point for the given EnqueueOp.
//
// This function selects the last DMA from the previous page, based on guarantees established
// by WlmInsertDummyDmasInPages. Each page is ensured to contain at least one DMA on tile 0, list 0.
// The last DMA in the page is constructed such that it directly or transitively depends on all boundary tasks from the
// preceding page.
//
// This means that inserting the enqueue DMA after the last DMA guarantees all barriers from the previous
// page have been consumed, ensuring safe task scheduling across page boundaries.
VPUMI40XX::NNDMAOp AddEnqueueDMAOps::getInsertionDMAInPreviousPage(VPURegMapped::EnqueueOp enqueueOp,
                                                                   std::ostringstream& logStream) {
    auto wlmPageForEnqueueTask = getWlmPageForTaskEnqueued(enqueueOp);
    logStream << "\n\t - Task to be Enqueued in page: " << wlmPageForEnqueueTask;

    auto prevPage = wlmPageForEnqueueTask - 1;
    auto it = _lastDmaTile0List0ByPage.find(prevPage);
    if (it != _lastDmaTile0List0ByPage.end()) {
        return it->second;
    }

    // WLM page split legalization passes are expected to guarantee there is DMA in every page.
    VPUX_THROW("Legalization failed! Could not find a DMA in page: {0}", prevPage);
}

void AddEnqueueDMAOps::createDMAToPushTaskInFIFO(mlir::OpBuilder& builder, VPURegMapped::EnqueueOp enqueueOp,
                                                 mlir::Operation* bufferInsertionPoint,
                                                 mlir::Operation* cstInsertionPoint, std::ostringstream& logStream) {
    auto enqueueBarrierOp = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(enqueueOp.getBarrier().getDefiningOp());
    auto enqueueIndexAttr = mlir::cast<VPURegMapped::IndexType>(enqueueOp.getIndex().getType());
    logStream << "Enqueue Operation Info: [" << enqueueIndexAttr.getValue() << "]";
    if (enqueueBarrierOp) {
        logStream << "\n\t - Enqueue at ViD: " << enqueueBarrierOp.getResult().getType().getValue()
                  << " ViD in Page: " << enqueueBarrierOp.getWlmPage().value();
    } else {
        logStream << "\n\t - Enqueue at BOOTSTRAP";
    }

    auto startOp = mlir::cast<VPURegMapped::TaskOpInterface>(enqueueOp.getStart().getDefiningOp());
    auto associatedFetchOp = getAssociatedFetchDMAOp(enqueueOp, _opFetchDMAMap);
    auto cmxTaskLocationBuf = startOp.getTaskLocation();
    const auto tileIdx = startOp.getIndexType().getTileIdx();
    const auto listIdx = startOp.getIndexType().getListIdx();
    auto fifoAddr = getFifoAddr(startOp, tileIdx, listIdx, _actShavePerTile);

    if (startOp.getTaskType() == VPURegMapped::TaskType::DPUVariant) {
        logStream << "\n\tEnqueue is for DPU (Tile: " << tileIdx << ") fifoAddr: " << std::hex << fifoAddr << std::dec;
        ++_dpuEnqueueDMACount;
    } else {
        logStream << "\n\tEnqueue is for SHV (Tile: " << tileIdx << ", List: " << listIdx << ") fifoAddr: " << std::hex
                  << fifoAddr << std::dec;
        ++_shvEnqueueDMACount;
    }

    auto insertionDMA = getInsertionDMAInPreviousPage(enqueueOp, logStream);
    auto insertionDMAIndexAttr = mlir::cast<VPURegMapped::IndexType>(insertionDMA.getIndex().getType());
    auto fetchIndexAttr = mlir::cast<VPURegMapped::IndexType>(
            mlir::cast<mlir::TypedValue<mlir::Type>>(associatedFetchOp->getResult(0)).getType());
    logStream << "\n\tInsertion DMA located at [" << insertionDMAIndexAttr.getValue()
              << "] in page: " << insertionDMA.getWlmPage().value();
    logStream << "\n\tAssociated FetchDMA located at [" << fetchIndexAttr.getValue()
              << "] in page: " << associatedFetchOp.getWlmPage().value();

    auto cmxTaskLocation = mlir::cast<VPUMI40XX::DeclareTaskBufferOp>(cmxTaskLocationBuf.getDefiningOp());
    createEnqueueNNDMAOp(builder, insertionDMA, bufferInsertionPoint, cstInsertionPoint, cmxTaskLocation, fifoAddr);

    logStream << "\n";
}

void AddEnqueueDMAOps::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto module = netFunc->getParentOfType<mlir::ModuleOp>();
    auto tileOp = vpux::IE::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Expected tileOp executor in order to query SHAVE_ACT executor.");
    VPUX_THROW_UNLESS(tileOp.hasSubExecutor(VPU::ExecutorKind::SHAVE_ACT),
                      "No SHAVE_ACT executor found, check your arch");
    auto actShaveExec = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
    _actShavePerTile = actShaveExec.getCount();

    auto builder = mlir::OpBuilder(mpi.getOperation());
    std::ostringstream logStream;

    auto head = mlir::cast<VPURegMapped::TaskOpInterface>(
            mpi.getListHead(VPURegMapped::TaskType::DMA, 0, 0).getDefiningOp());

    auto isBefore = [](VPUMI40XX::NNDMAOp lhs, VPUMI40XX::NNDMAOp rhs) {
        return lhs.getType().getValue() < rhs.getType().getValue();
    };

    // Get last DMAs on tile 0 list 0 for each page
    for (auto current = head; current != nullptr; current = current.getNextTask()) {
        auto dmaOp = mlir::cast<VPUMI40XX::NNDMAOp>(current.getOperation());
        VPUX_THROW_WHEN(dmaOp.getWlmPageAttr() == nullptr,
                        "Each Op is expected to have wlmPage assigned, '{0}' is missing wlmPage attribute", dmaOp);

        auto page = dmaOp.getWlmPage().value();
        auto it = _lastDmaTile0List0ByPage.find(page);

        // Insert dmaOp if it's the first for this page, or replace the existing one
        // if dmaOp comes later according to custom isBefore() ordering
        if (it == _lastDmaTile0List0ByPage.end() || isBefore(it->second, dmaOp)) {
            _lastDmaTile0List0ByPage[page] = dmaOp;
        }
    }

    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto declOps = netFunc.getOps<Const::DeclareOp>();
    auto cstInsertionPoint = !declOps.empty() ? *declOps.begin() : &netFunc.getBody().front().front();

    // Since we will delete enqueues for DPU/SHV, keep track of previous DMA enqueue to adjust previousTaskIdx
    VPURegMapped::EnqueueOp previousDMAEnqueue = nullptr;
    SmallVector<VPURegMapped::EnqueueOp> enqueueToErase;

    // It is crucial to process enqueue operations in IR order
    // This ensures that items are enqueued in the correct sequence, preventing the risk of enqueuing operations
    // "too early" before the required data or descriptors are fully available.
    for (auto enqueueOp : llvm::make_early_inc_range(netFunc.getOps<VPURegMapped::EnqueueOp>())) {
        auto taskType = enqueueOp.getTaskType();
        switch (taskType) {
        case VPURegMapped::TaskType::DPUVariant:
        case VPURegMapped::TaskType::ActKernelInvocation:
            createDMAToPushTaskInFIFO(builder, enqueueOp, bufferInsertionPoint, cstInsertionPoint, logStream);
            enqueueOp.getPreviousTaskIdxMutable().clear();
            enqueueToErase.push_back(enqueueOp);
            break;
        case VPURegMapped::TaskType::DMA:
            if (!enqueueOp.getPreviousTaskIdxMutable().empty()) {
                enqueueOp.getPreviousTaskIdxMutable().assign(previousDMAEnqueue);
            }
            previousDMAEnqueue = enqueueOp;
            break;
        default:
            VPUX_THROW("Unsupported Type");
        }
    }

    // Erase EnqueueOps which now has corrsponding Enqueue DMA
    std::for_each(enqueueToErase.begin(), enqueueToErase.end(), [](auto& enqueueOp) {
        enqueueOp.erase();
    });

    _log.trace("{0}", logStream.str());
    VPUMI40XX::reindexList<VPUMI40XX::NNDMAOp>(mpi, mlir::cast<VPUMI40XX::NNDMAOp>(head.getOperation()), 0, 0);
    VPUMI40XX::reindexTaskLinkAttrForDMA(head);

    auto firstEnqu = mpi.getWorkItemTasks();
    auto newCount = VPUMI40XX::reindexEnqueueList(mlir::cast<VPURegMapped::EnqueueOp>(firstEnqu.getDefiningOp()));
    mpi.setWorkItemCount(newCount);

    _log.info("Inserted {0} Enqueue DMAs for DPUTasks and {1} Enqueue DMAs for SHVTasks", _dpuEnqueueDMACount,
              _shvEnqueueDMACount);
}

}  // namespace

//
// createAddEnqueueDMAOps
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddEnqueueDMAOps(Logger log) {
    return std::make_unique<AddEnqueueDMAOps>(log);
}
