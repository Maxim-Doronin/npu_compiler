//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <npu_40xx_nnrt.hpp>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_REORDERMPIOPS
#define GEN_PASS_DEF_REORDERMPIOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;
using namespace npu40xx;

namespace {
class ReorderMPIOpsPass : public VPUMI40XX::impl::ReorderMPIOpsBase<ReorderMPIOpsPass> {
public:
    explicit ReorderMPIOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Operation* moveOrCloneOp(mlir::Operation* op, mlir::OpBuilder& builder) {
    const auto sameBlock = op->getBlock() == builder.getInsertionBlock();
    const auto moveSafe = sameBlock && mlir::isMemoryEffectFree(op);

    if (moveSafe) {
        op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
        return op;
    }

    auto* clonedOp = builder.clone(*op);
    op->replaceAllUsesWith(clonedOp);
    op->erase();
    return clonedOp;
}

template <typename OpT, typename Functor = vpux::FuncRef<bool(OpT)>>
mlir::Operation* linearizeOps(
        mlir::func::FuncOp func, mlir::OpBuilder& builder, Functor&& condition = [](OpT) {
            return true;
        }) {
    // Collect first to avoid iterator invalidation while moving ops.
    // Reserve a small default to avoid repeated growth for common cases.
    constexpr size_t expectedOpsToMove = 64;
    vpux::SmallVector<mlir::Operation*> opsToMove;
    opsToMove.reserve(expectedOpsToMove);
    for (auto op : func.getOps<OpT>()) {
        if (condition(op)) {
            opsToMove.push_back(op.getOperation());
        }
    }

    mlir::Operation* lastOp = nullptr;
    for (auto* op : opsToMove) {
        lastOp = moveOrCloneOp(op, builder);

        builder.setInsertionPointAfter(lastOp);
    }

    return lastOp;
}

// DeclareTaskBuffer ordering is particularly hot (many repeated scans). Bucket once and then
// emit/move in the required firmware order.
void linearizeDeclareTaskBufferOps(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
    using TaskType = VPURegMapped::TaskType;

    auto makeKey = [](size_t tileIdx, size_t listIdx, TaskType type) -> uint64_t {
        // Pack into a stable key: [ tileIdx (16) | listIdx (16) | taskType (32) ]
        return (static_cast<uint64_t>(tileIdx) << 48) | (static_cast<uint64_t>(listIdx) << 32) |
               static_cast<uint32_t>(type);
    };

    // Reserve for expected unique (tileIdx, listIdx, taskType) combinations.
    // Rule of thumb: VPU_MAX_TILES * expected_list_count * expected_task_types.
    // 256 keeps rehashing low for typical firmware list widths and task mix.
    constexpr size_t expectedBucketCount = 256;

    vpux::DenseMap<uint64_t, vpux::SmallVector<mlir::Operation*>> buckets;
    buckets.reserve(expectedBucketCount);

    // Bucket all DeclareTaskBufferOp once in original order.
    for (auto op : func.getOps<VPUMI40XX::DeclareTaskBufferOp>()) {
        const auto index = mlir::cast<vpux::VPURegMapped::IndexType>(op.getIndex().getType());
        const auto tileIdx = checked_cast<size_t>(index.getTileIdx());
        const auto listIdx = checked_cast<size_t>(index.getListIdx());
        const auto type = op.getTaskType();

        buckets[makeKey(tileIdx, listIdx, type)].push_back(op.getOperation());
    }

    auto moveBucket = [&](size_t tileIdx, size_t listIdx, TaskType type) {
        const auto key = makeKey(tileIdx, listIdx, type);
        auto it = buckets.find(key);
        if (it == buckets.end()) {
            return;
        }
        for (auto* op : it->second) {
            auto* lastOp = moveOrCloneOp(op, builder);
            builder.setInsertionPointAfter(lastOp);
        }
    };

    for (auto tileIndex : irange(nn_public::VPU_MAX_TILES)) {
        moveBucket(tileIndex, 0, TaskType::DPUInvariant);
        moveBucket(tileIndex, 0, TaskType::DPUVariant);

        moveBucket(tileIndex, 0, TaskType::ActKernelRange);
        moveBucket(tileIndex, 1, TaskType::ActKernelRange);

        moveBucket(tileIndex, 0, TaskType::ActKernelInvocation);
        moveBucket(tileIndex, 1, TaskType::ActKernelInvocation);

        moveBucket(tileIndex, 0, TaskType::DMA);
        moveBucket(tileIndex, 1, TaskType::DMA);
    }
}

template <VPURT::BufferSection SEC>
bool buffSec(VPURT::DeclareBufferOp op) {
    return op.getSection() == SEC;
}

template <int ENGINE_ID>
bool engineId(VPUMI40XX::NNDMAOp op) {
    return op.getPort() == ENGINE_ID;
}

template <VPURegMapped::TaskType TASK_TYPE>
auto taskType(size_t tileIndex, size_t listIndex = 0) {
    auto condition = [tileIndex, listIndex](VPUMI40XX::DeclareTaskBufferOp operation) {
        const auto index = mlir::cast<vpux::VPURegMapped::IndexType>(operation.getIndex().getType());
        return operation.getTaskType() == TASK_TYPE && index.getTileIdx() == tileIndex &&
               index.getListIdx() == listIndex;
    };
    return condition;
}

void ReorderMPIOpsPass::safeRunOnFunc() {
    auto func = getOperation();

    auto builder = mlir::OpBuilder::atBlockBegin(&func.getBody().front());

    // when ordering these function calls take care of the dominance relationships.
    // Ideally this should be a canonicalizer-like thing, but re-ordering ops via successive pattern rewrites//
    // seems overkill.
    // Since VPUMI40XX is a highly controlled dialect in terms of the forms it can take, manual implicit good
    // ordering of what to linearize first to not break dominance should be oke.

    // order of DeclareTaskBuffer is important as it must be aligned with firmware expectations
    // tile0: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
    // tile1: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
    // ...
    linearizeDeclareTaskBufferOps(func, builder);

    linearizeOps<Const::DeclareOp>(func, builder);

    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::NetworkInput>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::NetworkOutput>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::ProfilingOutput>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::DDR>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::CMX_NN>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::MAC_Accumulators>);
    linearizeOps<VPURT::DeclareBufferOp>(func, builder, buffSec<VPURT::BufferSection::Register>);

    linearizeOps<VPUMI40XX::DeclareKernelTextOp>(func, builder);
    linearizeOps<VPUMI40XX::DeclareKernelEntryOp>(func, builder);
    linearizeOps<VPUMI40XX::DeclareKernelArgsOp>(func, builder);
    linearizeOps<VPUMI40XX::KernelParamsOp>(func, builder);

    linearizeOps<VPUMI40XX::ConfigureBarrierOp>(func, builder);

    linearizeOps<VPUMI40XX::ActKernelRangeOp>(func, builder);
    linearizeOps<VPUMI40XX::ActKernelInvocationOp>(func, builder);

    linearizeOps<VPUMI40XX::DPUInvariantOp>(func, builder);
    linearizeOps<VPUMI40XX::DPUVariantOp>(func, builder);

    linearizeOps<VPURegMapped::ViewTaskRangeOp>(func, builder);

    linearizeOps<VPUMI40XX::NNDMAOp>(func, builder, engineId<0>);
    linearizeOps<VPUMI40XX::NNDMAOp>(func, builder, engineId<1>);
}

}  // namespace

//
// reorderMappedInferenceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::reorderMappedInferenceOpsPass(Logger log) {
    return std::make_unique<ReorderMPIOpsPass>(log);
}
