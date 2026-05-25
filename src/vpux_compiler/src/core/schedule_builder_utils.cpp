//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/JSON.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include "vpux/utils/logger/logger.hpp"

using namespace vpux;

namespace {
VPURT::TaskQueueType getQueueType(mlir::async::ExecuteOp execOp) {
    VPURT::TaskQueueType queueType = {};
    queueType.type = VPUIP::getExecutorType(execOp);

    if (auto dmaTask = VPUIP::getDmaTypeOp(execOp)) {
        queueType.id = getDMAQueueIdEncoding(dmaTask.getChannelType());
    }
    return queueType;
}

AllocationType getAllocationType(mlir::async::ExecuteOp execOp, bool hasValidDep) {
    // skip memory movement ops
    if (VPUIP::isDmaDDR2CMX(execOp)) {
        return AllocationType::DATA_IN;
    }

    if (hasValidDep && VPUIP::isDmaCMX2DDR(execOp)) {
        // valid dependency needed for DMA->DMA patterns
        return AllocationType::DATA_OUT;
    }

    return AllocationType::COMPUTE;
}

std::pair<SmallVector<mlir::Value>, SmallVector<mlir::Value>> getOperationBuffers(mlir::async::ExecuteOp execOp,
                                                                                  AliasesInfo& aliasInfo) {
    SmallVector<mlir::Value> inBuffers;
    SmallVector<mlir::Value> outBuffers;

    auto isTargetMemType = [&](mlir::Value buf) {
        auto bufType = getAsyncValueType(buf);
        auto bufNDType = mlir::dyn_cast<vpux::NDTypeInterface>(bufType);

        if (bufNDType == nullptr) {
            return false;
        }

        return bufNDType.getMemoryKind() == VPU::MemoryKind::CMX_NN;
    };

    auto updateBufferStorage = [&](const SmallVector<mlir::Value>& buffers, SmallVector<mlir::Value>& bufferStorage,
                                   mlir::DenseSet<mlir::Value>& cache) {
        bufferStorage.reserve(bufferStorage.size() + buffers.size());
        for (const auto& buffer : buffers) {
            if (!isTargetMemType(buffer)) {
                continue;
            }
            const auto& roots = aliasInfo.getRoots(buffer);
            if (roots.size() == 1 && roots.front() == buffer) {
                if (cache.count(buffer)) {
                    continue;
                }
                cache.insert(buffer);
                bufferStorage.push_back(buffer);
            } else {
                if (cache.count(aliasInfo.getRoot(buffer))) {
                    continue;
                }
                cache.insert(aliasInfo.getRoot(buffer));
                bufferStorage.push_back(aliasInfo.getRoot(buffer));
            }
        }
    };

    auto* bodyBlock = execOp.getBody();

    mlir::DenseSet<mlir::Value> inBuffersCache;
    mlir::DenseSet<mlir::Value> outBuffersCache;

    for (auto& innerOp : bodyBlock->getOperations()) {
        if (auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(innerOp)) {
            auto inputs = VPUIP::getInputsSanitized(layerOp);
            auto outputs = layerOp.getOutputs();

            updateBufferStorage(inputs, inBuffers, inBuffersCache);
            updateBufferStorage(outputs, outBuffers, outBuffersCache);
        }
    }

    return {std::move(inBuffers), std::move(outBuffers)};
}

bool hasNonDmaDependency(ArrayRef<size_t> depInd, const mlir::DenseSet<size_t>& nonDmaOps) {
    for (auto& depIdx : depInd) {
        if (nonDmaOps.find(depIdx) != nonDmaOps.end()) {
            return true;
        }
    }
    return false;
}

std::unique_ptr<SchedulingLoop> makeSchedulingLoopFromIterations(const std::vector<LoopBody>& iterations,
                                                                 LoopType type) {
    auto loop = std::make_unique<SchedulingLoop>();
    loop->type = type;
    loop->loopBodies.reserve(iterations.size());
    for (const auto& iter : iterations) {
        LoopBody body;
        body.reserve(iter.size());
        for (const auto& opAllocationInfo : iter) {
            body.emplace_back(opAllocationInfo);
        }
        loop->loopBodies.emplace_back(std::move(body));
    }
    return loop;
}

OpAllocationInfo createAllocInfo(size_t opIdx, AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo,
                                 [[maybe_unused]] Logger log) {
    const auto execOp = depsInfo.getExecuteOpAtIndex(opIdx);
    const auto dependencies = depsInfo.getOpDeps(opIdx);
    // has non-dma dependency
    bool hasNonDmaDep = false;
    hasNonDmaDep = std::any_of(dependencies.begin(), dependencies.end(), [&](size_t depIdx) {
        return VPUIP::getExecutorType(depIdx, depsInfo) != config::ExecutorKind::DMA_NN;
    });

    const auto queueType = getQueueType(execOp);
    const auto consumers = depsInfo.getConsumerOps(opIdx);
    auto [inBuffers, outBuffers] = getOperationBuffers(execOp, aliasInfo);
    const auto allocationType = getAllocationType(execOp, hasNonDmaDep);

#ifdef VPUX_DEVELOPER_BUILD
    SmallVector<mlir::Value> inBufferVals;
    inBufferVals.reserve(inBuffers.size());
    for (auto& buf : inBuffers) {
        inBufferVals.push_back(buf);
    }

    SmallVector<mlir::Value> outBufferVals;
    outBufferVals.reserve(outBuffers.size());
    for (auto& buf : outBuffers) {
        outBufferVals.push_back(buf);
    }

    log.trace("createAllocInfo: Op {0} executor {1} type {2} inBuffers {3} outBuffers {4} "
              "deps {5} cons {6}",
              opIdx, queueType.type, toString(allocationType), inBufferVals, outBufferVals, dependencies, consumers);
#endif

    return OpAllocationInfo(opIdx, queueType, inBuffers, outBuffers, allocationType);
}

/**
 *   @name createInnerLoopsFromIterations
 *   @brief create compute regions from loop iterations based on shared dependencies and consumers
 *
 *   @param loop - vector of LoopBody representing iterations
 *   @param allLoopOperations - set of all operations inside the loop
 *   @param depsInfo - async dependencies info
 *   @param nonDmaOps - set of all non-dma operation indexes
 *   @return map of compute regions grouped by insertion points,
 *           insertion point is the first compute operation index in the loop
 *
 *   @details
 *   Algorithm:
 *   for: every iteration in the loop
 *       get global deps and cons for the iteration
 *       for every other iteration in the loop
 *           compare local buffer counts with current iteration
 *           if they match add other iteration to matching iterations
 *       create compute region from matching iterations
 *
 *   Matching criteria considerations:
 *       - the same number of local buffers (raw and deduplicated)
 *       - no dependency-consumer conflicts (max dep < min con)
 *
 *   Limitations and assumptions:
 *       - Matching criteria met for all merged iterations
 *       - Minimum number of operations in the loop: MIN_LOOP_OPS
 *       - Loops are created inplace of first compute operation in the loop
 */

std::unordered_map<size_t, ComputeRegionVec> createInnerLoopsFromIterations(ArrayRef<LoopBody> loop,
                                                                            llvm::DenseSet<size_t>& allLoopOperations,
                                                                            AsyncDepsInfo& depsInfo,
                                                                            mlir::DenseSet<size_t>& nonDmaOps,
                                                                            Logger log) {
    // Precompute global deps/cons for each iteration to avoid recomputing inside matching loop
    SmallVector<std::set<size_t>> cachedGlobalDeps(loop.size());
    SmallVector<std::set<size_t>> cachedGlobalCons(loop.size());

    for (size_t iterIdx = 0; iterIdx < loop.size(); ++iterIdx) {
        // deps
        llvm::DenseSet<size_t> loopBodyOpsIdxs;
        for (const auto& op : loop[iterIdx]) {
            loopBodyOpsIdxs.insert(op.opIdx);
        }
        std::set<size_t> loopGlobalDeps;
        for (const auto& op : loop[iterIdx]) {
            for (const auto& dep : depsInfo.getOpDeps(op.opIdx)) {
                if (loopBodyOpsIdxs.count(dep) == 0) {
                    loopGlobalDeps.insert(dep);
                }
            }
        }
        cachedGlobalDeps[iterIdx] = std::move(loopGlobalDeps);

        // cons
        std::set<size_t> loopGlobalCons;
        for (const auto& op : loop[iterIdx]) {
            const auto execOp = depsInfo.getExecuteOpAtIndex(op.opIdx);
            const auto deps = depsInfo.getOpDeps(op.opIdx);
            const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, nonDmaOps));
            if (allocationType == AllocationType::DATA_IN) {
                continue;
            }
            for (const auto& con : depsInfo.getConsumerOps(op.opIdx)) {
                if (VPUIP::isDmaDDR2DDR(depsInfo.getExecuteOpAtIndex(con))) {
                    // TODO (E#203341): optimize pattern:

                    //   COMPUTE         COMPUTE
                    //     |                |
                    // DMA(CMX2DDR)     DMA(CMX2DDR)
                    //      \            /
                    //       DMA(DDR2DDR)

                    // with loop logic such DDR2DDR DMAs cause stalls
                    return {};
                }
                if (loopBodyOpsIdxs.count(con) == 0) {
                    loopGlobalCons.insert(con);
                }
            }
        }
        cachedGlobalCons[iterIdx] = std::move(loopGlobalCons);
    }

    auto getGlobalDeps = [&](size_t iterationIdx) {
        VPUX_THROW_UNLESS(iterationIdx < cachedGlobalDeps.size(), "Invalid loop iteration index {0}", iterationIdx);
        return cachedGlobalDeps[iterationIdx];
    };

    auto getGlobalDepsForLoop = [&](const std::vector<LoopBody>& loop, const std::set<size_t>& matchingIters) {
        // Collect all op indices across merged iterations
        llvm::DenseSet<size_t> loopOps;
        for (const auto& iteration : loop) {
            for (const auto& op : iteration) {
                loopOps.insert(op.opIdx);
            }
        }
        // Reuse cached per-iteration global deps, filter out cross-iteration references
        llvm::DenseSet<size_t> loopGlobalDeps;
        for (auto iterIdx : matchingIters) {
            for (auto dep : cachedGlobalDeps[iterIdx]) {
                if (loopOps.count(dep) == 0) {
                    loopGlobalDeps.insert(dep);
                }
            }
        }
        return loopGlobalDeps;
    };

    auto getGlobalCons = [&](size_t iterationIdx) {
        VPUX_THROW_UNLESS(iterationIdx < cachedGlobalCons.size(), "Invalid loop iteration index {0}", iterationIdx);
        return cachedGlobalCons[iterationIdx];
    };

    auto getComputeAlloc = [&](const LoopBody& iteration) {
        for (const auto& op : iteration) {
            if (op.allocationType == AllocationType::COMPUTE) {
                return op;
            }
        }
        VPUX_THROW("No compute op in loop");
    };

    auto getDedupBufferCount = [](ArrayRef<mlir::Value> buffers) -> size_t {
        mlir::DenseSet<mlir::Value> uniqueBuffers;
        for (const auto& buf : buffers) {
            uniqueBuffers.insert(buf);
        }
        return uniqueBuffers.size();
    };

    log.trace("Creating compute regions from iterations = {0}", loop.size());
    // Merge individual iterations into 1D loops based on shared dependencies and consumers
    std::unordered_map<size_t, ComputeRegionVec> computeRegions;
    llvm::DenseSet<size_t> handledIterations;
    for (size_t currentIdx = 0; currentIdx < loop.size(); ++currentIdx) {
        if (handledIterations.count(currentIdx) > 0) {
            log.trace("handled iteration idx {0}", currentIdx);
            continue;
        }
        const auto& currentIteration = loop[currentIdx];

        const auto computeAlloc = getComputeAlloc(currentIteration);
        const auto iterationDeps = getGlobalDeps(currentIdx);
        const auto iterationGlobalCons = getGlobalCons(currentIdx);

        auto lastDependOpIdx = iterationDeps.empty() ? iterationDeps.end() : std::prev(iterationDeps.end());
        auto firstConsumerOpIdx = iterationGlobalCons.begin();

        if (lastDependOpIdx != iterationDeps.end() && firstConsumerOpIdx != iterationGlobalCons.end() &&
            *lastDependOpIdx >= *firstConsumerOpIdx) {
            continue;
        }

        std::set<size_t> matchingIterations;
        matchingIterations.insert(currentIdx);

        // Check if any other iteration uses these deps.
        for (size_t otherIdx = 0; otherIdx < loop.size(); ++otherIdx) {
            if (handledIterations.count(otherIdx) > 0 || otherIdx == currentIdx) {
                log.trace("skip otherIdx {0}", otherIdx);
                continue;
            }
            const auto& otherIteration = loop[otherIdx];
            const auto otherComputeAlloc = getComputeAlloc(otherIteration);
            // all matching iterations must have the same number of local buffers
            if (computeAlloc.inBuffers.size() != otherComputeAlloc.inBuffers.size() ||
                computeAlloc.outBuffers.size() != otherComputeAlloc.outBuffers.size()) {
                log.trace("skip otherIdx {0} different number of local buffers", otherIdx);
                continue;
            }

            // Use deduplicated counts to handle repeated operands that alias the same buffer.
            const auto currentDedupInCount = getDedupBufferCount(computeAlloc.inBuffers);
            const auto currentDedupOutCount = getDedupBufferCount(computeAlloc.outBuffers);
            const auto otherDedupInCount = getDedupBufferCount(otherComputeAlloc.inBuffers);
            const auto otherDedupOutCount = getDedupBufferCount(otherComputeAlloc.outBuffers);
            if (currentDedupInCount != otherDedupInCount || currentDedupOutCount != otherDedupOutCount) {
                log.trace(
                        "skip otherIdx {0} different dedup local buffers, current in/out {1}/{2}, other in/out {3}/{4}",
                        otherIdx, currentDedupInCount, currentDedupOutCount, otherDedupInCount, otherDedupOutCount);
                continue;
            }

            // loops can be merged
            matchingIterations.insert(otherIdx);
        }

        if (matchingIterations.size() < MIN_LOOP_OPS) {
            log.trace("skip currentIdx {0} only {1} matching iterations", currentIdx, matchingIterations.size());
            continue;
        }

        // Merge matching iterations into a single loop and mark their ops as handled
        std::vector<LoopBody> mergedIterations;
        for (auto matchIdx : matchingIterations) {
            handledIterations.insert(matchIdx);
            LoopBody tempIteration;
            for (auto& op : loop[matchIdx]) {
                tempIteration.push_back(op);
                allLoopOperations.insert(op.opIdx);
            }
            log.nest().trace("Add op to loop {0}", depsInfo.getExecuteOpAtIndex(tempIteration[0].opIdx).getLoc());
            mergedIterations.push_back(std::move(tempIteration));
        }

        // Insert inplace of first compute
        size_t insertionPoint = std::numeric_limits<size_t>::max();
        for (auto& op : mergedIterations[0]) {
            if (op.allocationType == AllocationType::COMPUTE) {
                insertionPoint = std::min(insertionPoint, op.opIdx);
            }
        }

        const auto loopGlobalDeps = getGlobalDepsForLoop(mergedIterations, matchingIterations);
        // Ensure loop inserted after all deps
        if (lastDependOpIdx != iterationDeps.end()) {
            insertionPoint = std::max(insertionPoint, *lastDependOpIdx);
        }
        SmallVector<size_t> globalDeps = {loopGlobalDeps.begin(), loopGlobalDeps.end()};
        llvm::sort(globalDeps);

        // Create compute region
        const auto iterations = matchingIterations.size();
        log.nest().debug("Created sub-loop with iterations = {0}", iterations);
        auto schedulingLoop = makeSchedulingLoopFromIterations(mergedIterations, LoopType::Tiling);
        computeRegions[insertionPoint].emplace_back(std::move(schedulingLoop), std::move(globalDeps));
    }  // end current iteration loop

    return computeRegions;
}

// Create minimalistic description of tiled operations including unique/local data in/out dependencies and consumers
void createTiledOpDepsConsDescriptor(ArrayRef<size_t> tiles, AsyncDepsInfo& depsInfo,
                                     const mlir::DenseSet<size_t>& nonDmaOps,
                                     SmallVector<std::pair<size_t, SmallVector<size_t>>>& tilesWithDataDepsCons,
                                     Logger log) {
    log.trace("createTiledOpDepsConsDescriptor called for {0} tiles", tiles.size());
    auto iterations = tiles.size();
    for (size_t i = 0; i < iterations; ++i) {
        const auto opIdx = tiles[i];
        llvm::DenseSet<size_t> processed;
        SmallVector<size_t> dataInOps;
        for (auto depIdx : depsInfo.getOpDeps(opIdx)) {
            if (processed.count(depIdx)) {
                continue;
            }
            processed.insert(depIdx);

            const auto execOp = depsInfo.getExecuteOpAtIndex(depIdx);
            const auto deps = depsInfo.getOpDeps(depIdx);
            const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, nonDmaOps));

            if (allocationType != AllocationType::DATA_IN) {
                // include data in ops
                continue;
            }
            dataInOps.push_back(depIdx);
        }
        processed.insert(opIdx);
        // at this moment op itself and dependencies are included into subgraph and processed

        SmallVector<size_t> dataOutOps;
        for (auto conIdx : depsInfo.getConsumerOps(opIdx)) {
            if (processed.count(conIdx)) {
                continue;
            }
            processed.insert(conIdx);

            const auto deps = depsInfo.getOpDeps(conIdx);

            auto maxDep = std::max_element(deps.begin(), deps.end());
            // ensure data out fully unlocked at this stage
            if (maxDep != deps.end() && *maxDep > opIdx) {
                continue;
            }
            const auto execOp = depsInfo.getExecuteOpAtIndex(conIdx);
            const auto allocationType = getAllocationType(execOp, hasNonDmaDependency(deps, nonDmaOps));
            if (allocationType != AllocationType::DATA_OUT) {
                // include data out ops
                continue;
            }

            dataOutOps.push_back(conIdx);
        }

        // Now add data in/out ops and compute op idx-es to tilesWithDataDepsCons in order:
        // 1. dataInOps
        // 2. computeOp
        // 3. dataOutOps
        SmallVector<size_t> tileOps;
        tileOps.insert(tileOps.end(), dataInOps.begin(), dataInOps.end());
        size_t computeOpIdx = opIdx;
        tileOps.push_back(opIdx);
        tileOps.insert(tileOps.end(), dataOutOps.begin(), dataOutOps.end());
        tilesWithDataDepsCons.emplace_back(computeOpIdx, std::move(tileOps));
    }

    // find all shared operations
    llvm::DenseMap<size_t, size_t> loopOpCounts;
    for (const auto& iterationBody : tilesWithDataDepsCons) {
        for (const auto& op : iterationBody.second) {
            loopOpCounts[op] += 1;
        }
    }

    // re-create loops without global-shared ops
    // An op appearing in all iterations (global-shared) is factored out of the iteration body
    // so that createInnerLoopsFromIterations treats it as a global dependency. Partially-shared
    // ops (appearing in more than one but not all iterations) are kept inside their iteration
    // bodies so that createInnerLoopsFromIterations can use them for correct matching:
    // iterations sharing a partially-shared op (e.g., a weight DMA used by one C-tile's
    // H-tile group) will match on that kept op, while iterations with different partially-shared
    // ops form separate loops.
    SmallVector<std::pair<size_t, SmallVector<size_t>>> filteredTilesWithDataDepsCons;
    for (const auto& iterationBody : tilesWithDataDepsCons) {
        SmallVector<size_t> newLoopBody;
        for (const auto& op : iterationBody.second) {
            if (loopOpCounts[op] == tilesWithDataDepsCons.size()) {
                continue;
            }
            newLoopBody.push_back(op);
        }
        filteredTilesWithDataDepsCons.emplace_back(iterationBody.first, std::move(newLoopBody));
    }
    tilesWithDataDepsCons = std::move(filteredTilesWithDataDepsCons);
}

/**
    Improves determinism and keeps data-move ops aligned with the compute op’s operand order,
    which can simplify later allocation and scheduling logic.
 */
void sortDataOpsAroundComputeOp(LoopBody& allocInfos, size_t computeOpPosition) {
    // Sort data in allocations
    // 1. lookup based on compute op
    llvm::DenseMap<mlir::Value, size_t> position;
    auto recordPositions = [&](ArrayRef<mlir::Value> buffers) {
        position.reserve(position.size() + buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            // Keep the earliest occurrence to avoid overwriting when the same buffer appears multiple times.
            auto it = position.find(buffers[i]);
            if (it == position.end() || i < it->second) {
                position[buffers[i]] = i;
            }
        }
    };

    recordPositions(allocInfos[computeOpPosition].inBuffers);
    // 2. Sort based on compute op buffer order
    auto getOrderKey = [&](const OpAllocationInfo& a, bool useInputs) {
        size_t best = std::numeric_limits<size_t>::max();
        const auto& buffers = useInputs ? a.inBuffers : a.outBuffers;
        for (const auto& buf : buffers) {
            auto it = position.find(buf);
            if (it != position.end()) {
                best = std::min(best, it->second);
            }
        }
        return best;
    };
    // stable_sort keeps the original relative order when no compute buffer matches
    std::stable_sort(allocInfos.begin(), allocInfos.begin() + computeOpPosition,
                     [&](const OpAllocationInfo& a, const OpAllocationInfo& b) {
                         return getOrderKey(a, /*useInputs=*/false) < getOrderKey(b, /*useInputs=*/false);
                     });

    // Sort data out allocations
    // 1. lookup based on compute op
    position.clear();
    recordPositions(allocInfos[computeOpPosition].outBuffers);

    // 2. sort based on compute op buffer order
    if (computeOpPosition + 1 < allocInfos.size()) {
        std::stable_sort(allocInfos.begin() + computeOpPosition + 1, allocInfos.end(),
                         [&](const OpAllocationInfo& a, const OpAllocationInfo& b) {
                             return getOrderKey(a, /*useInputs=*/true) < getOrderKey(b, /*useInputs=*/true);
                         });
    }
}

}  // namespace

// Extract compute regions from async.execute ops based on tiling attributes
ComputeRegionVec vpux::getComputeRegionsFromAsyncExec(AliasesInfo& aliasInfo, AsyncDepsInfo& depsInfo, Logger log) {
    log.debug("getComputeRegionsFromAsyncExec");
    depsInfo.buildConsMap();

    // 1. Scan all async.exec ops to collect tiled regions (via loop attributes) and track non-DMA ops.
    mlir::DenseMap<size_t, SmallVector<size_t>> tiledRegions;
    mlir::DenseSet<size_t> nonDmaOps;

    log.debug("Scanning async.exec ops for tiled regions, total ops {0}", depsInfo.getExecOpCount());
    for (size_t opIdx = 0; opIdx < depsInfo.getExecOpCount(); ++opIdx) {
        auto execOp = depsInfo.getExecuteOpAtIndex(opIdx);
        auto* bodyBlock = execOp.getBody();

        for (auto& op : bodyBlock->getOperations()) {
            const auto loopAttributes = vpux::getLoopAttributes(&op);

            if (loopAttributes.tilingLoopIndex != nullptr && mlir::isa<I64Attr>(loopAttributes.tilingLoopIndex)) {
                auto tileIndex = mlir::cast<mlir::IntegerAttr>(loopAttributes.tilingLoopIndex).getInt();
                tiledRegions[tileIndex].push_back(opIdx);
                log.trace("Found tile opIdx {0} with tileIndex {1}", opIdx, tileIndex);
            }
        }

        // keep track of non-DMA ops even if they are not tiled
        const auto execKind = VPUIP::getExecutorType(execOp);
        if (execKind != config::ExecutorKind::DMA_NN) {
            nonDmaOps.insert(opIdx);
        }
    }

    ComputeRegionVec computeRegionVec;
    // If no tiled regions are found, return an empty result
    if (tiledRegions.empty()) {
        log.debug("No tiled regions found in async.exec ops");
        return computeRegionVec;
    }
    log.debug("Found {0} tiled regions", tiledRegions.size());

    std::unordered_map<size_t, ComputeRegionVec> insertionPoints;
    llvm::DenseSet<size_t> allLoopOperations;

    // Process tiled regions and create ComputeRegion per detected loop
    for (auto& [tileIndex, tiles] : tiledRegions) {
        // For each tiled region with enough iterations
        const auto iterations = tiles.size();
        if (iterations < MIN_LOOP_OPS) {
            continue;
        }

        // build per-iteration descriptors of compute + data-in/out ops (excluding shared ops)
        SmallVector<std::pair<size_t, SmallVector<size_t>>> tilesWithDataDepsCons;
        createTiledOpDepsConsDescriptor(tiles, depsInfo, nonDmaOps, tilesWithDataDepsCons, log);

        // Create OpAllocationInfo entries
        std::vector<LoopBody> loop;
        for (size_t tileNum = 0; tileNum < iterations; ++tileNum) {
            const auto& tileOps = tilesWithDataDepsCons[tileNum];

            LoopBody allocInfos;
            size_t computeOpPosition = 0;
            for (auto& idx : tileOps.second) {
                allocInfos.push_back(createAllocInfo(idx, depsInfo, aliasInfo, log));
                if (idx == tileOps.first) {
                    computeOpPosition = allocInfos.size() - 1;
                }
            }

            // Order data moves relative to compute operands
            sortDataOpsAroundComputeOp(allocInfos, computeOpPosition);

            loop.push_back(std::move(allocInfos));
        }

        // Merge per-iteration LoopBody entries into schedulable loops
        std::unordered_map<size_t, ComputeRegionVec> thisOpInsertionPoints =
                createInnerLoopsFromIterations(loop, allLoopOperations, depsInfo, nonDmaOps, log);

        for (auto& [insertionPoint, regions] : thisOpInsertionPoints) {
            auto& dest = insertionPoints[insertionPoint];
            dest.insert(dest.end(), std::make_move_iterator(regions.begin()), std::make_move_iterator(regions.end()));
        }
    }  // end process tiled regions loop

    for (size_t opIdx = 0; opIdx < depsInfo.getExecOpCount(); ++opIdx) {
        // handle non-loop operations. Insert them into computeRegionVec as trivial region with 1 iteration
        if (!allLoopOperations.count(opIdx)) {
            log.trace("Add opIdx {0} non-loop operation", opIdx);
            const auto alloc = createAllocInfo(opIdx, depsInfo, aliasInfo, log);
            auto schedulingLoop = makeSchedulingLoopFromIterations({{alloc}}, LoopType::None);
            computeRegionVec.emplace_back(std::move(schedulingLoop));
        }
        // insert loops at proper position
        if (insertionPoints.count(opIdx)) {
            auto& computeRegions = insertionPoints[opIdx];
            for (auto& computeRegion : computeRegions) {
                computeRegionVec.push_back(std::move(computeRegion));
                log.trace("Add tiled compute region: {0}", computeRegion);
            }
        }
    }

    log.debug("Total compute regions created: {0}", computeRegionVec.size());

    return computeRegionVec;
}
