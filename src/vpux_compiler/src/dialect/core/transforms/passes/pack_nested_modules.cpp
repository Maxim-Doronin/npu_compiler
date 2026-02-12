//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include <vpux/compiler/dialect/core/transforms/passes.hpp>
#include <vpux/compiler/utils/func_dialect.hpp>
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::Core {
#define GEN_PASS_DECL_PACKNESTEDMODULES
#define GEN_PASS_DEF_PACKNESTEDMODULES
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

// Make this pass deterministic by
// 1. Sorting the operations in every cluster by the order in which the operations appear in the block
// 2. Sorting the clusters by the order in which their first operations appear in the blocks. Since
//    clusters don't share any operations, this is well defined.
//
// To achieve this we use std::map and std::set instead of the unordered versions and order the ops
// using isBeforeInBlock.

using FuncOrModule = mlir::Operation*;

// Note: Do not use FuncOrModuleComparator in another context as it will crash or cause UB if 2 ops appear in different
// blocks.
struct FuncOrModuleComparator {
    bool operator()(FuncOrModule a, FuncOrModule b) const {
        assert(a->getBlock() == b->getBlock());
        return a->isBeforeInBlock(b);
    }
};

using CallGraph = std::map<FuncOrModule, SmallVector<FuncOrModule>, FuncOrModuleComparator>;
using FuncOrModuleOpSet = std::set<FuncOrModule, FuncOrModuleComparator>;

/// Returns true if there exists a FuncOp in `from` that has an edge to any FuncOp in `to` according to callGraph.
/// Note that ModuleOps are ignore here.
bool hasSpanningEdges(const FuncOrModuleOpSet& from, const FuncOrModuleOpSet& to, const CallGraph& callGraph) {
    for (const auto& u : from) {
        if (const auto it = callGraph.find(u); it != callGraph.end()) {
            for (const auto& v : it->second) {
                if (to.find(v) != to.end()) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// Merges the set src into dst.
void mergeInto(FuncOrModuleOpSet& dst, const FuncOrModuleOpSet& src) {
    dst.insert(src.begin(), src.end());
}

//
// PackNestedModules
//

struct Clusters {
    FuncOrModuleOpSet topCluster;
    SmallVector<FuncOrModuleOpSet> nestedClusters;
};

class PackNestedModules : public vpux::Core::impl::PackNestedModulesBase<PackNestedModules> {
public:
    explicit PackNestedModules(Logger log, Core::NestingMode nestingMode, bool enableProfiling)
            : _nestingMode(nestingMode), _enableProfiling(enableProfiling), _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit PackNestedModules(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    // utility functions
    mlir::Operation* getFirstClusterInsertionPoint(FuncOrModuleOpSet& topCluster);
    std::string getSubModuleName(size_t index);
    bool belongsIntoTopCluster(mlir::func::FuncOp caller, mlir::func::FuncOp mainFuncOp);
    // clustering
    CallGraph collectCallGraph(mlir::func::FuncOp root);
    SmallVector<FuncOrModuleOpSet> createClusters(const CallGraph& callGraph, mlir::func::FuncOp mainFuncOp);
    SmallVector<FuncOrModuleOpSet> createMainCluster(const CallGraph& callGraph);
    Clusters collectClusters(mlir::func::FuncOp mainFuncOp);
    void patchCallSites(const Clusters& clusters);
    mlir::ModuleOp nestCluster(mlir::OpBuilder& builder, FuncOrModuleOpSet&& cluster, size_t clusterIndex);
    // updating PipelineOptions, Resources and NetworkInfo
    void clonePipelineOptionsOp(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp);
    void cloneResourcesOps(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp);
    void createNetInfoForFuncOp(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp, bool enableProfiling = false);
    void nestNetworkInfo(mlir::ModuleOp subModuleOp, net::NetworkInfoOp& netInfo);

    void safeRunOnModule() final;

private:
    Core::NestingMode _nestingMode = Core::NestingMode::Default;
    bool _enableProfiling = false;
    Logger _log;
};

mlir::LogicalResult PackNestedModules::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (mode.hasValue()) {
        _nestingMode = Core::parseNestingMode(mode.getValue());
    }

    if (enableProfiling.hasValue()) {
        _enableProfiling = enableProfiling.getValue();
    }

    return mlir::success();
}

bool PackNestedModules::belongsIntoTopCluster(mlir::func::FuncOp caller, mlir::func::FuncOp mainFuncOp) {
    return (_nestingMode == Core::NestingMode::EntryPoint ? false : caller == mainFuncOp) ||
           caller->hasAttrOfType<mlir::UnitAttr>("do_not_nest");
}

std::string PackNestedModules::getSubModuleName(size_t index) {
    return _nestingMode == Core::NestingMode::EntryPoint ? Core::NPU_MODULE_NAME.str() : formatv("Module{0}", index);
}

mlir::Operation* PackNestedModules::getFirstClusterInsertionPoint(FuncOrModuleOpSet& topCluster) {
    if (_nestingMode == Core::NestingMode::EntryPoint) {
        // We assume that the top module is not empty.
        return &getOperation().getBodyRegion().front().back();
    } else {
        assert(!topCluster.empty());
        return *topCluster.begin();
    }
}

/// Returns a map of FuncOp -> Vector<FuncOp|ModuleOp> that models the call graph using cycle-aware BFS.
CallGraph PackNestedModules::collectCallGraph(mlir::func::FuncOp root) {
    auto topModuleOp = getOperation();

    CallGraph callGraph;

    mlir::DenseSet<mlir::func::FuncOp> visited;

    std::queue<mlir::func::FuncOp> workList;
    workList.push(root);

    while (!workList.empty()) {
        auto funcOp = workList.front();
        workList.pop();

        if (bool firstOccurrence = visited.insert(funcOp).second; !firstOccurrence) {
            continue;
        }

        auto& calleeVec = callGraph[funcOp];

        funcOp.walk([&](mlir::CallOpInterface callOp) {
            const auto calleeSymRef = mlir::cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
            const auto firstSymName =
                    calleeSymRef.getRootReference();  // first symbol in a chain @module_0::...::@module_N::@calleeFunc
            const auto calleeOp = topModuleOp.lookupSymbol(firstSymName);
            assert(calleeOp != nullptr);
            calleeVec.push_back(calleeOp);
            // We can ignore ModuleOps because we cannot call functions of a top module from nested modules
            if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(calleeOp); func != nullptr) {
                workList.push(func);
            }
        });
    }

    return callGraph;
}

/// Returns a vector of sets of FuncOps.
///
///     Assume we have the following call graph:
///
///         main* -> [f, h]
///
///         f -> g*
///
///         h -> i
///
///         i -> j
///         j -> i
///
///     Functions annotated with (*) will be put into the "top" cluster (i.e. they will not be nested into submodules),
///     either because that function is the entry point or it contains the attribute "do_not_nest".
/// 1)
///     The result will be initialized as follows. All FuncOps that belong into the top cluster are put into the first
///     set of the vector first. All other functions are then put into their own respective singleton set. In this
///     example:
///         [{main*, g*}, {f}, {h}, {i}, {j}]
/// 2)
///     Then all singleton sets are merged into the top cluster set when they have an edge into the top cluster set.
///     In our example, because of (f -> g*), we would get {main*, g*, f}. To account for transitive edges, this step
///     is done until a full iteration is performed without any merges.
/// 3)
///     Then the remaining clusters (except for the top cluster) are merged pairwise if they have edges from on set to
///     the other or vice versa. This is done until no more clusters can be merged In this example:
///         [{main*, g*, f}, {h}, {i}, {j}]     {i} and {j}    are merged because of i -> j and j -> i
///         [{main*, g*, f}, {h}, {i, j}]       {h} and {i, j} are merged because of h -> i
///         [{main*, g*, f}, {h, i, j}]         Nothing to merge
/// 4)
///     Done
///
SmallVector<FuncOrModuleOpSet> PackNestedModules::createClusters(const CallGraph& callGraph,
                                                                 mlir::func::FuncOp mainFuncOp) {
    SmallVector<FuncOrModuleOpSet> clusters{{}};

    for (auto [caller, callees] : callGraph) {
        const auto callerFuncOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(caller);
        if (callerFuncOp != nullptr && belongsIntoTopCluster(callerFuncOp, mainFuncOp)) {
            clusters[0].insert(caller);
        } else {
            clusters.push_back({});
            clusters.back().insert(caller);
        }
    }

    // merge singletons into top cluster
    auto mergeSingletonsIntoTopCluster = [&]() -> bool {
        for (auto src = clusters.begin() + 1; src < clusters.end(); src++) {
            auto dst = clusters.begin();
            const auto doMerge = hasSpanningEdges(*src, *dst, callGraph);
            if (doMerge) {
                mergeInto(*dst, *src);
                clusters.erase(src);
                return true;
            }
        }
        return false;
    };
    while (mergeSingletonsIntoTopCluster()) {
    }

    // correctly merge pairwise until fixed point is reached (ignore top cluster)
    auto mergePairwise = [&]() -> bool {
        for (auto dst = std::next(clusters.begin()); dst < clusters.end(); dst++) {
            for (auto src = std::next(dst); src < clusters.end(); src++) {
                const auto doMerge = hasSpanningEdges(*dst, *src, callGraph) || hasSpanningEdges(*src, *dst, callGraph);
                if (doMerge) {
                    mergeInto(*dst, *src);
                    clusters.erase(src);
                    return true;
                }
            }
        }
        return false;
    };
    while (mergePairwise()) {
    }

    return clusters;
}

/// Creates one main cluster for the entryPoint function
SmallVector<FuncOrModuleOpSet> PackNestedModules::createMainCluster(const CallGraph& callGraph) {
    FuncOrModuleOpSet mainCluster;

    for (auto& [caller, callees] : callGraph) {
        mainCluster.insert(caller);
        mainCluster.insert(callees.begin(), callees.end());
    }

    auto topModuleOp = getOperation();
    FuncOrModuleOpSet topCluster;
    // topCluster contains all func ops of topModule that doesn't belong to callGraph.
    // We can ignore ModuleOps because we don't need to update their callOps
    for (const auto& funcOp : topModuleOp.getOps<mlir::func::FuncOp>()) {
        auto funcOrModule = funcOp;
        if (callGraph.count(funcOrModule) == 0) {
            topCluster.insert(funcOrModule);
        }
    }

    return {std::move(topCluster), std::move(mainCluster)};
}

/// Returns a vector of vector of FuncOps. Every vector of FuncOps represents one set of operations that will
/// be nested inside a submodule. The resulting vectors will be sorted to ensure deterministic IR.
Clusters PackNestedModules::collectClusters(mlir::func::FuncOp mainFuncOp) {
    const auto callGraph = collectCallGraph(mainFuncOp);
    const auto clustersSet = _nestingMode == Core::NestingMode::EntryPoint ? createMainCluster(callGraph)
                                                                           : createClusters(callGraph, mainFuncOp);

    return {clustersSet.front(), SmallVector<FuncOrModuleOpSet>(std::next(clustersSet.begin()), clustersSet.end())};
}

/// Replaces all mlir::CallOps in the top cluster with Core::NestedCallOps that point to the newly nested functions.
void PackNestedModules::patchCallSites(const Clusters& clusters) {
    auto topModuleOp = getOperation();

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);

    std::map<FuncOrModule, size_t, FuncOrModuleComparator> nestedClusterMap;
    for (const auto& [index, nestedCluster] : clusters.nestedClusters | indexed) {
        for (auto funcOp : nestedCluster) {
            nestedClusterMap[funcOp] = index;
        }
    }

    for (auto funcOrModuleOp : clusters.topCluster) {
        funcOrModuleOp->walk([&](mlir::CallOpInterface callOp) {
            const auto calleeOpSymRef = mlir::dyn_cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
            const auto firstSymName = calleeOpSymRef.getRootReference();
            auto callee = topModuleOp.lookupSymbol(firstSymName);
            assert(callee != nullptr);
            auto it = nestedClusterMap.find(callee);
            // callOp does not point to a future nested function
            if (it == nestedClusterMap.end()) {
                return;
            }

            const auto clusterIndex = it->second;
            const auto moduleName = getSubModuleName(clusterIndex);

            // Construct a vector of FlatSymbolRefAttr from callee op reference symbols and then
            // build a new SymbolRefAttr as @Module + {symVec}
            SmallVector<mlir::FlatSymbolRefAttr> symVec{mlir::FlatSymbolRefAttr::get(firstSymName)};
            symVec.append(calleeOpSymRef.getNestedReferences().begin(), calleeOpSymRef.getNestedReferences().end());
            const auto nestedSymbolAttr = mlir::SymbolRefAttr::get(builder.getStringAttr(moduleName), symVec);

            builder.setInsertionPoint(callOp);
            const auto nestedCallOp = builder.create<Core::NestedCallOp>(
                    callOp.getLoc(), nestedSymbolAttr, callOp->getResultTypes(), callOp->getOperands());

            callOp->replaceAllUsesWith(nestedCallOp);
            callOp->erase();
        });
    }
}

/// Creates a new ModuleOp at the builder's current location and moves all FuncOps in the cluster into it.
mlir::ModuleOp PackNestedModules::nestCluster(mlir::OpBuilder& builder, FuncOrModuleOpSet&& cluster,
                                              size_t clusterIndex) {
    const auto moduleName = getSubModuleName(clusterIndex);

    auto topModuleOp = getOperation();
    auto subModuleOp = builder.create<mlir::ModuleOp>(appendLoc(topModuleOp.getLoc(), "{0}", moduleName), moduleName);
    auto& targetOps = subModuleOp.getBody()->getOperations();

    for (auto funcOrModuleOp : cluster) {
        targetOps.splice(targetOps.end(), funcOrModuleOp->getBlock()->getOperations(), funcOrModuleOp->getIterator());
    }

    return subModuleOp;
}

void PackNestedModules::createNetInfoForFuncOp(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp,
                                               bool enableProfiling) {
    auto netInfo = builder.create<net::NetworkInfoOp>(
            appendLoc(funcOp.getLoc(), "nested_network_info"),
            mlir::FlatSymbolRefAttr::get(funcOp->getContext(), funcOp.getName()), enableProfiling);
    net::setupSections(netInfo, enableProfiling);

    auto funcType = funcOp.getFunctionType();

    // Handle inputs
    auto& inputRegion = netInfo.getInputsInfo();
    builder.setInsertionPointToStart(&inputRegion.front());

    // These will be replaced with core dialect definitions when dynamic strides are removed
    llvm::StringRef funcArgDynamicStridesAttrName = HOST_EXEC_FUNC_ARG_DYNAMIC_STRIDES_ATTR_NAME;
    mlir::StringAttr funcArgDynamicStridesAttrNameAttr = builder.getStringAttr(funcArgDynamicStridesAttrName);
    llvm::StringRef dynamicStridesAttrName = HOST_EXEC_DYNAMIC_STRIDES_ATTR_NAME;

    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
        auto argType = mlir::cast<vpux::NDTypeInterface>(funcType.getInput(i));
        const auto newType = mlir::RankedTensorType::get(argType.getShape(), argType.getElementType(), nullptr);
        auto name = formatv("in_{0}", i).str();
        auto dataInfoOp = builder.create<net::DataInfoOp>(appendLoc(funcOp.getLoc(), name), name, newType);

        // If func op has dynamicStrides attribute for arguments then set the same attribute to inputsInfo
        auto dynamicStridesAttr =
                mlir::dyn_cast_or_null<mlir::BoolAttr>(funcOp.getArgAttr(i, funcArgDynamicStridesAttrName));
        if (dynamicStridesAttr && dynamicStridesAttr.getValue()) {
            dataInfoOp->setAttr(dynamicStridesAttrName, mlir::UnitAttr::get(dataInfoOp.getContext()));
            funcOp.removeArgAttr(i, funcArgDynamicStridesAttrNameAttr);
        }
    }

    // Handle outputs
    auto& outputsRegion = netInfo.getOutputsInfo();
    builder.setInsertionPointToStart(&outputsRegion.front());

    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
        auto resType = mlir::cast<vpux::NDTypeInterface>(funcType.getResult(i));
        const auto newType = mlir::RankedTensorType::get(resType.getShape(), resType.getElementType(), nullptr);
        auto name = formatv("out_{0}", i).str();
        auto dataInfoOp = builder.create<net::DataInfoOp>(appendLoc(funcOp.getLoc(), name), name, newType);

        // If func op has dynamicStrides attribute for results then set the same attribute to outputsInfo
        auto dynamicStridesAttr =
                mlir::dyn_cast_or_null<mlir::BoolAttr>(funcOp.getResultAttr(i, funcArgDynamicStridesAttrName));
        if (dynamicStridesAttr && dynamicStridesAttr.getValue()) {
            dataInfoOp->setAttr(dynamicStridesAttrName, mlir::UnitAttr::get(dataInfoOp.getContext()));
            funcOp.removeResultAttr(i, funcArgDynamicStridesAttrNameAttr);
        }
    }
}

void PackNestedModules::clonePipelineOptionsOp(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp) {
    auto topPipelineOptions = topModuleOp.getOps<config::PipelineOptionsOp>();
    auto optionsCount = std::distance(topPipelineOptions.begin(), topPipelineOptions.end());
    VPUX_THROW_WHEN(optionsCount != 1, "No valid count of config.PipelineOptionsOp found {0}", optionsCount);

    auto topPipelineOptionsOp = *topPipelineOptions.begin();
    builder.clone(*topPipelineOptionsOp);
}

void PackNestedModules::cloneResourcesOps(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp) {
    for (auto reservedResource : topModuleOp.getOps<config::ResourcesOp>()) {
        builder.clone(*reservedResource);
    }
}

void PackNestedModules::nestNetworkInfo(mlir::ModuleOp subModuleOp, net::NetworkInfoOp& netInfo) {
    auto& targetOps = subModuleOp.getBody()->getOperations();
    targetOps.splice(targetOps.begin(), netInfo->getBlock()->getOperations(), netInfo->getIterator());
}

void PackNestedModules::safeRunOnModule() {
    auto topModuleOp = getOperation();
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(topModuleOp, netInfo, mainFuncOp);

    auto clusters = collectClusters(mainFuncOp);
    VPUX_THROW_WHEN(_nestingMode == Core::NestingMode::EntryPoint && clusters.nestedClusters.size() != 1,
                    "Cannot nest entryPoint function: no valid count of nested clusters found {0}",
                    clusters.nestedClusters.size());
    patchCallSites(clusters);

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    builder.setInsertionPoint(getFirstClusterInsertionPoint(clusters.topCluster));
    for (auto&& [index, nestedCluster] : clusters.nestedClusters | indexed) {
        // move because the FuncOps in cluster will be invalidated
        auto subModuleOp = nestCluster(builder, std::move(nestedCluster), index);

        // subModule ops should have the same attributes as the top module
        // that have not already been set
        for (auto attr : topModuleOp->getAttrs()) {
            if (!subModuleOp->hasAttr(attr.getName())) {
                subModuleOp->setAttr(attr.getName(), attr.getValue());
            }
        }

        builder.setInsertionPointToStart(subModuleOp.getBody());

        if (_nestingMode == Core::NestingMode::EntryPoint) {
            // If we nest entryPoint func then networkInfo should be moved to the subModule as well
            nestNetworkInfo(subModuleOp, netInfo);
        } else {
            // In case of the default nesting mode we use the func of each subModule op as its new entryPoint
            // and create new NetworkInfo operations
            auto funcOps = subModuleOp.getOps<mlir::func::FuncOp>();
            auto funcOpsCount = std::distance(funcOps.begin(), funcOps.end());
            if (funcOpsCount != 1) {
                continue;
            }
            createNetInfoForFuncOp(builder, *funcOps.begin(), _enableProfiling);
        }

        builder.setInsertionPointToStart(subModuleOp.getBody());

        clonePipelineOptionsOp(builder, topModuleOp);
        cloneResourcesOps(builder, topModuleOp);

        // nestCluster invalidates IR so we have to update the insertion point
        builder.setInsertionPointAfter(subModuleOp);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createPackNestedModulesPass(Logger log, Core::NestingMode nestingMode,
                                                                    bool enableProfiling) {
    return std::make_unique<PackNestedModules>(log, nestingMode, enableProfiling);
}
