//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinOps.h>

using namespace vpux;

static mlir::LogicalResult checkFunctionPrototype(net::NetworkInfoOp cnnOp, mlir::func::FuncOp netFunc,
                                                  SmallVector<net::DataInfoOp>& inputsInfo,
                                                  SmallVector<net::DataInfoOp>& outputsInfo,
                                                  SmallVector<net::DataInfoOp>& profilingOutputsInfo) {
    const auto netFuncType = netFunc.getFunctionType();
    const auto args = netFunc.getArgumentTypes();

    const auto hoistedIOs = (netFuncType.getNumInputs() == 0) && (netFuncType.getNumResults() == 0);

    if (!hoistedIOs && netFuncType.getNumResults() != outputsInfo.size() + profilingOutputsInfo.size()) {
        return errorAt(cnnOp, "entryPoint '@{0}' outputs count '{1}' doesn't match userOutputs count '{2}'",
                       cnnOp.getEntryPoint(), netFuncType.getNumResults(), outputsInfo.size());
    }

    const auto isArgsTensorized = std::all_of(args.begin(), args.end(), [](mlir::Type type) {
        return mlir::isa<mlir::RankedTensorType>(type);
    });
    const auto isTensorized = (netFuncType.getNumInputs() == inputsInfo.size()) && isArgsTensorized;
    if (isTensorized) {
        return mlir::success();
    }

    const auto isArgsBufferized = std::all_of(args.begin(), args.end(), [](mlir::Type type) {
        return mlir::isa<mlir::BaseMemRefType>(type);
    });
    const auto isSemiBufferized = (netFuncType.getNumInputs() == inputsInfo.size()) && isArgsBufferized;
    if (isSemiBufferized) {
        return mlir::success();
    }

    const auto isBufferized =
            (netFuncType.getNumInputs() == inputsInfo.size() + outputsInfo.size() + profilingOutputsInfo.size()) &&
            isArgsBufferized;

    if (isBufferized && !hoistedIOs) {
        mlir::LogicalResult res = mlir::success();
        netFunc.walk([&inputsInfo, &netFunc, &res](mlir::func::ReturnOp op) {
            const auto operands = op.getOperands();
            for (const auto ind : irange(operands.size())) {
                const auto rawInd = checked_cast<unsigned>(inputsInfo.size() + ind);

                const auto output = operands[ind];
                const auto outputBuffer = netFunc.getArgument(rawInd);

                const ValueSourceInfo info(output);
                if (info.getRoot(output) != outputBuffer) {
                    op.emitError() << "function output at index=" << ind
                                   << " should be an alias of the output buffer, but it's not";
                    res = mlir::failure();
                    break;
                }
            }
        });

        return res.failed() ? res : mlir::success();
    }

    // E#69730: find cleaner representation of the FuncOp with no args
    if (hoistedIOs) {
        return mlir::success();
    }

    return errorAt(cnnOp,
                   "entryPoint '@{0}' has invalid state. inputs count '{1}', results count '{2}', user inputs "
                   "count '{3}', user outputs count '{4}'",
                   cnnOp.getEntryPoint(), netFuncType.getNumInputs(), netFuncType.getNumResults(), inputsInfo.size(),
                   outputsInfo.size());
}

void net::NetworkInfoOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                               mlir::FlatSymbolRefAttr entryPoint, bool withProfiling) {
    build(builder, state, entryPoint, nullptr, static_cast<unsigned>(withProfiling ? 1 : 0));
}

mlir::LogicalResult net::NetworkInfoOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    auto netFunc = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, getEntryPointAttr());

    if (netFunc == nullptr) {
        return errorAt(*this, "entryPoint '@{0}' doesn't refer to existing Function", getEntryPoint());
    }

    auto& cnnOp = *this;
    auto inputsInfo = to_small_vector(this->getInputsInfo().getOps<net::DataInfoOp>());
    auto outputsInfo = to_small_vector(this->getOutputsInfo().getOps<net::DataInfoOp>());
    SmallVector<net::DataInfoOp> profilingOutputsInfo;
    if (!this->getProfilingOutputsInfo().empty()) {
        profilingOutputsInfo = to_small_vector(this->getProfilingOutputsInfo().front().getOps<net::DataInfoOp>());
    }

    if (checkFunctionPrototype(cnnOp, netFunc, inputsInfo, outputsInfo, profilingOutputsInfo).failed()) {
        return mlir::failure();
    }

    const auto compareShape = [&cnnOp](NDTypeInterface funcType, NDTypeInterface userType, size_t ind) {
        if (funcType == nullptr) {
            return errorAt(cnnOp, "entryPoint '@{0}' input #{1} is not a 'NDTypeInterface'", cnnOp.getEntryPoint(),
                           ind);
        }

        if (userType == nullptr) {
            return errorAt(cnnOp, "User input #{0} is not a 'NDTypeInterface'", ind);
        }

        auto isDynamic = userType.getShape().isDynamic() || funcType.getShape().isDynamic();
        if (!isDynamic && funcType.getNumElements() != userType.getNumElements()) {
            return errorAt(cnnOp, "entryPoint '@{0}' input #{1} is not compatible with user type '{2}'",
                           cnnOp.getEntryPoint(), ind, userType);
        }

        return mlir::success();
    };

    const auto netFuncType = netFunc.getFunctionType();

    // E#69730: find cleaner representation of the FuncOp with no args
    const auto hoistedIOs = (netFuncType.getNumInputs() == 0) && (netFuncType.getNumResults() == 0);
    if (hoistedIOs) {
        return mlir::success();
    }

    for (const auto ind : irange(inputsInfo.size())) {
        const auto funcType = mlir::dyn_cast<NDTypeInterface>(netFuncType.getInput(static_cast<uint32_t>(ind)));
        const auto userType = mlir::dyn_cast<NDTypeInterface>(inputsInfo[ind].getUserType());

        if (compareShape(funcType, userType, ind).failed()) {
            return mlir::failure();
        }
    }

    for (const auto ind : irange(outputsInfo.size())) {
        const auto funcType = mlir::dyn_cast<NDTypeInterface>(netFuncType.getResult(static_cast<uint32_t>(ind)));
        const auto userType = mlir::dyn_cast<NDTypeInterface>(outputsInfo[ind].getUserType());

        if (compareShape(funcType, userType, ind).failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}

namespace {

mlir::LogicalResult verifyDataInfoRegion(mlir::Operation* op, mlir::Region& region, StringRef regionName) {
    if (region.getBlocks().size() != 1) {
        return errorAt(op, "'{0}' Region must contain exact 1 Block", regionName);
    }

    auto& allOps = region.front().getOperations();

    for (auto& infoOp : allOps) {
        if (!mlir::isa<net::DataInfoOp>(infoOp)) {
            return errorAt(op, "'{0}' Region must contain only DataInfo operations, got '{1}'", regionName,
                           infoOp.getName());
        }
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult net::NetworkInfoOp::verify() {
    if (getEntryPointAttr() == nullptr) {
        return errorAt(*this, "entryPoint attribute is NULL");
    }

    if (mlir::failed(verifyDataInfoRegion(*this, getInputsInfo(), "inputInfo"))) {
        return mlir::failure();
    }
    if (mlir::failed(verifyDataInfoRegion(*this, getOutputsInfo(), "outputsInfo"))) {
        return mlir::failure();
    }

    auto outputsInfo = getOutputsDataInfo();

    if (outputsInfo.empty()) {
        return errorAt(*this, "Operation has no user outputs information");
    }

    return mlir::success();
}

size_t net::NetworkInfoOp::getNetInputsCount() {
    return getInputsInfo().front().getOperations().size();
}

SmallVector<net::DataInfoOp, 1> net::NetworkInfoOp::getInputsDataInfo() {
    return to_vector<1>(getInputsInfo().getOps<net::DataInfoOp>());
}

size_t net::NetworkInfoOp::getNetOutputsCount() {
    return getOutputsInfo().front().getOperations().size();
}

SmallVector<net::DataInfoOp, 1> net::NetworkInfoOp::getOutputsDataInfo() {
    return to_vector<1>(getOutputsInfo().getOps<net::DataInfoOp>());
}

size_t net::NetworkInfoOp::getProfilingOutputsCount() {
    if (!getProfilingOutputsInfo().empty() && !getProfilingOutputsInfo().front().empty()) {
        return getProfilingOutputsInfo().front().front().getOperations().size();
    }
    return 0;
}

SmallVector<net::DataInfoOp, 1> net::NetworkInfoOp::getProfilingOutputsDataInfo() {
    if (!getProfilingOutputsInfo().empty()) {
        return to_vector<1>(getProfilingOutputsInfo().front().getOps<net::DataInfoOp>());
    }
    return SmallVector<net::DataInfoOp, 1>();
}

void net::NetworkInfoOp::getFromModule(mlir::ModuleOp module, net::NetworkInfoOp& netInfo,
                                       mlir::func::FuncOp& netFunc) {
    auto netOps = to_small_vector(module.getOps<net::NetworkInfoOp>());

    VPUX_THROW_UNLESS(netOps.size() == 1,
                      "Can't have more than one 'net::NetworkInfoOp' Operation in Module, got '{0}'", netOps.size());

    netInfo = netOps.front();
    netFunc = module.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPointAttr());

    VPUX_THROW_UNLESS(netFunc != nullptr, "Can't find entryPoint '@{0}' for '{1}' Operation", netInfo.getEntryPoint(),
                      netInfo->getName());
}
