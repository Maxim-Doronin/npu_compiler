//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include <vpux/compiler/dialect/core/IR/unified_func_inliner_interface.hpp>
#include <vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp>
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace {

using namespace vpux::Core;

// Fallback implementations are taken from Dialect/Func/Extensions/InlinerExtension.cpp.

bool isLegalToInlineFallback(mlir::Operation*, mlir::Operation*, bool) {
    return true;
}

bool isLegalToInlineFallback(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) {
    return true;
}

bool isLegalToInlineFallback(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) {
    return true;
}

void handleTerminatorFallback(mlir::Operation* op, mlir::ValueRange valuesToReplace) {
    // Only return needs to be handled here.
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (const auto& it : llvm::enumerate(returnOp.getOperands())) {
        valuesToReplace[it.index()].replaceAllUsesWith(it.value());
    }
}

void processInlinedCallBlocksFallback(mlir::Operation*, mlir::iterator_range<mlir::Region::iterator>) {
}

std::tuple<mlir::Block*, mlir::Block::iterator> getInlineBlockAndPointFallback(mlir::Operation* call) {
    return std::make_tuple(call->getBlock(), std::next(call->getIterator()));
}

void eraseCallFallback(mlir::Operation* call) {
    call->erase();
}

}  // namespace

namespace vpux::Core {

bool UnifiedFuncInlinerInterface::isLegalToInline(mlir::Operation* call, mlir::Operation* callable,
                                                  bool wouldBeCloned) const {
    const auto interface = getDispatchInterface(call);
    if (interface == nullptr) {
        return isLegalToInlineFallback(call, callable, wouldBeCloned);
    }

    return interface->isLegalToInline(call, callable, wouldBeCloned);
}

bool UnifiedFuncInlinerInterface::isLegalToInline(mlir::Region* dest, mlir::Region* src, bool wouldBeCloned,
                                                  mlir::IRMapping& valueMapping) const {
    const auto interface = getDispatchInterface(dest->getParentOp());
    if (interface == nullptr) {
        return isLegalToInlineFallback(dest, src, wouldBeCloned, valueMapping);
    }

    return interface->isLegalToInline(dest, src, wouldBeCloned, valueMapping);
}

bool UnifiedFuncInlinerInterface::isLegalToInline(mlir::Operation* op, mlir::Region* dest, bool wouldBeCloned,
                                                  mlir::IRMapping& valueMapping) const {
    const auto interface = getDispatchInterface(op);
    if (interface == nullptr) {
        return isLegalToInlineFallback(op, dest, wouldBeCloned, valueMapping);
    }

    return interface->isLegalToInline(op, dest, wouldBeCloned, valueMapping);
}

void UnifiedFuncInlinerInterface::handleTerminator(mlir::Operation* op, mlir::ValueRange valuesToReplace) const {
    const auto interface = getDispatchInterface(op);
    if (interface == nullptr) {
        handleTerminatorFallback(op, valuesToReplace);
        return;
    }

    interface->handleTerminator(op, valuesToReplace);
}

void UnifiedFuncInlinerInterface::processInlinedCallBlocks(
        mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    const auto interface = getDispatchInterface(call);
    if (interface == nullptr) {
        processInlinedCallBlocksFallback(call, inlinedBlocks);
        return;
    }

    interface->processInlinedCallBlocks(call, inlinedBlocks);
}

std::tuple<mlir::Block*, mlir::Block::iterator> UnifiedFuncInlinerInterface::getInlineBlockAndPoint(
        mlir::Operation* call) const {
    const auto interface = getDispatchInterface(call);
    if (interface == nullptr) {
        return getInlineBlockAndPointFallback(call);
    }

    return interface->getInlineBlockAndPoint(call);
}

void UnifiedFuncInlinerInterface::eraseCall(mlir::Operation* call) const {
    const auto interface = getDispatchInterface(call);
    if (interface == nullptr) {
        eraseCallFallback(call);
        return;
    }

    interface->eraseCall(call);
}

mlir::DialectInlinerInterface* UnifiedFuncInlinerInterface::getDispatchInterface(mlir::Operation* op) const {
    if (op == nullptr) {
        return nullptr;
    }

    auto dispatchAttr =
            op->getAttrOfType<InlinerDispatchAttrInterface>(InlinerDispatchAttrInterface::getInlinerDispatchAttrName());
    if (dispatchAttr == nullptr) {
        return nullptr;
    }

    const auto it = _dispatchTable.find(dispatchAttr.getTypeID());
    return (it == _dispatchTable.end()) ? nullptr : it->getSecond().get();
}

}  // namespace vpux::Core
