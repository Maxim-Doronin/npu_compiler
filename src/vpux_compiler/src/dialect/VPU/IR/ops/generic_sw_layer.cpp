//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <iterator>

using namespace vpux;

// E#152917 Analyze & settle on GenericSwLayerOp integration & vpux interface usage

//
// SymbolUserOpInterface
//

mlir::LogicalResult VPU::GenericSwLayerOp::verifySymbolUses(mlir::SymbolTableCollection&) {
    // E#152917 : Add proper implementation for interface method
    return mlir::success();
}

//
// CallOpInterface
//

mlir::CallInterfaceCallable VPU::GenericSwLayerOp::getCallableForCallee() {
    return getOperation()->getAttrOfType<mlir::SymbolRefAttr>("callee");
}

void VPU::GenericSwLayerOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
    setCalleeAttr(mlir::cast<mlir::SymbolRefAttr>(callee));
}

mlir::Operation::operand_range VPU::GenericSwLayerOp::getArgOperands() {
    return {operand_begin(), operand_end()};
}

mlir::MutableOperandRange VPU::GenericSwLayerOp::getArgOperandsMutable() {
    return mlir::MutableOperandRange(this->getOperation());
}

// -----------------------------------------------------------------------------
// CallOpInterface attribute hooks
// -----------------------------------------------------------------------------
// CallOpInterface requires operations to provide accessors for
// per-argument and per-result call-site attributes. We do not
// model call-site metadata, so these operations do not store such attributes.
// We return nullptr for the getters and treat setters as unsupported.
//
// These methods are required only to satisfy the CallOpInterface contract for
// tooling and generic passes. Actual call-site attributes are not used or
// expected
// -----------------------------------------------------------------------------
mlir::ArrayAttr vpux::VPU::GenericSwLayerOp::getArgAttrsAttr() {
    // no call-site arg attrs supported
    return nullptr;
}

void vpux::VPU::GenericSwLayerOp::setArgAttrsAttr(mlir::ArrayAttr) {
    VPUX_THROW("Call-site argument attributes are not supported for this op");
}

mlir::Attribute vpux::VPU::GenericSwLayerOp::removeArgAttrsAttr() {
    // no call-site arg attrs supported
    return nullptr;
}

mlir::ArrayAttr vpux::VPU::GenericSwLayerOp::getResAttrsAttr() {
    // no call-site result attrs supported
    return nullptr;
}

void vpux::VPU::GenericSwLayerOp::setResAttrsAttr(mlir::ArrayAttr) {
    VPUX_THROW("Call-site result attributes are not supported for this op");
}

mlir::Attribute vpux::VPU::GenericSwLayerOp::removeResAttrsAttr() {
    // no call-site result attrs supported
    return nullptr;
}
