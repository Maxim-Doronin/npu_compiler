//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

size_t VPUASM::InputBindingsOp::getNetInputsCount() {
    return getInputDeclarations().front().getOperations().size();
}

SmallVector<VPUASM::DeclareBufferOp, 1> VPUASM::InputBindingsOp::getInputDeclarationsOps() {
    return to_vector<1>(getInputDeclarations().getOps<VPUASM::DeclareBufferOp>());
}

VPUASM::InputBindingsOp VPUASM::InputBindingsOp::getFromModule(mlir::ModuleOp moduleOp) {
    auto bindingOps = to_small_vector(moduleOp.getOps<VPUASM::InputBindingsOp>());

    VPUX_THROW_UNLESS(bindingOps.size() <= 1,
                      "Can't have more than one 'VPUASM::InputBindingsOp' Operation in Module, got `{0}`",
                      bindingOps.size());

    VPUASM::InputBindingsOp bindingOp = bindingOps.size() == 1 ? bindingOps.front() : nullptr;
    return bindingOp;
}

void VPUASM::InputBindingsOp::build(::mlir::OpBuilder&, ::mlir::OperationState& odsState) {
    odsState.addRegion()->emplaceBlock();
}

size_t VPUASM::OutputBindingsOp::getNetOutputsCount() {
    return getOutputDeclarations().front().getOperations().size();
}

SmallVector<VPUASM::DeclareBufferOp, 1> VPUASM::OutputBindingsOp::getOutputDeclarationsOps() {
    return to_vector<1>(getOutputDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
}

VPUASM::OutputBindingsOp VPUASM::OutputBindingsOp::getFromModule(mlir::ModuleOp moduleOp) {
    auto bindingOps = to_small_vector(moduleOp.getOps<VPUASM::OutputBindingsOp>());

    VPUX_THROW_UNLESS(bindingOps.size() <= 1,
                      "Can't have more than one 'VPUASM::OutputBindingsOp' Operation in Module, got `{0}`",
                      bindingOps.size());

    VPUASM::OutputBindingsOp bindingOp = bindingOps.size() == 1 ? bindingOps.front() : nullptr;
    return bindingOp;
}

void VPUASM::OutputBindingsOp::build(::mlir::OpBuilder&, ::mlir::OperationState& odsState) {
    odsState.addRegion()->emplaceBlock();
}

size_t VPUASM::ProfilingBindingsOp::getNetProfilingCount() {
    return getProfilingDeclarations().front().getOperations().size();
}

SmallVector<VPUASM::DeclareBufferOp, 1> VPUASM::ProfilingBindingsOp::getProfilingDeclarationsOps() {
    return to_vector<1>(getProfilingDeclarations().front().getOps<VPUASM::DeclareBufferOp>());
}

VPUASM::ProfilingBindingsOp VPUASM::ProfilingBindingsOp::getFromModule(mlir::ModuleOp moduleOp) {
    auto bindingOps = to_small_vector(moduleOp.getOps<VPUASM::ProfilingBindingsOp>());

    VPUX_THROW_UNLESS(bindingOps.size() <= 1,
                      "Can't have more than one 'VPUASM::ProfilingBindingsOp' Operation in Module, got `{0}`",
                      bindingOps.size());

    VPUASM::ProfilingBindingsOp bindingOp = bindingOps.size() == 1 ? bindingOps.front() : nullptr;
    return bindingOp;
}

void VPUASM::ProfilingBindingsOp::build(::mlir::OpBuilder&, ::mlir::OperationState& odsState) {
    odsState.addRegion()->emplaceBlock();
}
