//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/utils/declaration_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

void vpux::moveDeclarationsToTop(mlir::func::FuncOp& netFunc) {
    auto& block = netFunc.getBody().front();

    SmallVector<mlir::Operation*> allDeclOps;
    for (auto& op : block) {
        if (op.hasTrait<DeclarationOp>() || mlir::isa<mlir::memref::AllocOp>(&op)) {
            allDeclOps.push_back(&op);
        }
    }

    if (allDeclOps.empty()) {
        return;
    }

    auto* firstDeclOp = allDeclOps.front();
    firstDeclOp->moveBefore(&block, block.begin());

    for (auto i : irange(allDeclOps.size() - 1)) {
        allDeclOps[i + 1]->moveAfter(allDeclOps[i]);
    }
}
