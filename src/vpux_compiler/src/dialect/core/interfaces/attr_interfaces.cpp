//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"

#include <mlir/Interfaces/CallInterfaces.h>

void vpux::Core::setInlinerDispatchAttr(mlir::ModuleOp moduleOp, InlinerDispatchAttrInterface attr) {
    // WalkOrder::PreOrder ensures that the tree is walked from top to bottom and from root to leaves.
    moduleOp->walk<mlir::WalkOrder::PreOrder>([moduleOp, attr](mlir::Operation* op) {
        if (op != moduleOp && mlir::isa<mlir::ModuleOp>(op)) {
            return mlir::WalkResult::skip();
        }

        const auto isOpEligible = mlir::isa<mlir::CallOpInterface, mlir::CallableOpInterface>(op);
        if (isOpEligible) {
            op->setAttr(Core::InlinerDispatchAttrInterface::getInlinerDispatchAttrName(), attr);
        }

        return mlir::WalkResult::advance();
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/core/attr_interfaces.cpp.inc>
