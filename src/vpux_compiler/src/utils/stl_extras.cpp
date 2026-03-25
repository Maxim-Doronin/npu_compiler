//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/stl_extras.hpp"

#include <cassert>

using namespace vpux;

bool vpux::OpOrderCmp::operator()(mlir::Operation* lhs, mlir::Operation* rhs) const {
    assert(lhs->getBlock() == rhs->getBlock());

    return lhs->isBeforeInBlock(rhs);
}

bool vpux::ValueOrderCmp::compare(mlir::Value lhs, mlir::Value rhs) {
    assert(lhs.getParentBlock() == rhs.getParentBlock());

    if (mlir::isa<mlir::OpResult>(lhs) && mlir::isa<mlir::OpResult>(rhs)) {
        if (lhs.getDefiningOp() == rhs.getDefiningOp()) {
            return mlir::cast<mlir::OpResult>(lhs).getResultNumber() <
                   mlir::cast<mlir::OpResult>(rhs).getResultNumber();
        } else {
            return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
        }
    } else if (mlir::isa<mlir::BlockArgument>(lhs) && mlir::isa<mlir::OpResult>(rhs)) {
        return true;
    } else if (mlir::isa<mlir::OpResult>(lhs) && mlir::isa<mlir::BlockArgument>(rhs)) {
        return false;
    } else {
        return mlir::cast<mlir::BlockArgument>(lhs).getArgNumber() <
               mlir::cast<mlir::BlockArgument>(rhs).getArgNumber();
    }
}

bool vpux::ValueOrderCmp::operator()(mlir::Value lhs, mlir::Value rhs) const {
    return compare(lhs, rhs);
}
