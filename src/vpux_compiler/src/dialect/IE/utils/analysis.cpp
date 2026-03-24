//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/analysis.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"

using namespace vpux;

mlir::FailureOr<mlir::Operation*> IE::searchOpConsumers(mlir::Operation* op,
                                                        const std::function<bool(mlir::Operation*)>& isTargetOpFound) {
    if (op == nullptr) {
        return mlir::failure();
    }

    for (auto user : op->getUsers()) {
        mlir::Operation* operation = user;
        while (operation) {
            if (isTargetOpFound(operation)) {
                return operation;
            } else if (IE::isPureViewOp(operation) && operation->hasOneUse()) {
                operation = *(operation->getUsers().begin());
                continue;
            } else {
                break;
            }
        }
    }
    return mlir::failure();
}
