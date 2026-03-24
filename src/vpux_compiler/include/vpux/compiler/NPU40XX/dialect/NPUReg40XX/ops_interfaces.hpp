//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace NPUReg40XX {

//
// SingleOutputAsIndexOp
//

mlir::LogicalResult verifySingleOutputAsIndexOp(mlir::Operation* op);

template <typename ConcreteOp>
class SingleOutputAsIndexOp : public mlir::OpTrait::TraitBase<ConcreteOp, SingleOutputAsIndexOp> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySingleOutputAsIndexOp(op);
    }
};

}  // namespace NPUReg40XX
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops_interfaces.hpp.inc>
