//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPUMI37XX {

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

}  // namespace VPUMI37XX
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI37XX/ops_interfaces.hpp.inc>
