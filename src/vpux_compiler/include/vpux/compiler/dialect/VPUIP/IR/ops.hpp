//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace VPUIP {

constexpr Bit FP16_SIZE = 16_Bit;
constexpr KB SHAVE_LIB_DATA_SIZE = 112_KB;

}  // namespace VPUIP
}  // namespace vpux

//
// Template methods
//

namespace vpux {
namespace VPUIP {

template <typename... Args>
VPUIP::PPETaskOp NCEClusterTaskOp::addPPETask(mlir::OpBuilder& builder, Args&&... args) {
    if (getPpe().empty()) {
        getPpe().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getPpe().front());

    return builder.create<VPUIP::PPETaskOp>(getLoc(), std::forward<Args>(args)...);
}

template <typename T>
T vpux::VPUIP::NCEClusterTilingOp::getInnerTaskOpOfType() {
    return mlir::dyn_cast<T>(&getBody().front().front());
}
}  // namespace VPUIP
}  // namespace vpux
