//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/act_shave_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

namespace vpux {
namespace IE {

bool isActShaveKernel(mlir::Operation* operation) {
    return VPU::NCEInvariant::isSupported(operation, Logger::global()).failed() &&
           !mlir::isa<IE::ViewLikeOpInterface>(operation);
}

}  // namespace IE
}  // namespace vpux
