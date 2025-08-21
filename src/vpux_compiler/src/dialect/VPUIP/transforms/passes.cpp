//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux::VPUIP {

ConditionFunc makeStubCondition() {
    return &vpux::VPUIP::isOp<IE::MemPermuteOp>;
}

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerVPUIPPasses();
}

}  // namespace vpux::VPUIP
