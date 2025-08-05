//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux::HostExec {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerHostExecPasses();
}

}  // namespace vpux::HostExec
