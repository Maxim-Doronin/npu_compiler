//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"

namespace vpux::bytecode {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerbytecodePasses();
}

}  // namespace vpux::bytecode
