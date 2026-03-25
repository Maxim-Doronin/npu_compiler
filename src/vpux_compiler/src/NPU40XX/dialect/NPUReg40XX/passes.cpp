//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/passes.hpp"

namespace vpux::NPUReg40XX {

namespace {
#define GEN_PASS_REGISTRATION
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/passes.hpp.inc"
}  // namespace

void registerPasses() {
    registerNPUReg40XXPasses();
}

}  // namespace vpux::NPUReg40XX
