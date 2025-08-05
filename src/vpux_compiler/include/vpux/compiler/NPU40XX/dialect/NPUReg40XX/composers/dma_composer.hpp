//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace NPUReg40XX {

namespace DMADescriptorComposer {

Descriptors::DMARegister compose(VPUASM::NNDMAOp origOp, ELF::SymbolReferenceMap& symRefMap);

}  // namespace DMADescriptorComposer
}  // namespace NPUReg40XX
}  // namespace vpux
