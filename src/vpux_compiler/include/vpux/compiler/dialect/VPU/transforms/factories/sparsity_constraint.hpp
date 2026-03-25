//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux {
namespace VPU {

VPU::SparsityConstraint getSparsityConstraint(config::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
