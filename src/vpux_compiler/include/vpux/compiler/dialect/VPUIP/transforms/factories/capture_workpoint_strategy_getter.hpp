//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/impl/capture_workpoint_strategy.hpp"

namespace vpux::VPUIP {

std::unique_ptr<ICaptureWorkpointStrategy> createCaptureWorkpointStrategy(config::ArchKind arch);

}  // namespace vpux::VPUIP
