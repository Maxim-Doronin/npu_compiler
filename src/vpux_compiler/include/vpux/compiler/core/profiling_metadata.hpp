//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {

std::vector<uint8_t> buildProfilingMetadataBuffer(net::NetworkInfoOp netOp, mlir::func::FuncOp funcOp, Logger log);

};  // namespace vpux
