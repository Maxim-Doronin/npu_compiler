//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "intel_npu/config/config.hpp"
#include "vpux/utils/logger/common_logger.hpp"

#include <openvino/runtime/properties.hpp>

namespace vpux {

//
// Used to initialize VPUX logger
//

LogLevel getLogLevel(ov::log::Level);
LogLevel getLogLevel(const intel_npu::Config&);

}  // namespace vpux
