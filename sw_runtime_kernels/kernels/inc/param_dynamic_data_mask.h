//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "kernel_params.hpp"

namespace sw_params {

struct __attribute__((packed)) DynamicDataMaskParams : KernelTensors<1, 1> {};

}  // namespace sw_params
