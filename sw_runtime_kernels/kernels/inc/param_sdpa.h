//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <moviVectorTypes.h>
#include "kernel_params.hpp"

#ifdef __cplusplus
namespace sw_params {
#endif

struct __attribute__((packed)) SDPAParams : KernelTensors<5, 1> {};

#ifdef __cplusplus
}  // namespace sw_params
#endif
