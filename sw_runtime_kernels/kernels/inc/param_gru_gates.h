//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef __MOVICOMPILE__
#include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>
#include <cstddef>

#ifdef __cplusplus
namespace sw_params {
#endif

#pragma pack(push, 1)

struct GRUGatesParams {
    // Inputs
    struct MemRefData inputData;
    struct MemRefData initialHiddenState;
    struct MemRefData inputHidden;
    struct MemRefData biases;

    // Outputs
    struct MemRefData outputHiddenState;
};

#pragma pack(pop)

#ifdef __cplusplus
}  // namespace sw_params
#endif
