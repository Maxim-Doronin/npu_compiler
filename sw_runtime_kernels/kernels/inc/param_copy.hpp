//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>
#include <stddef.h>

#ifdef __cplusplus
namespace sw_params {
#endif

#define MAX_COPY_DIMS 4
struct __attribute__((packed)) CopyParams {
    struct MemRefData input;
    struct MemRefData output;

    int64_t inBitOffsets[MAX_COPY_DIMS];
    int64_t outBitOffsets[MAX_COPY_DIMS];
};

#ifdef __cplusplus
}  // namespace sw_params
#endif
