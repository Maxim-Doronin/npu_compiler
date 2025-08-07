//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// see comments for 37xx-specific version

#pragma once

#include <cstdint>
#include <cstdlib>

namespace npu40xx {

// clang-format off

#include <details/api/vpu_cmx_info_40xx.h>
#include <details/api/vpu_dma_hw_40xx.h>
#include <details/api/vpu_media_hw.h>
#include <details/api/vpu_nce_hw_40xx.h>
#include <details/api/vpu_nnrt_api_40xx.h>
#include <details/api/vpu_nnrt_wlm.h>
#include <details/api/vpu_pwrmgr_api.h>

static constexpr size_t NNRT_API_UD2024_44_MAJOR_VERSION = 11;
static constexpr size_t NNRT_API_UD2024_44_MINOR_VERSION = 4;
static constexpr size_t NNRT_API_UD2024_44_PATCH_VERSION = 10;

// In 11.5.0 barrier FIFOs support for WLM was introduced
static constexpr size_t NNRT_API_WLM_BARRIER_FIFO_MAJOR_VERSION = 11;
static constexpr size_t NNRT_API_WLM_BARRIER_FIFO_MINOR_VERSION = 5;
static constexpr size_t NNRT_API_WLM_BARRIER_FIFO_PATCH_VERSION = 0;

static constexpr auto VPU_METADATA_STORAGE_START = static_cast<uint32_t>(npu40xx::nn_public::align_storage(alignof(npu40xx::nn_public::VpuDPUInvariant),
                                                                 npu40xx::nn_public::VPU_METADATA_STORAGE_ADDR));

// clang-format on

}  // namespace npu40xx
