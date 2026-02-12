//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {
// TODO: E#194197 Move these utility functions to VPUIP utils

SmallVector<VPUIP::DmaChannelType> getDMAChannelsWithIndependentLinkAgents(config::ArchKind arch);

// Encode DMA port and channel setting into a single integer for convenient usage by scheduling modules
int64_t getDMAQueueIdEncoding(int64_t port, int64_t channelIdx);
int64_t getDMAQueueIdEncoding(int64_t port, std::optional<vpux::VPUIP::DmaChannelType> channel);
int64_t getDMAQueueIdEncoding(std::optional<vpux::VPUIP::DmaChannelType> channel);
int64_t getDMAQueueIdEncoding(VPU::MemoryKind srcMemKind, config::ArchKind arch);

int64_t getDMAPortFromEncodedId(int64_t dmaQueueIdEncoding);

VPUIP::DmaChannelType getDMAChannelTypeFromEncodedId(int64_t dmaQueueIdEncoding, config::ArchKind arch);

}  // namespace vpux
