//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

using namespace vpux;

SmallVector<VPUIP::DmaChannelType> vpux::getDMAChannelsWithIndependentLinkAgents(config::ArchKind arch) {
    if (arch <= config::ArchKind::NPU37XX) {
        return {VPUIP::DmaChannelType::NOT_SPECIFIED};
    }

    return {VPUIP::DmaChannelType::DDR, VPUIP::DmaChannelType::CMX};
}

// Encode DMA port and channel setting into a single integer for convenient usage during barrier scheduling
int64_t vpux::getDMAQueueIdEncoding(int64_t port, int64_t channelIdx) {
    return port * (VPUIP::getMaxEnumValForDmaChannelType() + 1) + channelIdx;
}
int64_t vpux::getDMAQueueIdEncoding(int64_t port, std::optional<vpux::VPUIP::DmaChannelType> channel) {
    return getDMAQueueIdEncoding(port, static_cast<int64_t>(channel.value_or(VPUIP::DmaChannelType::NOT_SPECIFIED)));
}
int64_t vpux::getDMAQueueIdEncoding(std::optional<vpux::VPUIP::DmaChannelType> channel) {
    return getDMAQueueIdEncoding(0, static_cast<int64_t>(channel.value_or(VPUIP::DmaChannelType::NOT_SPECIFIED)));
}

int64_t vpux::getDMAQueueIdEncoding(VPU::MemoryKind srcMemKind, config::ArchKind arch) {
    if (arch <= config::ArchKind::NPU37XX) {
        return getDMAQueueIdEncoding(std::nullopt);
    }

    if (srcMemKind == VPU::MemoryKind::DDR) {
        return getDMAQueueIdEncoding(VPUIP::DmaChannelType::DDR);
    }
    return getDMAQueueIdEncoding(VPUIP::DmaChannelType::CMX);
}

int64_t vpux::getDMAPortFromEncodedId(int64_t dmaQueueIdEncoding) {
    return dmaQueueIdEncoding / (VPUIP::getMaxEnumValForDmaChannelType() + 1);
}

VPUIP::DmaChannelType vpux::getDMAChannelTypeFromEncodedId(int64_t dmaQueueIdEncoding, config::ArchKind arch) {
    if (arch <= config::ArchKind::NPU37XX) {
        return VPUIP::DmaChannelType::NOT_SPECIFIED;
    }

    return static_cast<VPUIP::DmaChannelType>(dmaQueueIdEncoding % (VPUIP::getMaxEnumValForDmaChannelType() + 1));
}
