//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/small_vector.hpp"

using namespace vpux;

int64_t vpux::getSizeAlignmentForSwizzling(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return SWIZZLING_SIZE_ALIGNMENT_VPUX37XX;
    case config::ArchKind::NPU40XX:
        return SWIZZLING_SIZE_ALIGNMENT_VPUX40XX;
    default: {
        return SWIZZLING_SIZE_ALIGNMENT_VPUX50XX;
    }
    }
    VPUX_THROW("Architecture {0} does not support swizzling", arch);
}

int64_t vpux::getAddressAlignmentForSwizzling(int64_t swizzlingKey, config::ArchKind archKind) {
    if (swizzlingKey < 1 || swizzlingKey > 5) {
        return 0;
    }

    // Alignment for arch is defined by ( 2^swizzleKey * Smallest RamCut Size)
    const EnumMap<int64_t, int64_t> swizzlingAddressAlignment = {{1, 1024},
                                                                 {2, 2048},
                                                                 {3, 4096},
                                                                 {4, 8192},
                                                                 {5, 16384}};
    int64_t archMultiplier = archKind >= config::ArchKind::NPU40XX ? 2 : 1;
    return swizzlingAddressAlignment.at(swizzlingKey) * archMultiplier;
}

int64_t vpux::alignSizeForSwizzling(int64_t size, int64_t sizeAlignment) {
    if (size % sizeAlignment) {
        size += sizeAlignment - size % sizeAlignment;
    }
    return size;
}

Byte vpux::calculateAlignedBuffersMemoryRequirement(SmallVector<Byte>& bufferSizes, const Byte offsetAlignment,
                                                    const Byte sizeAlignment) {
    int64_t bufferSizesSum = 0;

    VPUX_THROW_UNLESS(offsetAlignment.count() > 0, "offsetAlignment parameter should be >=1 byte.");
    VPUX_THROW_UNLESS(sizeAlignment.count() > 0, "sizeAlignment parameter should be >=1 byte.");
    for (auto buffSize : bufferSizes) {
        VPUX_THROW_UNLESS(buffSize.count() > 0, "Zero-sized buffer allocation requested.");
        bufferSizesSum += buffSize.count();
    }

    if (offsetAlignment == Byte(1) && sizeAlignment == Byte(1)) {
        // A simple sum will do in this case.
        return Byte(bufferSizesSum);
    }

    // sort buffers by decreasing size of offset required to fill the offsetAlignment alignment requirement
    SmallVector<std::pair<int64_t, int64_t>> buffersAlignments;
    SmallVector<int64_t> bufferSizesSorted;

    for (auto buff : bufferSizes) {
        int64_t delta = buff.count() % offsetAlignment.count() == 0
                                ? 0
                                : offsetAlignment.count() - buff.count() % offsetAlignment.count();
        buffersAlignments.push_back(std::make_pair(buff.count(), delta));
    }
    llvm::sort(buffersAlignments.begin(), buffersAlignments.end(),
               [](std::pair<int64_t, int64_t> a, std::pair<int64_t, int64_t> b) {
                   return a.second > b.second;
               });
    for (auto ba : buffersAlignments) {
        bufferSizesSorted.push_back(ba.first);
    }

    // calculate allocation requirements
    int64_t offset = 0;
    for (auto& buffSize : bufferSizesSorted) {
        if (offset % offsetAlignment.count() != 0) {
            // can't allocate here, calculate next possible start address
            offset += offsetAlignment.count() - offset % offsetAlignment.count();
        }
        // calculate memory requirement for the buffer
        offset += buffSize % sizeAlignment.count() == 0
                          ? buffSize
                          : (buffSize / sizeAlignment.count() + 1) * sizeAlignment.count();
    }

    return Byte(offset);
}
