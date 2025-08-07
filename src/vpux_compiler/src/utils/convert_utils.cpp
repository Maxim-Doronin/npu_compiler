//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/bit.h>

using namespace vpux;

namespace {
template <typename DataType>
void unpackSubByteData(MutableArrayRef<DataType> targetData, ArrayRef<DataType> sourceData, bool isSplat,
                       size_t bitWidth, size_t elemPerByte) {
    static_assert(sizeof(DataType) == sizeof(uint8_t), "This code assumes 8-bit inputs / outputs");

    const size_t rShift = CHAR_BIT - bitWidth;
    if (isSplat) {
        // when splat, all sub-byte elements must be identical - so take any, in
        // this case, leftmost
        auto unpacked = sourceData.front() >> rShift;
        std::fill_n(targetData.data(), targetData.size(), unpacked);
        return;
    }

    const auto numBytes = sourceData.size();
    for (size_t byteIdx = 0; byteIdx < numBytes; byteIdx++) {
        for (size_t elemIdxPerByte = 0; elemIdxPerByte < elemPerByte; elemIdxPerByte++) {
            const size_t lShift = rShift - (elemIdxPerByte * bitWidth);
            // convert to *unsigned* type to perform well-defined left shift
            auto unsignedVal = llvm::bit_cast<uint8_t>(sourceData[byteIdx]);
            unsignedVal <<= lShift;
            // convert back to (potentially signed) data type to perform right
            // shift (with sign extension when signed)
            const auto val = llvm::bit_cast<DataType>(unsignedVal);
            targetData[byteIdx * elemPerByte + elemIdxPerByte] = (val >> rShift);
        }
    }
}
}  // namespace

// Unpack subbyte constants.
// Consider this configuration for a bit width of 4:
// 0x89, 0x88
// Will end like:
// 0x9, 0x8, 0x8, 0x8
vpux::Const::Content vpux::subByteConversion(vpux::Const::Content& input, vpux::NDTypeInterface outputType,
                                             bool outputIsSplat, size_t bitWidth) {
    const auto elementType = outputType.getElementType();
    auto output = Const::Content::allocTempBuffer(outputType, elementType, outputIsSplat);

    const auto elemPerByte = CHAR_BIT / bitWidth;
    VPUX_THROW_UNLESS(vpux::isPowerOfTwo(elemPerByte), "Invalid number of elements per byte '{0}'", elemPerByte);

    // When bitWidth is 1, we treat it as unsigned numbers
    if (elementType.isUnsignedInteger() || elementType.isSignlessIntOrIndex() || bitWidth == 1) {
        const auto sourceData = input.getStorageBuf<uint8_t>();
        auto targetData = output.getTempBuf<uint8_t>();
        unpackSubByteData(targetData, sourceData, input.isSplat(), bitWidth, elemPerByte);
        return output;
    } else if (elementType.isSignedInteger()) {
        // For signed integer, we need maintain the sign bit when unpacking
        const auto sourceData = input.getStorageBuf<int8_t>();
        auto targetData = output.getTempBuf<int8_t>();
        unpackSubByteData(targetData, sourceData, input.isSplat(), bitWidth, elemPerByte);
        return output;
    } else {
        VPUX_THROW("Unsupported subByte conversion type '{0}'", elementType);
    }

    return output;
}
