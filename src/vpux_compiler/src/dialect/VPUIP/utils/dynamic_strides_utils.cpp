//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dynamic_strides_utils.hpp"

using namespace vpux;

namespace {
Shape canonicalizeStrides(const MemStrides& strides, Bit elemSize) {
    Shape canonicalStrides;
    Bit previousInStride = elemSize;
    std::for_each(strides.rbegin(), strides.rend(), [&](Bit stride) {
        if (stride.count() / previousInStride.count() != 1) {
            canonicalStrides.push_back(stride.count() / elemSize.count());
        }
        previousInStride = stride;
    });

    return canonicalStrides;
}
}  // namespace

/*
    Strides are considered compatible when they are the same save for the expansion and contraction
    of unit dimensions. For example consider following cases:
    1. Trivial case:
    inType: 10x10xf16 outType 10x10xf16
    In this case strides are the same so they are compatible.
    2. Type expanded and contracted with unit dimensions
    inType 10x10x1xf16 outType 1x10x1x10xf16
    In this case strides are different since type has been expanded but they still should be
    considered compatible. To account for this case below algorithm first canonicalizes strides
    to remove unit dimensions and then compares the strides. True shape recovery is later on
    ensured by the backend code.
    3. Type reshaped to be incompatible
    inType 10x10xf16 outType 1x100xf16
    In this case below algorithm returns false.
*/
bool VPUIP::areStridesCompatible(const MemStrides& inStrides, Bit inElemSize, const MemStrides& outStrides,
                                 Bit outElemSize) {
    auto canonicalInStrides = canonicalizeStrides(inStrides, inElemSize);
    auto canonicalOutStrides = canonicalizeStrides(outStrides, outElemSize);
    /*
        Due to tiling on final dimension which can't be recovered from strides, strides can
        mismatch by at most 1 element. To understand why this can be the case consider following pattern:

        %arg : memref<2x2xui8>
        %reshaped = VPUIP.GenericReshape input(%arg) -> memref<1x2x2x1xui8>
        %slice = VPUIP.SubView %reshaped [0, 0, 0, 0] [1, 1, 2, 1] : memref<1x2x2x1xui8> to memref<1x1x2x1xui8>

        For a final slice input strides are [4, 2, 1, 1, 1] but output strides are [2, 2, 1, 1, 1].
        Notice how final stride on SubView output is not a real stride of the tensor. After canonicalization:
        input: [4, 2, 1], output: [2, 1]
        And the final dimension is mismatched. Since tiling on final dimension is always valid we simply allow
        strides to mismatch by at most 1.
    */
    if (std::abs(static_cast<int>(canonicalInStrides.size()) - static_cast<int>(canonicalOutStrides.size())) > 1) {
        return false;
    }

    return std::equal(canonicalInStrides.begin(),
                      canonicalInStrides.begin() + std::min(canonicalInStrides.size(), canonicalOutStrides.size()),
                      canonicalOutStrides.begin());
}
