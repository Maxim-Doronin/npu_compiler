//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/Support/FormatVariadic.h>

#include <sstream>
#include <vector>

using namespace vpux::profiling;

namespace {

std::string stringifyShapes(const std::vector<TensorShapeInfo>& shapes) {
    std::ostringstream buffer;
    buffer << "[";
    size_t counter = 0;
    for (const auto& shape : shapes) {
        const auto& [dimensions, typeString] = shape;
        if (counter++ > 0) {
            buffer << ", ";
        }
        for (auto dimension : dimensions) {
            buffer << dimension << "x";
        }
        buffer << typeString;
    }
    buffer << "]";
    return buffer.str();  // e.g. [1x3x10x62xf32, 1x3x10x62xf32]
}

auto formatVariantOffsets(const std::vector<uint32_t>& offsets) {
    constexpr unsigned maxVariantOffsetDimensionN = 3;
    VPUX_THROW_UNLESS(offsets.size() == maxVariantOffsetDimensionN, "Incorrect size");
    return llvm::formatv("[{0}, {1}, {2}]", offsets[0], offsets[1], offsets[2]);
}

}  // namespace

namespace vpux::profiling {

std::string to_string(const std::vector<TensorShapeInfo>& shapeInfo, unsigned short gatherIndices, bool isDynamic) {
    std::ostringstream shapeString;
    if (gatherIndices) {
        shapeString << std::to_string(gatherIndices) << " chunks of ";
    }
    shapeString << stringifyShapes(shapeInfo);
    if (isDynamic) {
        shapeString << " dynamic strides";
    }
    return shapeString.str();
}

CustomArgsVector to_custom_args(const TensorInfo& tensorInfo) {
    auto const& [inputs, output] = tensorInfo;
    return {{"Input tensors", stringifyShapes(inputs)}, {"Output tensors", stringifyShapes({output})}};
}

CustomArgsVector to_custom_args(const DPUVariantInfo& variantInfo) {
    auto const [inStart, inEnd, outStart, outEnd] = variantInfo;
    CustomArgsVector result;
    if (!inStart.empty()) {  // empty on NPU37XX
        result.emplace_back("inStart", formatVariantOffsets(inStart));
    }
    if (!inEnd.empty()) {  // empty on NPU37XX
        result.emplace_back("inEnd", formatVariantOffsets(inEnd));
    }
    result.emplace_back("outStart", formatVariantOffsets(outStart));
    result.emplace_back("outEnd", formatVariantOffsets(outEnd));
    return result;
}

}  // namespace vpux::profiling
