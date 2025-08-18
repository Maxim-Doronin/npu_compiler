//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

VPU::Padding VPU::Padding::getClassFromAttr(PaddingAttr paddingAttr) {
    if (paddingAttr == nullptr) {
        return {};
    }

    auto left = paddingAttr.getLeft().getInt();
    auto right = paddingAttr.getRight().getInt();
    auto top = paddingAttr.getTop().getInt();
    auto bottom = paddingAttr.getBottom().getInt();

    return Padding(left, right, top, bottom);
}

VPU::PaddingAttr VPU::Padding::getAttrFromClass(mlir::MLIRContext* ctx, const Padding& padding) {
    auto topAttr = vpux::getIntAttr(ctx, padding.top);
    auto bottomAttr = vpux::getIntAttr(ctx, padding.bottom);
    auto leftAttr = vpux::getIntAttr(ctx, padding.left);
    auto rightAttr = vpux::getIntAttr(ctx, padding.right);

    return PaddingAttr::get(ctx, leftAttr, rightAttr, topAttr, bottomAttr);
};

void VPU::Padding::printFormat(llvm::raw_ostream& stream) const {
    std::unordered_map<std::string, int64_t> map;
    map["left"] = left;
    map["right"] = right;
    map["top"] = top;
    map["bottom"] = bottom;
    printTo(stream, "pads = ");
    vpux::MapFormatProvider::format(map, stream, {});
}
