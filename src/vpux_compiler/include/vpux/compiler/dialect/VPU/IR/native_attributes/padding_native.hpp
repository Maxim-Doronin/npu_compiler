//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>

namespace vpux::VPU {
class PaddingAttr;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {
class Padding {
private:
    int64_t left;
    int64_t right;
    int64_t top;
    int64_t bottom;

public:
    Padding() = default;
    Padding(int64_t leftPad, int64_t rightPad, int64_t topPad, int64_t bottomPad)
            : left(leftPad), right(rightPad), top(topPad), bottom(bottomPad) {
    }
    ~Padding() = default;

    friend bool operator==(const Padding& lhs, const Padding& rhs) {
        return lhs.left == rhs.left && lhs.right == rhs.right && lhs.top == rhs.top && lhs.bottom == rhs.bottom;
    }

    int64_t getTopPad() const {
        return top;
    }
    int64_t getBottomPad() const {
        return bottom;
    }
    int64_t getLeftPad() const {
        return left;
    }
    int64_t getRightPad() const {
        return right;
    }

    static Padding getClassFromAttr(PaddingAttr paddingAttr);

    static PaddingAttr getAttrFromClass(mlir::MLIRContext* ctx, const Padding& padding);

    void printFormat(llvm::raw_ostream& stream) const;
};
}  // namespace VPU
}  // namespace vpux
