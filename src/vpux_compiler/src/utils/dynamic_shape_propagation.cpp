//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/utils/dynamic_shape_propagation.hpp>

namespace vpux {

void assignDynamicTypeComponents(TypeComponents& typeComponents, VPU::BoundsRepresentation boundsRepresentation,
                                 ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds) {
    if (boundsRepresentation == VPU::BoundsRepresentation::BOUNDS) {
        typeComponents.setShape(Shape(shape)).setBounds(Bounds(bounds));
    } else {
        auto dimsMask = to_small_vector(shape | transformed([&](auto dim) -> int64_t {
                                            return (dim == mlir::ShapedType::kDynamic) ? 1 : 0;
                                        }));

        typeComponents.setShape(Shape(bounds)).setDynamicDimsMask(DynamicDimsMask(dimsMask));
    }
}

}  // namespace vpux
