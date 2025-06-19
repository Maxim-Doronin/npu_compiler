//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

namespace vpux::VPU {
/* @brief
 * Static class for generating MPEEngine attributes.
 */
class MPEEngineConfig {
public:
    static MPEEngineAttr retrieveMPEEngineAttribute(mlir::Operation* operation) {
        const auto arch = VPU::getArch(operation);

        VPUX_THROW_WHEN(arch == VPU::ArchKind::UNKNOWN,
                        "An unknown architecture is associated to the provided operation");

        return MPEEngine37XXAttr::get(operation->getContext(),
                                      MPEEngine37XXModeAttr::get(operation->getContext(), MPEEngine37XXMode::SCL));
    }

    template <typename ConcreteOp>
    static MPEEngineAttr retrieveMPEEngineAttribute(ConcreteOp operation, bool) {
        static_assert(std::is_same_v<ConcreteOp, IE::ConvolutionOp>, "Invalid operation, expected IE::ConvolutionOp");

        return retrieveMPEEngineAttribute(operation);
    }

    static bool useNewWeightTableFormat(mlir::Operation*, MPEEngineAttr) {
        return false;
    }
};

}  // namespace vpux::VPU
