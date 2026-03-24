//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::IE {
enum class InterpolateCoordMode : uint64_t;
class InterpolateCoordModeAttr;
class InterpolateModeAttr;
}  // namespace vpux::IE
namespace vpux::VPU {
enum class NCEInterpolateMode : uint64_t;
class NCEInterpolateModeAttr;
}  // namespace vpux::VPU

namespace vpux::VPU {

VPU::NCEInterpolateModeAttr getNCEInterpolateModeAttr(IE::InterpolateModeAttr origModeAttr);

bool isSupportedNCEInterpolateScales(ArrayRef<double> scales, vpux::LogCb logCb = emptyLogCb);
std::optional<SmallVector<double>> getNCEInterpolateScales(NDTypeInterface inputType, NDTypeInterface outputType,
                                                           IE::InterpolateCoordModeAttr coordModeAttr);

SmallVector<int64_t> getNCEInterpolateFactors(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                              IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolatePadsBegin(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolatePadsEnd(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                              IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolateKernelSize(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                                 IE::InterpolateCoordModeAttr coordModeAttr);
SmallVector<int64_t> getNCEInterpolateStrides(ArrayRef<double> scales, VPU::NCEInterpolateModeAttr modeAttr,
                                              IE::InterpolateCoordModeAttr coordModeAttr);

SmallVector<float> getNCEInterpolateKernelContent(ArrayRef<int64_t> kernelSize, const VPU::NCEInterpolateMode& mode,
                                                  const IE::InterpolateCoordMode& coordMode, ArrayRef<double> scales);

}  // namespace vpux::VPU
