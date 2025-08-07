//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/parser/hw.hpp"
namespace npu37xx {
#include <vpux/compiler/NPU37XX/dialect/NPUReg37XX/firmware_headers/details/api/vpu_nce_hw_37xx.h>
}
namespace npu40xx {
#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nce_hw_40xx.h>
}

using namespace vpux;

constexpr bool shaveControlsDpuValue = false;

bool VPU::getShaveControlsDpu(VPU::ArchKind arch) {
    (void)arch;
    return shaveControlsDpuValue;
}

size_t VPU::getDpuDebugDataSize(VPU::ArchKind /*arch*/) {
    return sizeof(HwpDpuIduOduData_t);
}

size_t VPU::getDPUInvariantDataSize(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return sizeof(npu37xx::nn_public::VpuDPUInvariantRegisters);
    case VPU::ArchKind::NPU40XX:
        return sizeof(npu40xx::nn_public::VpuDPUInvariantRegisters);
    default:
        VPUX_THROW("Unable to get DPUInvariantDataSize for arch {0}", arch);
    }
}

size_t VPU::getDPUVariantDataSize(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return sizeof(npu37xx::nn_public::VpuDPUVariantRegisters);
    case VPU::ArchKind::NPU40XX:
        return sizeof(npu40xx::nn_public::VpuDPUVariantRegisters);
    default:
        VPUX_THROW("Unable to get DPUVariantDataSize for arch {0}", arch);
    }
}
