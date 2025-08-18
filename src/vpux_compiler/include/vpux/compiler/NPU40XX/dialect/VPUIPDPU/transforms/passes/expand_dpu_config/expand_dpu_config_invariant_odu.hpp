//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>

#include <optional>

namespace vpux {
class NDTypeInterface;
}
namespace vpux::VPU {
enum class MPEMode : uint64_t;
}
namespace vpux::VPUIP {
enum class NCETaskType : uint64_t;
}

namespace vpux::VPUIPDPU::arch40xx::ODU {

struct ODUConfig {
    struct OutTensorSize {
        uint32_t dimX = 0;
        uint32_t dimY = 0;
        uint32_t dimZ = 0;
    } outTensorSize;
    struct DataReuse {
        ODUActivationReuseMode activationReuse = ODUActivationReuseMode::NTHW_1;
    } dataReuse;
    struct PermuteData {
        ODUPermuteDataMode permuteMode = ODUPermuteDataMode::PERMUTE_ZXY;
    } permuteData;
    struct Sparsity {
        std::optional<bool> compressionEnabled;
        std::optional<int64_t> sparseValue;
    } sparsity;
    struct SwizzleData {
        DPUSwizzleKey swizzleKey = DPUSwizzleKey::SWIZZLE_OFF;
    } swizzleData;
    struct OutActivations {
        std::optional<ODUDataBitWidth> dataWidth;
    } outActivations;
    struct MemoryMode {
        ODUMemoryMode memMode = ODUMemoryMode::MODE_DENSE;
    } memoryMode;
};

std::optional<ODUDataBitWidth> getOutDataWidth(mlir::Type outDataType);

mlir::LogicalResult configureOutTensorSize(const Logger& log, ODUConfig::OutTensorSize& config,
                                           ODUPermuteDataMode permuteMode, const Strides& outStrides);
mlir::LogicalResult configureDataReuse(const Logger& log, ODUConfig::DataReuse& config, VPU::MPEMode mpeFrequentMode,
                                       VPUIP::NCETaskType dpuTaskType);
mlir::LogicalResult configurePermuteMode(const Logger& log, ODUConfig::PermuteData& config,
                                         const DimsOrder& outDimsOrder);
mlir::LogicalResult configureSparsity(const Logger&, ODUConfig::Sparsity& config, bool outSparsityEnabled,
                                      int64_t sparseValue);
mlir::LogicalResult configureSwizzleData(const Logger& log, ODUConfig::SwizzleData& config,
                                         std::optional<int64_t> outSwizzling);
mlir::LogicalResult configureOutActivations(const Logger& log, ODUConfig::OutActivations& config,
                                            mlir::Type outDataType);
mlir::LogicalResult configureMemoryMode(const Logger& log, ODUConfig::MemoryMode& config,
                                        std::optional<bool> isSuperdense);
mlir::LogicalResult configureODU(const Logger& log, ODUConfig& config, const NDTypeInterface& outActType,
                                 VPU::MPEMode mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                 std::optional<int64_t> outSwizzling, std::optional<bool> isSuperdense,
                                 bool outSparsityEnabled);

mlir::LogicalResult buildODUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const ODUConfig& config, mlir::Value outAct, mlir::Value outSparsityMap);

}  // namespace vpux::VPUIPDPU::arch40xx::ODU
