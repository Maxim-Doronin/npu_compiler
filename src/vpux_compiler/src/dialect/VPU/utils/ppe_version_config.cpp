//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

using namespace vpux;

namespace vpux {
namespace VPU {

PPEVersionConfig& getPpeConfig(mlir::MLIRContext* context) {
    return getCache<VPU::PPEVersionConfig, VPU::VPUDialect>(context);
}

void setPpeFactory(mlir::MLIRContext* context, std::unique_ptr<IPpeFactory> ppeFactory) {
    auto& registeredInterface = getPpeConfig(context);
    registeredInterface.setPpeFactory(std::move(ppeFactory));
}

const IPpeFactory& getPpeFactory(mlir::MLIRContext* context) {
    auto& registeredInterface = getPpeConfig(context);
    return registeredInterface.getFactory();
}

}  // namespace VPU
}  // namespace vpux
