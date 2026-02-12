//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/singleton_initializer.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/singleton_initializer.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <functional>
#include <memory>

using namespace vpux;

namespace {

FuncRef<void(mlir::MLIRContext*, std::optional<config::Platform>)> getSingletonInitializer(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPU::arch37xx::initializeSingletonCache;
    case config::ArchKind::NPU40XX:
        return VPU::arch40xx::initializeSingletonCache;
    case config::ArchKind::NPU50XX:
        return VPU::arch50xx::initializeSingletonCache;
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

}  // namespace

namespace vpux {
namespace VPU {

// Extension class to register constraints for a specific architecture
class SingletonExtension : public mlir::DialectExtension<SingletonExtension, VPUDialect> {
public:
    explicit SingletonExtension(const DeviceVersion& deviceVersion): _deviceVersion(deviceVersion) {
    }

    void apply(mlir::MLIRContext* context, VPUDialect* /*dialect*/) const override {
        auto singletonInitializer = getSingletonInitializer(_deviceVersion.arch);
        singletonInitializer(context, _deviceVersion.platform);
    }

private:
    DeviceVersion _deviceVersion;
};

void initializeSingletonCache(mlir::DialectRegistry& registry, const DeviceVersion& deviceVersion) {
    registry.addExtension(mlir::TypeID::get<SingletonExtension>(), std::make_unique<SingletonExtension>(deviceVersion));
}

}  // namespace VPU
}  // namespace vpux
