//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/hw_strategy_registry.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"

#include "vpux/compiler/NPU37XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/strategies_initializer.hpp"
#include "vpux/compiler/NPU50XX/dialect/config/constraints_initializer.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <functional>
#include <memory>

using namespace vpux;

namespace {

std::unique_ptr<config::IConstraintsInitializer> createConstraintsInitializer(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<config::ConstraintsInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<config::ConstraintsInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<config::ConstraintsInitializer50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

}  // namespace

namespace vpux {

IStrategiesInitializer::~IStrategiesInitializer() = default;

namespace config {

// Extension class to register constraints for a specific architecture
class ConstraintsExtension : public mlir::DialectExtension<ConstraintsExtension, ConfigDialect> {
public:
    explicit ConstraintsExtension(PlatformOrArch target): _target(target) {
    }

    void apply(mlir::MLIRContext* context, ConfigDialect* /*dialect*/) const override {
        auto constraintsInitializer = createConstraintsInitializer(getArch(_target));
        constraintsInitializer->initialize(context, _target);
    }

private:
    PlatformOrArch _target;
};

void registerConstraints(mlir::DialectRegistry& registry, PlatformOrArch target) {
    registry.addExtension(mlir::TypeID::get<ConstraintsExtension>(), std::make_unique<ConstraintsExtension>(target));
}

}  // namespace config

namespace IE {

std::unique_ptr<IStrategiesInitializer> createStrategiesInitializer(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<IE::StrategiesInitializer37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<IE::StrategiesInitializer40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<IE::StrategiesInitializer50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

class StrategiesExtension : public mlir::DialectExtension<StrategiesExtension, IEDialect> {
public:
    explicit StrategiesExtension(config::ArchKind arch): _arch(arch) {
    }

    void apply(mlir::MLIRContext* context, IEDialect*) const override {
        auto strategiesInitializer = IE::createStrategiesInitializer(_arch);
        strategiesInitializer->initialize(context);
    }

private:
    config::ArchKind _arch;
};

void registerStrategies(mlir::DialectRegistry& registry, config::ArchKind arch) {
    registry.addExtension(mlir::TypeID::get<StrategiesExtension>(), std::make_unique<StrategiesExtension>(arch));
}

}  // namespace IE
}  // namespace vpux
