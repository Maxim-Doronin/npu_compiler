//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/DialectInterface.h>
#include "vpux/compiler/dialect/config/version.hpp"

#include <cstdint>
#include <optional>

namespace vpux::config {

/**
 * @brief Structure for aggregating configuration constants.
 */
struct NPUConstraints {
    // PerformanceMetrics frequency values.
    struct {
        uint32_t base = 0;
        uint32_t step = 0;
    } frequencyTable;

    // perf_clk value after dividing by the default frequency divider.
    struct {
        double defaultFreq = 0;
    } perfClock;

    enum class MappedInferenceFormat {
        MappedInference,
        ManagedMappedInference
    } mappedInferenceFormat = MappedInferenceFormat::ManagedMappedInference;

    // Base ELF ABI version for given target, this may be overridden to a higher
    // version during compilation if a specific feature requires newer ABI
    // support
    std::optional<Version> baseElfAbiVersion;

    // Minimum ELF ABI version required to support dynamic strides if applicable
    std::optional<Version> dynamicStridesMinElfAbiVersion;
};

// ConfigCache inside of the MLIRContext to be able to access NPUConstraints from other modules.
class ConfigCache final : public mlir::DialectInterface::Base<ConfigCache> {
    NPUConstraints _npuConstraints;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConfigCache)

    ConfigCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const NPUConstraints& getConstraints() const {
        return _npuConstraints;
    }

    void setConstraints(const NPUConstraints& npuConstraints) {
        _npuConstraints = npuConstraints;
    }
};

/**
 * @brief Set the constraint structure cached in the MLIRContext at initialization stage.
 */
void setNPUConstraints(mlir::MLIRContext* context, const NPUConstraints& constraint);

/**
 * @brief Get constraints struct cached in the MLIRContext.
 */
const NPUConstraints& getNPUConstraints(mlir::MLIRContext* context);

}  // namespace vpux::config
