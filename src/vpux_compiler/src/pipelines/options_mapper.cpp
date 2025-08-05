//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/options.hpp"

#include "vpux/compiler/compilation_options.hpp"
#include "vpux/compiler/compiler.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/utils/IE/private_properties.hpp"

#include "vpux/compiler/NPU37XX/pipeline_options.hpp"
#include "vpux/compiler/NPU40XX/pipeline_options.hpp"

#include <openvino/runtime/properties.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

namespace {

uint32_t getPlatformDPUClusterNum(const std::string& platform) {
    if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPUX37XX_MAX_DPU_GROUPS;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPUX40XX_MAX_DPU_GROUPS;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

std::optional<int> getMaxTilesValue(const intel_npu::Config& config) {
    if (config.has<intel_npu::MAX_TILES>()) {
        auto logger = vpux::Logger::global();
        int maxTiles = checked_cast<int>(config.get<intel_npu::MAX_TILES>());
        std::string platformName = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
        // E#117389: remove overrides and change to exceptions once driver & plugin will be fixed
        const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(platformName));
        if (maxTiles < 1 || maxTiles > maxArchTiles) {
            logger.warning("Invalid number of NPU_MAX_TILES for requested arch, got {0}. Override to {1}", maxTiles,
                           maxArchTiles);
            maxTiles = maxArchTiles;
        }
        return maxTiles;
    }
    return std::nullopt;
}

int getMaxDPUClusterNum(const intel_npu::Config& config) {
    std::string platformName = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    const int maxArchTiles = checked_cast<int>(getPlatformDPUClusterNum(platformName));
    const auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value()) {
        return maybeMaxTiles.value();
    }
    return maxArchTiles;
}

template <typename Options>
std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->performanceHintOverride;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == config::CompilationMode::ReferenceSW) {
        return getPerformanceHintOverride<ReferenceSWOptions>(config);
    } else if (compilationMode == config::CompilationMode::DefaultHW ||
               compilationMode == config::CompilationMode::WSMonolithic ||
               compilationMode == config::CompilationMode::WSInit ||
               compilationMode == config::CompilationMode::HostCompile) {
        return getPerformanceHintOverride<DefaultHWOptions>(config);
    } else if (compilationMode == config::CompilationMode::ShaveCodeGen) {
        return getPerformanceHintOverride<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getPerformanceHintOverride<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getPerformanceHintOverride<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

int getNumberOfDPUGroupsUnchecked(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    const auto& performanceHintOverride = getPerformanceHintOverride(config);
    // NPUPerformanceMode consists of same enums as ov::hint::PerformanceMode + EFFICIENCY
    // In future, ov::hint::PerformanceMode can be extended with the new value, so
    // we do not need to have our own enum class
    enum class NPUPerformanceMode {
        LATENCY = 1,                //!<  Optimize for latency
        THROUGHPUT = 2,             //!<  Optimize for throughput
        CUMULATIVE_THROUGHPUT = 3,  //!<  Optimize for cumulative throughput
        EFFICIENCY = 4,             //!<  Optimize for power efficiency
    };

    const auto performanceMode = [&] {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::LATENCY:
            VPUX_THROW_WHEN(!performanceHintOverride.has_value(), "performance-hint-override does not hold a value.");
            if (performanceHintOverride.value() == "efficiency") {
                return NPUPerformanceMode::EFFICIENCY;
            } else if (performanceHintOverride.value() == "latency") {
                return NPUPerformanceMode::LATENCY;
            }
            VPUX_THROW("Unknown value `{0}` for performance-hint-override. Possible values: `latency`, `efficiency`",
                       performanceHintOverride.value());
        case ov::hint::PerformanceMode::THROUGHPUT:
        default:
            break;
        }
        return static_cast<NPUPerformanceMode>(config.get<intel_npu::PERFORMANCE_HINT>());
    }();

    if (platform == ov::intel_npu::Platform::NPU3720) {
        switch (performanceMode) {
        case NPUPerformanceMode::THROUGHPUT:
        case NPUPerformanceMode::LATENCY:
        case NPUPerformanceMode::EFFICIENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        switch (performanceMode) {
        case NPUPerformanceMode::LATENCY:
            return getMaxDPUClusterNum(config);
        case NPUPerformanceMode::EFFICIENCY:
            return 4;
        case NPUPerformanceMode::THROUGHPUT:
        default:
            return 2;
        }
    } else {
        switch (performanceMode) {
        case NPUPerformanceMode::THROUGHPUT:
            return 1;
        case NPUPerformanceMode::EFFICIENCY:
        case NPUPerformanceMode::LATENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    }
}

}  // namespace

namespace vpux {

//
// getArchKind
//

VPU::ArchKind getArchKind(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::AUTO_DETECT) {
        return VPU::ArchKind::UNKNOWN;
    } else if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPU::ArchKind::NPU37XX;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPU::ArchKind::NPU40XX;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

//
// getCompilationMode
//

config::CompilationMode getCompilationMode(const intel_npu::Config& config) {
    if (!config.has<intel_npu::COMPILATION_MODE>()) {
        return config::CompilationMode::DefaultHW;
    }

    const auto parsed = config::symbolizeCompilationMode(config.get<intel_npu::COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported compilation mode '{0}'",
                      config.get<intel_npu::COMPILATION_MODE>());
    return parsed.value();
}

//
// getRevisionID
//

std::optional<int> getRevisionID(const intel_npu::Config& config) {
    if (config.has<intel_npu::STEPPING>()) {
        return checked_cast<int>(config.get<intel_npu::STEPPING>());
    }
    return std::nullopt;
}

//
// getNumberOfDPUGroups
//

std::optional<int> getNumberOfDPUGroups(const intel_npu::Config& config) {
    if (config.has<intel_npu::TILES>()) {
        int requestedNpuTiles = checked_cast<int>(config.get<intel_npu::TILES>());
        int maxTiles = getMaxDPUClusterNum(config);
        if (requestedNpuTiles > maxTiles) {
            vpux::Logger::global().warning(
                    "Requested number of NPU tiles is larger than maximum available tiles: ({0}) "
                    "> ({1}). Override to ({1})",
                    requestedNpuTiles, maxTiles);
            requestedNpuTiles = maxTiles;
        }
        return requestedNpuTiles;
    }

    int numOfDpuGroups = getNumberOfDPUGroupsUnchecked(config);
    auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value() && (numOfDpuGroups > maybeMaxTiles.value())) {
        vpux::Logger::global().warning(
                "PERFORMANCE_HINT parameter used more NPU_TILES ({0}) than MAX_TILES ({1}). Override to ({1})",
                numOfDpuGroups, maybeMaxTiles.value());
        numOfDpuGroups = maybeMaxTiles.value();
    }

    return numOfDpuGroups;
}

//
// getNumberOfDMAEngines
//

std::optional<int> getNumberOfDMAEngines(const intel_npu::Config& config) {
    if (config.has<intel_npu::DMA_ENGINES>()) {
        return checked_cast<int>(config.get<intel_npu::DMA_ENGINES>());
    }

    auto archKind = vpux::getArchKind(config);
    auto numOfDpuGroups = getNumberOfDPUGroups(config);
    int maxDmaPorts = VPU::getMaxDMAPorts(archKind);
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());
    const auto maxArchTiles = getPlatformDPUClusterNum(platform);

    auto getNumOfDmaPortsWithDpuCountLimit = [&]() {
        /*For architectures that have only 1 cluster, we want to bypass
        (numOfDPUGroups >= numOfDMAPorts) requirement

        Revert this back to "return maxDmaPorts" and implement E#135226 */
        if (maxArchTiles == 1) {
            return 1;
        } else {
            return std::min(maxDmaPorts, numOfDpuGroups.value_or(maxDmaPorts));
        }
    };

    if (platform == ov::intel_npu::Platform::NPU3720) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    } else {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    }
}

//
// getAvailableCmx
//

Byte getAvailableCmx(const intel_npu::Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPUX37XX_CMX_WORKSPACE_SIZE;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPUX40XX_CMX_WORKSPACE_SIZE;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

template <typename Options>
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableVerifiers;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == config::CompilationMode::ReferenceSW) {
        return getEnableVerifiers<ReferenceSWOptions>(config);
    } else if (compilationMode == config::CompilationMode::DefaultHW ||
               compilationMode == config::CompilationMode::HostCompile) {
        return getEnableVerifiers<DefaultHWOptions>(config);
    } else if (compilationMode == config::CompilationMode::ShaveCodeGen) {
        return getEnableVerifiers<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getWlmEnabled(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->workloadManagementEnable;
}

std::optional<bool> getWlmEnabled(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return std::nullopt;
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getWlmEnabled<DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getWlmRollback(const intel_npu::Config& config) {
    if (getCompilationMode(config) == config::CompilationMode::ReferenceSW) {
        return std::nullopt;
    }

    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    return options != nullptr ? std::optional<bool>{options->wlmRollback} : std::nullopt;
}

std::optional<bool> getWlmRollback(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return std::nullopt;
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getWlmRollback<DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

// Adaptive Stripping

std::optional<bool> getQDQOptimization(const intel_npu::Config& config) {
    if (config.has<intel_npu::QDQ_OPTIMIZATION>()) {
        return config.get<intel_npu::QDQ_OPTIMIZATION>();
    }

    return std::nullopt;
}

std::optional<bool> getEnableVerifiers(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableVerifiers<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableVerifiers<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableMemoryUsageCollector;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == config::CompilationMode::ReferenceSW) {
        return getEnableMemoryUsageCollector<ReferenceSWOptions>(config);
    } else if (compilationMode == config::CompilationMode::DefaultHW ||
               compilationMode == config::CompilationMode::HostCompile) {
        return getEnableMemoryUsageCollector<DefaultHWOptions>(config);
    } else if (compilationMode == config::CompilationMode::ShaveCodeGen) {
        return getEnableMemoryUsageCollector<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableMemoryUsageCollector(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableMemoryUsageCollector<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableMemoryUsageCollector<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableFunctionStatisticsInstrumentation;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == config::CompilationMode::ReferenceSW) {
        return getEnableFunctionStatisticsInstrumentation<ReferenceSWOptions>(config);
    } else if (compilationMode == config::CompilationMode::DefaultHW ||
               compilationMode == config::CompilationMode::HostCompile) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions>(config);
    } else if (compilationMode == config::CompilationMode::ShaveCodeGen) {
        return getEnableFunctionStatisticsInstrumentation<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getEnableFunctionStatisticsInstrumentation(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getEnableFunctionStatisticsInstrumentation<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getEnableFunctionStatisticsInstrumentation<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

template <typename Options>
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return options->enableDummyOpReplacement ? DummyOpMode::ENABLED : DummyOpMode::DISABLED;
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == config::CompilationMode::ReferenceSW) {
        return getDummyOpReplacement<ReferenceSWOptions>(config);
    } else if (compilationMode == config::CompilationMode::DefaultHW ||
               compilationMode == config::CompilationMode::HostCompile) {
        return getDummyOpReplacement<DefaultHWOptions>(config);
    } else if (compilationMode == config::CompilationMode::ShaveCodeGen) {
        return getDummyOpReplacement<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<DummyOpMode> getDummyOpReplacement(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getDummyOpReplacement<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getDummyOpReplacement<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getCompilerDynamicQuantization(const intel_npu::Config& config) {
    if (config.has<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>()) {
        return config.get<intel_npu::COMPILER_DYNAMIC_QUANTIZATION>();
    }

    return std::nullopt;
}

#ifdef BACKGROUND_FOLDING_ENABLED

template <typename Options>
std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto options =
            parseCompilationModeParams<Options>(config.get<intel_npu::COMPILATION_MODE_PARAMS>(), getArchKind(config));
    if (options == nullptr) {
        return std::nullopt;
    }
    return ConstantFoldingConfig{options->constantFoldingInBackground, options->constantFoldingInBackgroundNumThreads,
                                 options->constantFoldingInBackgroundCollectStatistics,
                                 options->constantFoldingInBackgroundMemoryUsageLimit,
                                 options->constantFoldingInBackgroundCacheCleanThreshold};
}

template <typename ReferenceSWOptions, typename DefaultHWOptions>
std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == config::CompilationMode::ReferenceSW) {
        return getConstantFoldingInBackground<ReferenceSWOptions>(config);
    } else if (compilationMode == config::CompilationMode::DefaultHW ||
               compilationMode == config::CompilationMode::HostCompile) {
        return getConstantFoldingInBackground<DefaultHWOptions>(config);
    } else if (compilationMode == config::CompilationMode::ShaveCodeGen) {
        return getConstantFoldingInBackground<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<ConstantFoldingConfig> getConstantFoldingInBackground(const intel_npu::Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU37XX) {
        return getConstantFoldingInBackground<ReferenceSWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getConstantFoldingInBackground<ReferenceSWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

#endif

}  // namespace vpux
