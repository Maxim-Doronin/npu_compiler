//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/options_mapper.hpp"
#include "vpux/utils/IE/config.hpp"

#include "intel_npu/config/options.hpp"

namespace vpux {

//
// OptionsWrapper is used for lit-test
// Please note the class does not own options
//

template <class OptionsType>
class OptionsWrapper final {
public:
    using value_type = OptionsType;

    OptionsWrapper(const VPU::InitCompilerOptions* initCompilerOptions, const OptionsType* options)
            : _initCompilerOptions(initCompilerOptions), _options(options) {
        VPUX_THROW_WHEN(_initCompilerOptions == nullptr, "initCompilerOptions is nullptr");
        VPUX_THROW_WHEN(_options == nullptr, "options is nullptr");
    }

    // The class is supposed to exist only while FrontendStrategy is building a pipeline for vpux-opt.
    // Please don't cache or move it outside the usage "context".
    OptionsWrapper(const OptionsWrapper&) = delete;
    OptionsWrapper& operator=(const OptionsWrapper&) = delete;
    OptionsWrapper(OptionsWrapper&&) = delete;
    OptionsWrapper& operator=(OptionsWrapper&&) = delete;
    ~OptionsWrapper() = default;

    const OptionsType& getPipelineOptions() const {
        return *_options;
    }

    const VPU::InitCompilerOptions& getInitCompilerOptions() const {
        return *_initCompilerOptions;
    }

protected:
    const VPU::InitCompilerOptions* _initCompilerOptions;
    const OptionsType* _options;
};

//
// This class serves two purposes:
//  1. Parsing options from config and setting specific values depending on platform and compilation mode
//  2. From the PipelineStrategy pov it is simply a container for a specific type of options
//

template <class ConcreteModel, class OptionsType>
class OptionsSetup {
public:
    using value_type = OptionsType;

    explicit OptionsSetup(const intel_npu::Config& config): _config(config) {
        _options = OptionsType::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_WHEN(_options == nullptr, "Failed to parse COMPILATION_MODE_PARAMS");

        _initCompilerOptions = vpux::getInitCompilerOptions(config);
        const auto& numOfDPUGroups = _initCompilerOptions->numberOfDPUGroups;
        const auto& numOfDMAPorts = _initCompilerOptions->numberOfDMAPorts;
        VPUX_THROW_WHEN(
                numOfDPUGroups.hasValue() && numOfDMAPorts.hasValue() &&
                        numOfDMAPorts.getValue() > numOfDPUGroups.getValue(),
                "Requested configuration not supported by runtime. Number of DMA ports ({0}) larger than NCE clusters "
                "({1})",
                numOfDMAPorts.getValue(), numOfDPUGroups.getValue());

        setupOptions();
    }

    virtual ~OptionsSetup() = default;

    const OptionsType& getPipelineOptions() const {
        return *_options;
    }

    const VPU::InitCompilerOptions& getInitCompilerOptions() const {
        return *_initCompilerOptions;
    }

protected:
    void setupOptions() {
        _options->matchAndCopyOptionValuesFrom(*_initCompilerOptions);

        // this way user can setup specific option values
        // for different platforms and compilation modes
        ConcreteModel::setupOptionsImpl(*_options, _config);
    }

private:
    intel_npu::Config _config;
    std::unique_ptr<OptionsType> _options;
    std::unique_ptr<VPU::InitCompilerOptions> _initCompilerOptions;
};

//
// Below are helper classes for HW-agnostic options settings
// Developer can use them directly or implement new ones for specific platform
//

template <class ArchSpecificOptionsType>
class DefaultHWSetupBase : public OptionsSetup<DefaultHWSetupBase<ArchSpecificOptionsType>, ArchSpecificOptionsType> {
public:
    using Base = OptionsSetup<DefaultHWSetupBase<ArchSpecificOptionsType>, ArchSpecificOptionsType>;
    using Base::Base;
    friend Base;

private:
    // Note: must be static as we call ConcreteModel::setupOptionsImpl() from the ctor of base class
    static void setupOptionsImpl(ArchSpecificOptionsType& options, const intel_npu::Config& config) {
        options.enableProfiling = config.get<intel_npu::PERF_COUNT>();
        options.enableConvertAvgPoolToDWConv = false;
        options.enableHandleAsymmetricStrides = false;
        options.updateBatchCompileOptionsFromString(config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>());
    }
};

template <class ArchSpecificOptionsType>
class ShaveCodeGenSetupBase :
        public OptionsSetup<ShaveCodeGenSetupBase<ArchSpecificOptionsType>, ArchSpecificOptionsType> {
public:
    using Base = OptionsSetup<ShaveCodeGenSetupBase<ArchSpecificOptionsType>, ArchSpecificOptionsType>;
    using Base::Base;
    friend Base;

private:
    // Note: must be static as we call ConcreteModel::setupOptionsImpl() from the ctor of base class
    static void setupOptionsImpl(ArchSpecificOptionsType& options, const intel_npu::Config& config) {
        options.enableProfiling = config.get<intel_npu::PERF_COUNT>();
        options.enableConvertAvgPoolToDWConv = false;
        options.enableHandleAsymmetricStrides = false;
        options.updateBatchCompileOptionsFromString(config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>());
    }
};

template <class ArchSpecificOptionsType>
class ReferenceSwSetupBase :
        public OptionsSetup<ReferenceSwSetupBase<ArchSpecificOptionsType>, ArchSpecificOptionsType> {
public:
    using Base = OptionsSetup<ReferenceSwSetupBase<ArchSpecificOptionsType>, ArchSpecificOptionsType>;
    using Base::Base;
    friend Base;

private:
    // Note: must be static as we call ConcreteModel::setupOptionsImpl() from the ctor of base class
    static void setupOptionsImpl(ArchSpecificOptionsType& options, const intel_npu::Config& config) {
        options.enableProfiling = config.get<intel_npu::PERF_COUNT>();
        options.updateBatchCompileOptionsFromString(config.get<intel_npu::BATCH_COMPILER_MODE_SETTINGS>());
    }
};

}  // namespace vpux
