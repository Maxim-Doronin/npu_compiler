//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/pipelines/developer_config.hpp"
#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/utils/dump_intermediate_values.hpp"
#include "vpux/compiler/locverif/locations_verifier.hpp"
#include "vpux/compiler/pipelines/function_statistics_instrumentation.hpp"
#include "vpux/compiler/pipelines/options_mapper.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"
#include "vpux/compiler/utils/memory_usage_collector.hpp"
#include "vpux/utils/core/error.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
#endif

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Format.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/Timing.h>

namespace vpux {

DeveloperConfig::DeveloperConfig(Logger log): _log(log) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_NPU_CRASH_REPRODUCER_FILE", _crashReproducerFile);
    parseEnv("IE_NPU_GEN_LOCAL_REPRODUCER", _localReproducer);

    parseEnv("IE_NPU_IR_PRINTING_FILTER", _irPrintingFilter);
    parseEnv("IE_NPU_IR_PRINT_TO_FILE_TREE", _printToFileTree);
    parseEnv("IE_NPU_IR_PRINTING_LOCATION", _irPrintingLocation);
    parseEnv("IE_NPU_IR_PRINTING_ORDER", _irPrintingOrderStr);
    parseEnv("IE_NPU_PRINT_FULL_IR", _printFullIR);
    parseEnv("IE_NPU_PRINT_FULL_CONSTANT", _printFullConstant);
    parseEnv("IE_NPU_USE_SHARED_CONSTANTS", _useSharedConstants);
    parseEnv("IE_NPU_PRINT_HEX_CONSTANT", _allowPrintingHexConstant);
    parseEnv("IE_NPU_PRINT_DEBUG_INFO", _printDebugInfo);
    parseEnv("IE_NPU_PRINT_DEBUG_INFO_PRETTY_FORM", _printDebugInfoPrettyForm);
    parseEnv("IE_NPU_PRINT_SSA_PRETTY_FORM", _printSsaPrettyForm);
    parseEnv("IE_NPU_PRINT_AS_TEXTUAL_PIPELINE_FILE", _printAsTextualPipelineFilePath);
    parseEnv("IE_NPU_PRINT_DOT", _printDotOptions);
    parseEnv("IE_NPU_DUMP_INTERMEDIATE_VALUES", _dumpIntermediateValues);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    if (_printToFileTree && _irPrintingLocation.empty()) {
        _log.info("PrintTree location not specified, defaulting to .");
        _irPrintingLocation = ".";
    }

    if (!_irPrintingOrderStr.empty()) {
        auto orderString = _irPrintingOrderStr;
        std::transform(orderString.begin(), orderString.end(), orderString.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        if (orderString == "BEFORE") {
            _irPrintingOrder = IRPrintingOrder::BEFORE;
        } else if (orderString == "AFTER") {
            _irPrintingOrder = IRPrintingOrder::AFTER;
        } else if (orderString == "BEFORE_AFTER") {
            _irPrintingOrder = IRPrintingOrder::BEFORE_AFTER;
        } else {
            VPUX_THROW("Invalid IR printing order: {0}.\nValid cases are: before, after and before_after. They are not "
                       "case-sensitive.\nExample: IE_NPU_IR_PRINTING_ORDER=Before",
                       _irPrintingOrderStr);
        }
    }

    if (!_irPrintingFilter.empty()) {
        _irDumpFilter = std::make_unique<llvm::Regex>(_irPrintingFilter, llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_irDumpFilter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", _irPrintingFilter, regexErr);
        }

        if (!_printToFileTree) {
            if (_irPrintingLocation.empty()) {
                _irDumpStream = &Logger::getBaseStream();
            } else {
                std::error_code err;
                _irDumpFile = std::make_unique<llvm::raw_fd_ostream>(_irPrintingLocation, err);
                if (err) {
                    VPUX_THROW("Failed to open file '{0}' for write : {1}", _irPrintingLocation, err.message());
                }

                _irDumpStream = _irDumpFile.get();
            }
        }
    }
}

DeveloperConfig::~DeveloperConfig() {
    if (_irDumpStream != nullptr) {
        _irDumpStream->flush();
    }
}

void DeveloperConfig::setup(mlir::DefaultTimingManager& tm) const {
    if (_log.isActive(LogLevel::Info)) {
        tm.setEnabled(true);
        tm.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);
        tm.setOutput(
                mlir::createOutputStrategy(mlir::DefaultTimingManager::OutputFormat::Text, Logger::getBaseStream()));
    } else {
        tm.setEnabled(false);
    }
}

class PassLogging final : public mlir::PassInstrumentation {
public:
    explicit PassLogging(Logger log): _log(log) {
    }

    void runBeforePipeline(std::optional<mlir::OperationName> name, const PipelineParentInfo&) final {
        if (name.has_value()) {
            _log.debug("Start Pass Pipeline {0}", *name);
        }
    }

    void runAfterPipeline(std::optional<mlir::OperationName> name, const PipelineParentInfo&) final {
        if (name.has_value()) {
            _log.debug("End Pass Pipeline {0}", *name);
        }
    }

    void runBeforePass(mlir::Pass* pass, mlir::Operation* op) final {
        if (pass->getName() != "mlir::detail::OpToOpPassAdaptor") {
            _log.debug("Start Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        }
    }

    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
        if (pass->getName() != "mlir::detail::OpToOpPassAdaptor") {
            _log.debug("End Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        }
    }

    void runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) override {
        if (pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
            return;
        }

        auto module =
                mlir::isa<mlir::ModuleOp>(op) ? mlir::cast<mlir::ModuleOp>(op) : op->getParentOfType<mlir::ModuleOp>();

        const bool isWlmFailed = config::getWorkloadManagementStatus(module) == WorkloadManagementStatus::FAILED;
        if (isWlmFailed) {
            _log.warning("WLM Failed Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        } else {
            _log.error("Failed Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        }
    }

    void runBeforeAnalysis(StringRef name, mlir::TypeID, mlir::Operation* op) override {
        _log.trace("Start Analysis {0} on Operation {1}", name, op->getLoc());
    }

    void runAfterAnalysis(StringRef name, mlir::TypeID, mlir::Operation* op) override {
        _log.trace("End Analysis {0} on Operation {1}", name, op->getLoc());
    }

private:
    Logger _log;
};

void addLogging(mlir::PassManager& pm, Logger log) {
    pm.addInstrumentation(std::make_unique<PassLogging>(log));
}

void DeveloperConfig::setup(mlir::PassManager& pm, const intel_npu::Config& config, bool isSubPipeline) const {
    addLogging(pm, _log);

    // Crash reproducer
    if (!_crashReproducerFile.empty()) {
        // In case the pass manager represents a sub-pipeline (e.g. for the backend), multithreading cannot be safely
        // disabled since the context could be in a multithreading execution context
        if (_localReproducer && !isSubPipeline) {
            pm.getContext()->disableMultithreading();
        }

        pm.enableCrashReproducerGeneration(_crashReproducerFile, _localReproducer);
    }

    // IR printing
    if (_irDumpFilter != nullptr) {
        const bool printAfterOnlyOnChange = false;
        const bool printAfterOnlyOnFailure = false;

        const auto shouldPrintBeforePass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::BEFORE || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };
        const auto shouldPrintAfterPass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::AFTER || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };

        if (_printFullIR && !isSubPipeline) {
            pm.getContext()->disableMultithreading();
        }

        mlir::OpPrintingFlags flags;
        if (!_printFullConstant) {
            flags.elideLargeElementsAttrs();
            flags.elideLargeResourceString();
        }
        if (!_allowPrintingHexConstant) {
            flags.printLargeElementsAttrWithHex(-1);
        }
        if (_printDebugInfo) {
            flags.enableDebugInfo(true, _printDebugInfoPrettyForm);
        }
        if (_printSsaPrettyForm) {
            flags.printNameLocAsPrefix();
        }

        if (_printToFileTree) {
            pm.enableIRPrintingToFileTree(shouldPrintBeforePass, shouldPrintAfterPass, _printFullIR,
                                          printAfterOnlyOnChange, printAfterOnlyOnFailure, _irPrintingLocation, flags);
        } else {
            assert(_irDumpStream && "IR dump stream must be set in all cases");
            pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, _printFullIR, printAfterOnlyOnChange,
                                printAfterOnlyOnFailure, *_irDumpStream, flags);
        }
    }

    // Dot printing
    if (!_printDotOptions.empty()) {
        addDotPrinter(pm, _printDotOptions);
    }

    if (!_dumpIntermediateValues.empty()) {
        addIntermediateValueDumper(pm, _dumpIntermediateValues, _log);
    }

    // Locations verifier
    addLocationsVerifier(pm);

    const auto shouldEnableFunctionStatistics = getEnableFunctionStatisticsInstrumentation(config).value_or(false);
    if (shouldEnableFunctionStatistics) {
        _log.info("The function statistics instrumentation is enabled");
        addFunctionStatisticsInstrumentation(pm, _log);
    }

    // Memory usage instrumentation
    const auto shouldEnableMemoryCollector = getEnableMemoryUsageCollector(config).value_or(false);
    _log.info("The memory usage collector is {0}", shouldEnableMemoryCollector ? "enabled" : "disabled");
    if (shouldEnableMemoryCollector) {
        addMemoryUsageCollector(pm, _log);
    }

    // Enable pass verifiers
    const auto shouldEnableVerifiers = getEnableVerifiers(config).value_or(false);
    _log.info("Verifiers are {0}", shouldEnableVerifiers ? "enabled" : "disabled");
    pm.enableVerifier(shouldEnableVerifiers);
}

void DeveloperConfig::dump(mlir::PassManager& pm) const {
    if (!_printAsTextualPipelineFilePath.empty()) {
        std::error_code err;
        auto passesDumpFile = std::make_unique<llvm::raw_fd_ostream>(_printAsTextualPipelineFilePath, err);
        if (err) {
            VPUX_THROW("Failed to open file '{0}' for write : {1}", _printAsTextualPipelineFilePath, err.message());
        }
        pm.printAsTextualPipeline(*passesDumpFile);
    }
}

}  // namespace vpux
