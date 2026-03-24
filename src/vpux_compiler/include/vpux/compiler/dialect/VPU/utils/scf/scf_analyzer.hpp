//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <memory>

namespace vpux::VPU {

// Base interface for all SCF analyzers
class IScfAnalyzer {
public:
    virtual ~IScfAnalyzer() = default;

    // Run analysis with the given context
    virtual void analyze(AnalysisContext& context) = 0;

    // Print analysis results with indentation
    virtual void printResults(std::ostream& stream, size_t indentLevel = 0) const = 0;

    // Query if analyzer has results to display
    virtual bool hasResults() const = 0;

    // Get analyzer name for identification
    virtual StringRef getName() const = 0;
};

class OpDebugInfo {
public:
    OpDebugInfo(mlir::Operation* op, const Logger& log = Logger::global().nest("op-debug-info")): _op(op), _log(log) {
    }

    virtual void print(std::ostream& stream, size_t indentLevel = 0, bool skipOffsets = true) = 0;
    virtual bool equals(const OpDebugInfo& other) const = 0;
    virtual void evaluate(ValueRangeMap& valueMap) = 0;
    virtual void setAttribute() = 0;
    virtual int64_t computeHash() const = 0;
    virtual ~OpDebugInfo() = default;

protected:
    void printDimensionData(std::ostream& stream, size_t indentLevel, const char* header,
                            ArrayRef<SmallVector<int64_t>> data);

    mlir::Operation* _op;
    Logger _log;
    OpChainAnalysis _opAnalyzer;
};

class OffsetSizeStrideOpDebugInfo final : public OpDebugInfo {
public:
    OffsetSizeStrideOpDebugInfo(mlir::Operation* op)
            : OpDebugInfo(op, Logger::global().nest("offset-size-stride-op-debug-info")) {
    }

    void evaluate(ValueRangeMap& valueMap) final {
        if (valueMap.empty()) {
            _log.warning("Value map is empty during OffsetSizeStrideOpDebugInfo construction");
            return;
        }
        evaluateDynamicDims(valueMap);
    }

    void print(std::ostream& stream, size_t indentLevel = 0, bool skipOffsets = true) final;
    bool equals(const OpDebugInfo& other) const final;
    int64_t computeHash() const final;
    void setAttribute() final;

    const SmallVector<SmallVector<int64_t>>& getEvaluatedSizes() const {
        return _evaluatedSizes;
    }

    const SmallVector<SmallVector<int64_t>>& getEvaluatedOffsets() const {
        return _evaluatedOffsets;
    }

private:
    void evaluateDynamicDims(ValueRangeMap& valueMap);
    std::tuple<SmallVector<mlir::OpFoldResult>, SmallVector<mlir::OpFoldResult>> getOffsetsAndSizes();

    SmallVector<SmallVector<int64_t>> _evaluatedSizes;
    SmallVector<SmallVector<int64_t>> _evaluatedOffsets;
};

class PadOpDebugInfo final : public OpDebugInfo {
public:
    PadOpDebugInfo(mlir::Operation* op): OpDebugInfo(op, Logger::global().nest("pad-op-debug-info")) {
    }

    void print(std::ostream& stream, size_t indentLevel = 0, bool skipOffsets = true) final;
    bool equals(const OpDebugInfo& other) const final;
    void evaluate(ValueRangeMap& valueMap) final;
    int64_t computeHash() const final;
    void setAttribute() final;

    const SmallVector<SmallVector<int64_t>>& getEvaluatedLowPads() const {
        return _evaluatedLowPads;
    }

    const SmallVector<SmallVector<int64_t>>& getEvaluatedHighPads() const {
        return _evaluatedHighPads;
    }

private:
    SmallVector<SmallVector<int64_t>> _evaluatedLowPads;
    SmallVector<SmallVector<int64_t>> _evaluatedHighPads;
};

class AffineDebugInfo final : public OpDebugInfo {
public:
    AffineDebugInfo(mlir::Operation* op): OpDebugInfo(op, Logger::global().nest("affine-op-debug-info")) {
    }

    void evaluate(ValueRangeMap& valueMap) final;
    void print(std::ostream& stream, size_t indentLevel = 0, bool skipOffsets = true) final;
    bool equals(const OpDebugInfo& other) const final;
    int64_t computeHash() const final;
    void setAttribute() final;

private:
    SmallVector<int64_t> _evaluatedValues;
};

// Base class providing common analyzer functionality
class ScfAnalyzerBase : public IScfAnalyzer {
private:
    // Helper method to perform deep copy of OpInfoMap
    void deepCopyOpInfoMap(const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& sourceMap) {
        for (const auto& [op, opInfo] : sourceMap) {
            if (auto sliceInfo = std::dynamic_pointer_cast<OffsetSizeStrideOpDebugInfo>(opInfo)) {
                _opInfoMap[op] = std::make_shared<OffsetSizeStrideOpDebugInfo>(*sliceInfo);
            } else if (auto padInfo = std::dynamic_pointer_cast<PadOpDebugInfo>(opInfo)) {
                _opInfoMap[op] = std::make_shared<PadOpDebugInfo>(*padInfo);
            } else if (auto affineInfo = std::dynamic_pointer_cast<AffineDebugInfo>(opInfo)) {
                _opInfoMap[op] = std::make_shared<AffineDebugInfo>(*affineInfo);
            }
        }
    }

protected:
    SmallVector<mlir::Operation*> _operations;
    llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>> _opInfoMap;
    vpux::Logger _log;
    StringLiteral _analyzerName;

    explicit ScfAnalyzerBase(StringLiteral analyzerName, const Logger& log = Logger::global())
            : _log(log.nest(analyzerName)), _analyzerName(analyzerName) {
    }

    ScfAnalyzerBase(const ScfAnalyzerBase& other)
            : _operations(other._operations), _log(other._log), _analyzerName(other._analyzerName) {
        deepCopyOpInfoMap(other._opInfoMap);
    }

    ScfAnalyzerBase& operator=(const ScfAnalyzerBase& other) {
        if (this != &other) {
            _operations = other._operations;
            _analyzerName = other._analyzerName;
            _log = other._log;

            _opInfoMap.clear();
            deepCopyOpInfoMap(other._opInfoMap);
        }
        return *this;
    }

    void buildOpInfoMap(ArrayRef<mlir::Operation*> operations) {
        for (auto op : operations) {
            mlir::TypeSwitch<mlir::Operation*>(op)
                    .Case<mlir::OffsetSizeAndStrideOpInterface>([&](mlir::OffsetSizeAndStrideOpInterface) {
                        _opInfoMap.try_emplace(op, std::make_shared<OffsetSizeStrideOpDebugInfo>(op));
                    })
                    .Case<mlir::tensor::PadOp>([&](mlir::tensor::PadOp) {
                        _opInfoMap.try_emplace(op, std::make_shared<PadOpDebugInfo>(op));
                    })
                    .Default([&](mlir::Operation*) {
                        if (mlir::isa<mlir::affine::AffineDialect>(op->getDialect())) {
                            _opInfoMap.try_emplace(op, std::make_shared<AffineDebugInfo>(op));
                        } else {
                            _log.trace("Unsupported operation for static analysis: {0}", op->getName());
                        }
                    });
        }
    }

    std::string getIndentString(size_t indentLevel) const {
        return std::string(indentLevel, ' ');
    }

public:
    ~ScfAnalyzerBase() override {
    }

    bool hasResults() const override {
        return !_operations.empty() && !_opInfoMap.empty();
    }
};

class ScfAnalysisInfo : public ScfAnalyzerBase {
public:
    explicit ScfAnalysisInfo(ArrayRef<mlir::Operation*> operations = {}): ScfAnalyzerBase("scf-analysis-info") {
        if (!operations.empty()) {
            _operations.assign(operations.begin(), operations.end());
        }
    }

    void analyze(AnalysisContext& context) override {
        _log.trace("Starting unified SCF analysis");
        SmallVector<InputRange> inputRanges = context.getInputRanges();

        for (auto op : _operations) {
            analyzeOperation(op, inputRanges);
        }
    }

    // Legacy per-operation analysis method (kept for backward compatibility)
    void analyzeOperation(mlir::Operation* op, SmallVector<InputRange>& iterationSpace) {
        mlir::TypeSwitch<mlir::Operation*>(op)
                .Case<mlir::OffsetSizeAndStrideOpInterface>([&](mlir::OffsetSizeAndStrideOpInterface) {
                    auto opInfo = std::make_shared<OffsetSizeStrideOpDebugInfo>(op);
                    _opInfoMap.try_emplace(op, std::move(opInfo));
                })
                .Case<mlir::tensor::PadOp>([&](mlir::tensor::PadOp) {
                    _opInfoMap.try_emplace(op, std::make_shared<PadOpDebugInfo>(op));
                })
                .Default([&](mlir::Operation*) {
                    if (mlir::isa<mlir::affine::AffineDialect>(op->getDialect())) {
                        _opInfoMap.try_emplace(op, std::make_shared<AffineDebugInfo>(op));
                    } else {
                        _log.trace("Unsupported operation for static analysis: {0}", op->getName());
                    }
                });

        ValueRangeMap valueMap;
        for (auto inputRange : iterationSpace) {
            valueMap[inputRange.arg] = inputRange.values;
        }

        auto it = _opInfoMap.find(op);
        if (it != _opInfoMap.end()) {
            it->second->evaluate(valueMap);
        }
    }

    void print(std::ostream& stream, size_t indentLevel = 0) const {
        for (const auto& [op, info] : _opInfoMap) {
            stream << std::string(indentLevel + 1, ' ') << "Operation: " << op->getName().getStringRef().str();

            if (op->hasAttr(ANALYZE_ATTR)) {
                stream << " [ " << op->getAttrOfType<mlir::StringAttr>(ANALYZE_ATTR).getValue().str() << " ]";
            }

            stream << "\n";

            info->print(stream, indentLevel + 1);
            stream << "\n";
        }
    }

    void printResults(std::ostream& stream, size_t indentLevel = 0) const override {
        print(stream, indentLevel);
    }

    StringRef getName() const override {
        return "ScfAnalysisInfo";
    }

    SmallVector<mlir::scf::ForOp> getParentForOps(mlir::Operation* op) const;
    SmallVector<InputRange> getIterationSpace(ArrayRef<mlir::scf::ForOp> forOpChain) const;

    void setAttribute() const {
        for (const auto& [op, opInfo] : _opInfoMap) {
            opInfo->setAttribute();
        }
    }

private:
    OpChainAnalysis _opAnalyzer;
};

/**
 * @brief Block-level analyzer for detecting stable state intervals in SCF loops
 *
 * This analyzer identifies and coalesces regions of iteration space where operations
 * maintain consistent behavior (stable states). The analysis proceeds in three phases:
 *
 * Phase 1 - Sampling and Unit Region Construction:
 *   - Samples the entire iteration space point-by-point
 *   - Evaluates operation states at each sampled point
 *   - Creates unit regions (single-point regions) for each sampled coordinate
 *
 * Phase 2 - State Registration and Deduplication:
 *   - Computes hash values for each operation state
 *   - Registers unique stable states or matches against existing states
 *   - Assigns state IDs to classify behavioral equivalence
 *
 * Phase 3 - Region Coalescing via Connectivity Analysis:
 *   - Identifies adjacent unit regions sharing the same state (connected in n-D space)
 *   - Coalesces connected components into larger hyperrectangular regions
 *   - Optimizes representation by merging regions that share edges/faces in n-dimensional space
 *
 * The result is a compact representation of state intervals that can be used for
 * optimization, visualization, and understanding loop behavior patterns.
 */
class ScfBlockAnalyzer : public ScfAnalyzerBase {
public:
    // Represents a stable state with unique ID
    struct StableState {
        size_t stateId;
        int64_t hashValue;
        llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>> opStates;

        bool equals(const StableState& other) const;
    };

    // Represents an n-dimensional interval region with a stable state
    struct StateRegion {
        size_t stateId;
        // Bounds for each dimension: [start, end] inclusive
        SmallVector<std::pair<int64_t, int64_t>> bounds;
        // Actual iteration points in this region
        SmallVector<SmallVector<int64_t>> points;
        // Stride for each dimension (0 if single value)
        SmallVector<int64_t> strides;

        // Helper to convert to string for debugging (point list format)
        std::string toString() const;

        // Convert to constraint representation
        std::string toConstraintString() const;
    };

    explicit ScfBlockAnalyzer(ArrayRef<mlir::Operation*> opsToBeAnalyzed = {}): ScfAnalyzerBase("scf-block-analyzer") {
        if (!opsToBeAnalyzed.empty()) {
            _operations.assign(opsToBeAnalyzed.begin(), opsToBeAnalyzed.end());
            buildOpInfoMap(opsToBeAnalyzed);
        }
    }

    void analyze(AnalysisContext& context) override {
        detectStableStateRegions(context);
    }

    void detectStableStateRegions(AnalysisContext& analysisContext);

    ArrayRef<StableState> getUniqueStates() const {
        return _uniqueStates;
    }

    size_t getUniqueStateCount() const {
        return _uniqueStates.size();
    }

    void printResults(std::ostream& stream, size_t indentLevel = 0) const override;
    void printState(size_t stateId, std::ostream& stream, size_t indentLevel = 0) const;
    void printRegion(const StateRegion& region, std::ostream& stream, size_t indentLevel = 0) const;

    bool hasResults() const override {
        return !_uniqueStates.empty();
    }

    StringRef getName() const override {
        return "ScfBlockAnalyzer";
    }

private:
    void sampleIterationSpace(AnalysisContext& analysisContext,
                              SmallVector<std::pair<SmallVector<int64_t>, size_t>>& pointToState);
    size_t registerState(const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& opStates);
    void coalesceRegions(ArrayRef<std::pair<SmallVector<int64_t>, size_t>> pointToState, ArrayRef<int64_t> dimSizes,
                         AnalysisContext& analysisContext);
    llvm::DenseMap<size_t, SmallVector<size_t>> buildConnectivityGraph(
            ArrayRef<std::pair<SmallVector<int64_t>, size_t>> pointToState, AnalysisContext& analysisContext,
            SmallVector<size_t>& parent);
    void createRegionsFromComponents(const llvm::DenseMap<size_t, SmallVector<size_t>>& components,
                                     ArrayRef<std::pair<SmallVector<int64_t>, size_t>> pointToState, size_t numDims,
                                     AnalysisContext& analysisContext);
    int64_t computeStateHash(const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& opStates) const;
    bool compareStates(const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& state1,
                       const llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& state2) const;
    void evaluateAtPoint(ValueRangeMap& valueMap,
                         llvm::MapVector<mlir::Operation*, std::shared_ptr<OpDebugInfo>>& states);

    // Stable state registry
    SmallVector<StableState> _uniqueStates;
    llvm::DenseMap<size_t, size_t> _hashToStateId;  // hash -> index in _uniqueStates

    // Regions organized by state
    SmallVector<StateRegion> _allRegions;
    llvm::DenseMap<size_t, SmallVector<size_t>> _stateToRegions;  // stateId -> indices in _allRegions
};

}  // namespace vpux::VPU
