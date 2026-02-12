//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/dialect_processors.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {

namespace {

constexpr int64_t MAX_SHIFT = 64;

}  // namespace

std::optional<int64_t> getIntValueFromDimOp(mlir::tensor::DimOp dimOp, const Logger& log) {
    auto dimIndex = mlir::getConstantIntValue(dimOp.getIndex());
    if (!dimIndex.has_value()) {
        log.warning("Dim index is not a constant!");
        return std::nullopt;
    }

    if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(dimOp.getSource().getType())) {
        const auto dimIndexValue = dimIndex.value();
        if (rankedType.hasStaticShape() && dimIndexValue < rankedType.getRank()) {
            return rankedType.getShape()[dimIndexValue];
        }

        if (auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(rankedType)) {
            auto bounds = boundedType.getBounds().raw();
            if (dimIndexValue < static_cast<int64_t>(bounds.size())) {
                return bounds[dimIndexValue];
            }
        }
    }

    return std::nullopt;
}

mlir::scf::YieldOp getYieldOperation(mlir::Block* block) {
    if (!block) {
        return nullptr;
    }

    if (auto terminator = block->getTerminator()) {
        if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(terminator)) {
            return yieldOp;
        }
    }

    for (auto& op : block->getOperations()) {
        if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(&op)) {
            return yieldOp;
        }
    }

    return nullptr;
}

mlir::Operation* getBlockTerminator(mlir::Block* block) {
    if (!block) {
        return nullptr;
    }
    return block->getTerminator();
}

namespace {

std::optional<int64_t> checkAndGetValueFromConstOrDimOp(mlir::Value operand, const Logger& log) {
    auto integerValue = mlir::getConstantIntValue(operand);
    if (integerValue.has_value()) {
        return integerValue;
    }

    if (auto dimOp = mlir::dyn_cast<mlir::tensor::DimOp>(operand.getDefiningOp())) {
        auto dimensionValue = getIntValueFromDimOp(dimOp, log);
        if (dimensionValue.has_value()) {
            return dimensionValue;
        }
    }
    return std::nullopt;
}

}  // namespace

void DialectProcessorRegistry::registerProcessor(std::unique_ptr<IDialectProcessor> processor) {
    _processors.push_back(std::move(processor));
    _dialectCache.clear();
}

IDialectProcessor* DialectProcessorRegistry::getProcessor(mlir::Operation* op) const {
    auto* dialect = op->getDialect();

    auto cacheIterator = _dialectCache.find(dialect);
    if (cacheIterator != _dialectCache.end()) {
        return cacheIterator->second;
    }

    for (const auto& processor : _processors) {
        if (processor->canProcess(op)) {
            _dialectCache[dialect] = processor.get();
            return processor.get();
        }
    }

    _dialectCache[dialect] = nullptr;
    return nullptr;
}

bool DialectProcessorRegistry::hasProcessor(mlir::Operation* op) const {
    return getProcessor(op) != nullptr;
}

std::unique_ptr<DialectProcessorRegistry> DialectProcessorRegistry::createDefault() {
    auto registry = std::make_unique<DialectProcessorRegistry>();
    registry->registerProcessor(
            std::make_unique<AffineDialectProcessor>(Logger::global().nest("affine-dialect-processor")));
    registry->registerProcessor(
            std::make_unique<ArithmeticDialectProcessor>(Logger::global().nest("arith-dialect-processor")));
    registry->registerProcessor(std::make_unique<SCFDialectProcessor>(Logger::global().nest("scf-dialect-processor")));
    return registry;
}

bool AffineDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::affine::AffineDialect>(op->getDialect());
}

bool AffineDialectProcessor::processOperation(mlir::Operation* op, llvm::DenseMap<mlir::Value, int64_t>& valueMap,
                                              BlockProcessor blockProcessor) const {
    (void)blockProcessor;
    auto [affineMap, mapOperands] = getAffineMapAndOperands(op);
    SmallVector<mlir::Attribute> operandAttrs;
    bool success = true;
    for (auto operand : mapOperands) {
        int64_t operandValue = 0;
        auto valueIterator = valueMap.find(operand);
        if (valueIterator != valueMap.end()) {
            operandValue = valueIterator->second;
        } else {
            auto value = checkAndGetValueFromConstOrDimOp(operand, _log);
            if (value.has_value()) {
                operandValue = value.value();
            } else {
                _log.warning("Missing operand value for affine operation: {0}", op->getName());
                success = false;
                break;
            }
        }
        operandAttrs.push_back(mlir::IntegerAttr::get(operand.getType(), operandValue));
    }

    if (!success) {
        return false;
    }

    SmallVector<mlir::Attribute> resultsAttrs;
    if (affineMap.constantFold(operandAttrs, resultsAttrs).failed()) {
        return false;
    }

    SmallVector<int64_t> results;
    for (auto attr : resultsAttrs) {
        results.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }

    int64_t result = getAffineResult(op, results);
    valueMap[op->getResult(0)] = result;
    return true;
}

std::pair<mlir::AffineMap, mlir::ValueRange> AffineDialectProcessor::getAffineMapAndOperands(
        mlir::Operation* op) const {
    if (auto affineOp = mlir::dyn_cast<mlir::affine::AffineMinOp>(op)) {
        return {affineOp.getAffineMap(), affineOp.getOperands()};
    }
    if (auto affineOp = mlir::dyn_cast<mlir::affine::AffineMaxOp>(op)) {
        return {affineOp.getAffineMap(), affineOp.getOperands()};
    }
    if (auto applyOp = mlir::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
        return {applyOp.getAffineMap(), applyOp.getOperands()};
    }

    VPUX_THROW("Unsupported affine operation type: {0}", op->getName());
}

int64_t AffineDialectProcessor::getAffineResult(mlir::Operation* op, llvm::ArrayRef<int64_t> results) const {
    if (results.empty()) {
        VPUX_THROW("Empty results array for operation: {0}", op->getName());
    }

    if (mlir::isa<mlir::affine::AffineMinOp>(op)) {
        return *llvm::min_element(results);
    }
    if (mlir::isa<mlir::affine::AffineMaxOp>(op)) {
        return *llvm::max_element(results);
    }
    if (mlir::isa<mlir::affine::AffineApplyOp>(op)) {
        return results[0];
    }

    VPUX_THROW("Unsupported affine operation type: {0}", op->getName());
}

bool ArithmeticDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::arith::ArithDialect>(op->getDialect());
}

bool ArithmeticDialectProcessor::processOperation(mlir::Operation* op, llvm::DenseMap<mlir::Value, int64_t>& valueMap,
                                                  BlockProcessor blockProcessor) const {
    (void)blockProcessor;
    if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValueAttr())) {
            valueMap[op->getResult(0)] = intAttr.getInt();
            return true;
        }
        return false;
    }

    for (auto operand : op->getOperands()) {
        if (!valueMap.contains(operand)) {
            auto intValue = checkAndGetValueFromConstOrDimOp(operand, _log);
            if (intValue.has_value()) {
                valueMap[operand] = intValue.value();
                continue;
            }

            _log.warning("Missing operand value for arith operation: {0}", op->getName());
        }
    }

    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<mlir::arith::MinUIOp>([&](auto minOp) {
                auto lhs = checked_cast<uint64_t>(valueMap[minOp.getLhs()]);
                auto rhs = checked_cast<uint64_t>(valueMap[minOp.getRhs()]);
                auto resultValue = std::min(lhs, rhs);
                valueMap[op->getResult(0)] = checked_cast<int64_t>(resultValue);
                return true;
            })
            .Case<mlir::arith::AddIOp>([&](auto addOp) {
                auto resultValue = valueMap[addOp.getLhs()] + valueMap[addOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::SubIOp>([&](auto subOp) {
                auto resultValue = valueMap[subOp.getLhs()] - valueMap[subOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::MulIOp>([&](auto mulOp) {
                auto resultValue = valueMap[mulOp.getLhs()] * valueMap[mulOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::DivSIOp>([&](auto divOp) {
                if (valueMap[divOp.getRhs()] == 0) {
                    _log.warning("Division by zero in DivSIOp");
                    return false;
                }
                auto resultValue = valueMap[divOp.getLhs()] / valueMap[divOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::DivUIOp>([&](auto divOp) {
                if (valueMap[divOp.getRhs()] == 0) {
                    _log.warning("Division by zero in DivUIOp");
                    return false;
                }
                auto resultValue = valueMap[divOp.getLhs()] / valueMap[divOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::CmpIOp>([&](auto cmpOp) {
                auto lhs = valueMap[cmpOp.getLhs()];
                auto rhs = valueMap[cmpOp.getRhs()];
                auto predicate = cmpOp.getPredicate();
                bool result = false;
                switch (predicate) {
                case mlir::arith::CmpIPredicate::eq:
                    result = (lhs == rhs);
                    break;
                case mlir::arith::CmpIPredicate::ne:
                    result = (lhs != rhs);
                    break;
                case mlir::arith::CmpIPredicate::slt:
                    result = (lhs < rhs);
                    break;
                case mlir::arith::CmpIPredicate::sle:
                    result = (lhs <= rhs);
                    break;
                case mlir::arith::CmpIPredicate::sgt:
                    result = (lhs > rhs);
                    break;
                case mlir::arith::CmpIPredicate::sge:
                    result = (lhs >= rhs);
                    break;
                case mlir::arith::CmpIPredicate::ult:
                    result = (static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs));
                    break;
                case mlir::arith::CmpIPredicate::ule:
                    result = (static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs));
                    break;
                case mlir::arith::CmpIPredicate::ugt:
                    result = (static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs));
                    break;
                case mlir::arith::CmpIPredicate::uge:
                    result = (static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs));
                    break;
                }
                valueMap[op->getResult(0)] = result ? 1 : 0;
                return true;
            })
            .Case<mlir::arith::RemSIOp>([&](auto remOp) {
                if (valueMap[remOp.getRhs()] == 0) {
                    _log.warning("Division by zero in RemSIOp");
                    return false;
                }
                auto resultValue = valueMap[remOp.getLhs()] % valueMap[remOp.getRhs()];
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::RemUIOp>([&](auto remOp) {
                if (valueMap[remOp.getRhs()] == 0) {
                    _log.warning("Division by zero in RemUIOp");
                    return false;
                }
                auto lhs = static_cast<uint64_t>(valueMap[remOp.getLhs()]);
                auto rhs = static_cast<uint64_t>(valueMap[remOp.getRhs()]);
                auto resultValue = static_cast<int64_t>(lhs % rhs);
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::SelectOp>([&](auto selectOp) {
                auto condition = valueMap[selectOp.getCondition()];
                auto trueValue = valueMap[selectOp.getTrueValue()];
                auto falseValue = valueMap[selectOp.getFalseValue()];
                auto resultValue = (condition != 0) ? trueValue : falseValue;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::ShLIOp>([&](auto shiftOp) {
                auto lhs = valueMap[shiftOp.getLhs()];
                auto rhs = valueMap[shiftOp.getRhs()];
                if (rhs < 0 || rhs >= MAX_SHIFT) {
                    _log.warning("Invalid shift amount in ShLIOp: {0}", rhs);
                    return false;
                }
                auto resultValue = lhs << rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::ShRSIOp>([&](auto shiftOp) {
                auto lhs = valueMap[shiftOp.getLhs()];
                auto rhs = valueMap[shiftOp.getRhs()];
                if (rhs < 0 || rhs >= MAX_SHIFT) {
                    _log.warning("Invalid shift amount in ShRSIOp: {0}", rhs);
                    return false;
                }
                auto resultValue = lhs >> rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::ShRUIOp>([&](auto shiftOp) {
                auto lhs = valueMap[shiftOp.getLhs()];
                auto rhs = valueMap[shiftOp.getRhs()];
                if (rhs < 0 || rhs >= MAX_SHIFT) {
                    _log.warning("Invalid shift amount in ShRUIOp: {0}", rhs);
                    return false;
                }
                auto unsignedLhs = static_cast<uint64_t>(lhs);
                auto resultValue = static_cast<int64_t>(unsignedLhs >> rhs);
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::OrIOp>([&](auto orOp) {
                auto lhs = valueMap[orOp.getLhs()];
                auto rhs = valueMap[orOp.getRhs()];
                auto resultValue = lhs | rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Case<mlir::arith::AndIOp>([&](auto andOp) {
                auto lhs = valueMap[andOp.getLhs()];
                auto rhs = valueMap[andOp.getRhs()];
                auto resultValue = lhs & rhs;
                valueMap[op->getResult(0)] = resultValue;
                return true;
            })
            .Default([&](mlir::Operation*) {
                _log.trace("Unsupported arith operation: {0}", op->getName());
                return false;
            });
}

bool SCFDialectProcessor::canProcess(mlir::Operation* op) const {
    return mlir::isa<mlir::scf::SCFDialect>(op->getDialect());
}

bool SCFDialectProcessor::processOperation(mlir::Operation* op, llvm::DenseMap<mlir::Value, int64_t>& valueMap,
                                           BlockProcessor blockProcessor) const {
    return llvm::TypeSwitch<mlir::Operation*, bool>(op)
            .Case<mlir::scf::IfOp>([&](auto ifOp) {
                return processIfOp(ifOp, valueMap, blockProcessor);
            })
            .Case<mlir::scf::YieldOp>([&](auto yieldOp) {
                for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
                    if (idx < op->getNumResults()) {
                        auto it = valueMap.find(operand);
                        if (it != valueMap.end()) {
                            valueMap[op->getResult(idx)] = it->second;
                        } else {
                            auto result = checkAndGetValueFromConstOrDimOp(operand, _log);
                            if (result.has_value()) {
                                valueMap[op->getResult(idx)] = result.value();
                            } else {
                                _log.warning("Failed to get integer value for yield operand: {0}", operand);
                                return false;
                            }
                        }
                    }
                }
                return true;
            })
            .Default([&](mlir::Operation*) {
                _log.trace("Unsupported scf operation: {0}", op->getName());
                return false;
            });
}

bool SCFDialectProcessor::processIfOp(mlir::scf::IfOp ifOp, llvm::DenseMap<mlir::Value, int64_t>& valueMap,
                                      const BlockProcessor& blockProcessor) const {
    auto condition = ifOp.getCondition();
    if (!valueMap.contains(condition)) {
        _log.warning("Condition value not found in value map for scf.if operation {0}", condition);
        return false;
    }

    bool condResult = valueMap[condition] != 0;
    mlir::Block* activeBlock = condResult ? ifOp.thenBlock() : ifOp.elseBlock();

    if (!activeBlock) {
        return false;
    }

    blockProcessor(activeBlock, valueMap);

    auto terminator = activeBlock->getTerminator();
    for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
        auto valueIterator = valueMap.find(terminator->getOperand(idx));
        if (valueIterator != valueMap.end()) {
            valueMap[result] = valueIterator->second;
        } else {
            auto termResult = checkAndGetValueFromConstOrDimOp(terminator->getOperand(idx), _log);
            if (termResult.has_value()) {
                valueMap[result] = termResult.value();
            } else {
                _log.warning("Failed to get integer value for if terminator operand: {0}", terminator->getOperand(idx));
                return false;
            }
        }
    }

    return true;
}

}  // namespace vpux::VPU
