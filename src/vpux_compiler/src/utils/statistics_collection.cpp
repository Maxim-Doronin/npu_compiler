//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/statistics_collection.hpp"

namespace vpux::utils {
bool OpCounter::record(mlir::Operation* op) {
    if (!_predicate(op)) {
        return false;
    }

    ++_count;
    return true;
}

void OpCounter::recordAsUnrecognized(mlir::Operation* op) {
    if (!_predicate(op) || !_handleUnrecognizedCounter) {
        return;
    }
    _unrecognizedCounters[_handleUnrecognizedCounter(op)]++;
}

void OpCounter::printStatistics(const vpux::Logger& log) const {
    if (_count == 0) {
        return;
    }
    log.info("{0} - {1} ops", _category, _count);
}

void OpCounter::printUnrecognizedStatistics(const vpux::Logger& log) const {
    for (const auto& [name, count] : _unrecognizedCounters) {
        log.info("{0} - {1} ops", name, count);
    }
}

bool AddOpRecordVisitor::visit(const Node& node) {
    if (!node.data()->record(_op)) {
        return false;
    }

    _opUsedByNestedCounter.back() = true;
    _opUsedByNestedCounter.push_back(false);
    return true;
}

void AddOpRecordVisitor::endVisit(const Node& node) {
    const bool opRecognizedByNestedCounters = _opUsedByNestedCounter.back();
    _opUsedByNestedCounter.pop_back();

    if (!opRecognizedByNestedCounters) {
        node.data()->recordAsUnrecognized(_op);
    }
}

bool PrintOpRecordVisitor::visit(const Node& node) {
    node.data()->printStatistics(_log);
    _log = _log.nest();
    return true;
}

void PrintOpRecordVisitor::endVisit(const Node& node) {
    node.data()->printUnrecognizedStatistics(_log);
    _log = _log.unnest();
}

}  // namespace vpux::utils
