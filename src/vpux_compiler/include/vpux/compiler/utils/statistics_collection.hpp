//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <functional>
#include <map>
#include <string>

namespace mlir {
class Operation;
}

namespace vpux::utils {

/** @brief Represents a generic operation counter.

    This class is a simple virtual base class that provides default
    implementation for counting and printing statistics of operations in a
    certain way.
 */
struct OpCounter {
    //! @brief A predicate that specifies whether the operation can be recorded.
    using IsOperationSuitable = std::function<bool(mlir::Operation*)>;
    //! @brief A handler for unrecognized operation that returns a string
    //! (usually, a name) associated with the operation.
    using HandleUnrecognizedCounter = std::function<std::string(mlir::Operation*)>;

    OpCounter(const std::string& category, IsOperationSuitable pred, HandleUnrecognizedCounter handleUnrecognized = {})
            : _category(category),
              _predicate(std::move(pred)),
              _handleUnrecognizedCounter(std::move(handleUnrecognized)) {
    }

    //! @brief Counts the provided operation. Returns whether the operation can
    //! be "recorded".
    virtual bool record(mlir::Operation*);

    //! @brief Prints the recorded operation statistics.
    virtual void printStatistics(const vpux::Logger& log) const;

    //! @brief Records the provided operation as "unrecognized". Note that if
    //! the operation can not be recorded in general, it won't be recorded as
    //! unrecognized.
    virtual void recordAsUnrecognized(mlir::Operation*);

    //! @brief Prints the recorded "unrecognized" operation statistics.
    virtual void printUnrecognizedStatistics(const vpux::Logger& log) const;

    virtual ~OpCounter() = default;

protected:
    std::string _category;
    IsOperationSuitable _predicate;
    HandleUnrecognizedCounter _handleUnrecognizedCounter;  // Note: optional callback

    size_t _count{0};
    std::map<std::string, size_t> _unrecognizedCounters{};
};

/** @brief Represents a hierarchy of operation counters in the form of a tree.

    This is a hierarchy of operation counters that can be constructed to group
    multiple counters under the universal one. For example:
    ```
    VPUIP               <--- top-level counter
        |- VPUIP.NNDMA  <--- NNDMA-specific (nested) counter
        |- VPUIP.NCE    <--- NCE-specific (nested) countter
    where
    size(VPUIP) == size(VPUIP.NNDMA) + size(VPUIP.NCE)
    ```
 */
using OpCounterTree = AbstractTree<std::unique_ptr<OpCounter>>;

/** @brief Counter hierarchy visitor that implements generic "recording".

    Traverses the counter hierarchy, calling "record" for every counter.
    Additionally, when there are nested counters, keeps track of whether the
    operation is recognized by any of the nested counters and, if not, records
    it as "unrecognized" at the suitable level.
 */
struct AddOpRecordVisitor final : OpCounterTree::Visitor {
    AddOpRecordVisitor(mlir::Operation* op): _op(op), _opUsedByNestedCounter({false}) {
    }

    bool visit(const Node& node) override;
    void endVisit(const Node& node) override;

private:
    mlir::Operation* _op;
    std::vector<bool> _opUsedByNestedCounter;
};

/** @brief Counter hierarchy visitor that implements generic printing.

    Traverses the counter hierarchy and prints statistics for every counter in a
    particular fashion.
 */
struct PrintOpRecordVisitor final : public OpCounterTree::Visitor {
    PrintOpRecordVisitor(const Logger& log): _log(log) {
    }

    bool visit(const Node& node) override;
    void endVisit(const Node& node) override;

private:
    Logger _log;
};

}  // namespace vpux::utils
