//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Visitors.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::Core {
#define GEN_PASS_DECL_UNPACKNESTEDMODULES
#define GEN_PASS_DEF_UNPACKNESTEDMODULES
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

mlir::func::FuncOp getFirstFunctionDirectlyInModule(mlir::ModuleOp moduleOp) {
    auto allFuncs = moduleOp.getOps<mlir::func::FuncOp>();
    assert(!allFuncs.empty() && "Module must have at least one operation");
    auto firstFunc = *allFuncs.begin();
    assert(getModuleOp(firstFunc) == moduleOp && "Module operations are assumed to be non-nested");
    return firstFunc;
}

class UnpackNestedModulesPass final : public Core::impl::UnpackNestedModulesBase<UnpackNestedModulesPass> {
    /// Returns all "top-level" nested modules within a specified module.
    /// Top-level nested modules are direct children of the module.
    SmallVector<mlir::ModuleOp> collectTopLevelNestedModules(mlir::ModuleOp mainModule);

    /// Moves any discovered nested functions into the specified module.
    void moveNestedFunctionsIntoMainModule(ArrayRef<mlir::ModuleOp> nestedModules, mlir::ModuleOp mainModule);

    /// Converts any calls to nested functions into direct calls to non-nested
    /// functions. The precondition of this procedure is that the nested
    /// functions are no longer nested (i.e. reachable).
    void flattenCallSites(mlir::ModuleOp mainModule);

    /// Erases any nested modules.
    void eraseNestedModules(ArrayRef<mlir::ModuleOp> nestedModules);

public:
    explicit UnpackNestedModulesPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    void safeRunOnModule() final;
};

SmallVector<mlir::ModuleOp> UnpackNestedModulesPass::collectTopLevelNestedModules(mlir::ModuleOp mainModule) {
    SmallVector<mlir::ModuleOp> topLevelNested;

    // Note: assumed to be "fast" since we only walk modules.
    mainModule.walk([&](mlir::ModuleOp nestedModule) {
        if (nestedModule == mainModule) {  // Note: walk visits "self" as well
            return mlir::WalkResult::advance();
        }

        const bool directChildOfMainModule = (nestedModule->getParentOfType<mlir::ModuleOp>() == mainModule);
        if (directChildOfMainModule) {
            _log.trace("Found top-level nested module '{0}' inside '{1}'", nestedModule.getSymName(),
                       mainModule.getSymName());
            topLevelNested.push_back(nestedModule);
        }
        return mlir::WalkResult::skip();  // ignore nested ops, regions, etc.
    });

    return topLevelNested;
}

void UnpackNestedModulesPass::moveNestedFunctionsIntoMainModule(ArrayRef<mlir::ModuleOp> nestedModules,
                                                                mlir::ModuleOp mainModule) {
    auto referenceFuncOp = getFirstFunctionDirectlyInModule(mainModule);
    const auto moveNestedFunctionOutside = [&](mlir::func::FuncOp funcOp) {
        auto nestedModule = getModuleOp(funcOp);
        assert(nestedModule != mainModule);

        _log.trace("Moving op '{0}' from the nested module '{1}' to '{2}'", funcOp.getSymName(),
                   nestedModule.getSymName(), mainModule.getSymName());
        // Note: move nested function to a place relative to the "reference"
        // function (some function that is guaranteed to be in the main module).
        funcOp->moveBefore(referenceFuncOp);
        funcOp.setPrivate();
        return mlir::WalkResult::skip();
    };

    for (auto nested : nestedModules) {
        // Note: walk here is "recursive". That is, given the below structure:
        // ```mlir
        // module @Nested {
        //  module @Nested2 {
        //      func.func private @bar(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> { ... }
        //  }
        //
        //  func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf16> { ... }
        // }
        // ```
        // *both* @foo and @bar would be visited.
        nested.walk(moveNestedFunctionOutside);
    }
}

void UnpackNestedModulesPass::flattenCallSites(mlir::ModuleOp mainModule) {
    // Note: so far the only call operation that supports nested calls is
    // Core::NestedCallOp.
    mainModule.walk([&](Core::NestedCallOp callOp) {
        _log.trace("Unnestifying the call-site '{0}'", callOp);
        // Note: this walk unconditionally replaces *all* Core.NestedCall ops with
        // func.call alternatives.
        auto calleeSymbol = callOp.getCallee();
        auto leafRef = calleeSymbol.getLeafReference();
        auto flatCalleeSymbol = mlir::FlatSymbolRefAttr::get(leafRef);

        // Note: MLIR's own call ops support flat symbol references.
        mlir::OpBuilder builder(callOp);
        auto newCallOp = builder.create<mlir::func::CallOp>(callOp.getLoc(), flatCalleeSymbol, callOp->getResultTypes(),
                                                            callOp.getOperands());
        callOp->replaceAllUsesWith(newCallOp->getResults());
        callOp->erase();
    });
}

void UnpackNestedModulesPass::eraseNestedModules(ArrayRef<mlir::ModuleOp> nestedModules) {
    for (auto nested : nestedModules) {
        _log.trace("Erasing nested module '{0}'", nested.getSymName());
        nested.erase();
    }
}

// Note: `safeRunOnModule()` is assumed to run *only* for the "main" (the most
// top-level) module. Thus, this pass should run exactly once and unnestify
// "everything", leaving just the "main" module.
void UnpackNestedModulesPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    const auto nestedModules = collectTopLevelNestedModules(moduleOp);
    moveNestedFunctionsIntoMainModule(nestedModules, moduleOp);
    flattenCallSites(moduleOp);
    eraseNestedModules(nestedModules);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createUnpackNestedModulesPass(const Logger& log) {
    return std::make_unique<UnpackNestedModulesPass>(log);
}
