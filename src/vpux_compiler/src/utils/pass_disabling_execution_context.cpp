//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/pass_disabling_execution_context.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

namespace {
std::shared_ptr<llvm::Regex> makeRegex(llvm::StringRef pattern) {
    if (pattern.empty()) {
        return nullptr;
    }

    std::string error;
    auto regex = std::make_shared<llvm::Regex>(pattern);
    VPUX_THROW_UNLESS(regex->isValid(error), "Invalid regular expression '{0}' : {1}", pattern, error);
    return regex;
}
}  // namespace

vpux::PassDisablingExecutionContext::PassDisablingExecutionContext(llvm::StringRef disabledPasses)
        : _disabledPasses(makeRegex(disabledPasses)) {
}

void vpux::PassDisablingExecutionContext::operator()(llvm::function_ref<void()> transform,
                                                     const mlir::tracing::Action& action) {
    if (const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&action)) {
        const auto passId = passAction->getPass().getArgument();
        const auto passName = passAction->getPass().getName();

        if (_disabledPasses != nullptr && (_disabledPasses->match(passId) || _disabledPasses->match(passName))) {
            return;
        }
    }

    transform();
}
