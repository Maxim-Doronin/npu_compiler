//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/Regex.h>
#include <mlir/IR/Action.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

class PassDisablingExecutionContext {
private:
    std::shared_ptr<llvm::Regex> _disabledPasses;

public:
    explicit PassDisablingExecutionContext(llvm::StringRef disabledPasses);
    void operator()(llvm::function_ref<void()> transform, const mlir::tracing::Action& action);
};

}  // namespace vpux
