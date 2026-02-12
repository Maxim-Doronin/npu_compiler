//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/ClangTidyModuleRegistry.h>

namespace clang::tidy::plugin {

struct ClangTidyPlugin : ClangTidyModule {
    void addCheckFactories(ClangTidyCheckFactories&) override {
    }
};

static ClangTidyModuleRegistry::Add<ClangTidyPlugin> Register("Clang-Tidy Plugin", "clang-tidy custom checks");

}  // namespace clang::tidy::plugin
