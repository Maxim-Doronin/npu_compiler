//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <memory>

namespace vpux {

constexpr bool arePrivateOptionsEnabled() {
#ifdef PRIVATE_COMPILER_OPTIONS_ENABLED
    return true;
#else
    return false;
#endif
}

template <typename T>
std::unique_ptr<T> parseCompilationModeParams(StringRef compilationModeParams, VPU::ArchKind arch) {
    if (arePrivateOptionsEnabled()) {
        return T::createFromString(compilationModeParams, arch);
    }
    if (auto publicOptions = PublicOptions::createFromString(compilationModeParams, arch)) {
        return PublicOptions::createFrom<T>(publicOptions);
    }
    return nullptr;
}

}  // namespace vpux
