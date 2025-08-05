//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux {

// TODO: E#162744 remove this

// For defining force link symbols (e.g., in source files)
#define DEFINE_FORCE_LINK(sym)         \
    extern "C" void forceLink##sym() { \
    }

// For declaring force link symbols (where they are undefined)
#define DECLARE_FORCE_LINK(sym) extern "C" void forceLink##sym()
#define FORCE_LINK(sym) forceLink##sym()

}  // namespace vpux
