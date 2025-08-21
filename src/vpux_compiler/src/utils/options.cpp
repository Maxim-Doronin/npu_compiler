//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/utils/options.hpp"
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace llvm {
inline ::llvm::raw_ostream& operator<<(::llvm::raw_ostream& p, vpux::WorkloadManagementMode value) {
    auto valueStr = vpux::stringifyEnum(value);
    return p << valueStr;
}

template <>
struct format_provider<vpux::WorkloadManagementMode> {
    static void format(const vpux::WorkloadManagementMode& val, raw_ostream& OS, StringRef /*Options*/) {
        OS << vpux::stringifyEnum(val);
    }
};

}  // namespace llvm
