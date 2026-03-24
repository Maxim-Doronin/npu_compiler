//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/FormatVariadicDetails.h>

#include <cstdint>
#include <tuple>

namespace vpux::config {

struct Version final {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;

    Version(uint32_t major_, uint32_t minor_, uint32_t patch_): major(major_), minor(minor_), patch(patch_) {
    }
};

inline bool operator==(const Version& lhs, const Version& rhs) {
    return std::tie(lhs.major, lhs.minor, lhs.patch) == std::tie(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator<(const Version& lhs, const Version& rhs) {
    return std::tie(lhs.major, lhs.minor, lhs.patch) < std::tie(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator!=(const Version& lhs, const Version& rhs) {
    return !(lhs == rhs);
}

inline bool operator>(const Version& lhs, const Version& rhs) {
    return rhs < lhs;
}

inline bool operator<=(const Version& lhs, const Version& rhs) {
    return !(rhs < lhs);
}

inline bool operator>=(const Version& lhs, const Version& rhs) {
    return !(lhs < rhs);
}

}  // namespace vpux::config

namespace llvm {
template <>
struct format_provider<::vpux::config::Version> {
    static void format(const ::vpux::config::Version& version, raw_ostream& stream, StringRef) {
        stream << version.major << "." << version.minor << "." << version.patch;
    }
};
}  // namespace llvm
