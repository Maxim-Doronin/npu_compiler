//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/tools/options.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassOptions.h>

#include <algorithm>

using namespace vpux;

//
// parseArchKind
//

namespace {
constexpr char VPU_ARCH_OPTION_NAME[] = "vpu-arch";

// Parses 'vpu-arch' followed by the alphanumerical string which represents the
// architecture field. Example: 'vpu-arch=NPU40XX'
StringRef parseVpuArch(StringRef candidate) {
    const auto start = candidate.find(VPU_ARCH_OPTION_NAME);
    if (bool notFound = (start == StringRef::npos); notFound) {
        return {};
    }

    // Note: std::size("vpu-arch") includes null terminator, so adding 1 that
    // skips over '=' is not needed as it's accounted for via std::size()
    const auto optionValueStart = start + std::size(VPU_ARCH_OPTION_NAME);
    // take everything until first non-alphanumeric character
    const auto endIt = std::find_if_not(candidate.begin() + optionValueStart, candidate.end(), [](char c) {
        return std::isalnum(c);
    });
    const auto length = std::distance(candidate.begin() + optionValueStart, endIt);
    return candidate.substr(optionValueStart, length);
}

// Parses '-h' / '-help' and other options that end up printing "usage"
// information (or similar) instead of compiling the IR.
bool isSpecialCaseOption(StringRef candidate) {
    // handle '-' and '--' suffixes (it seems that MLIR allows both cases e.g.
    // -help and --help)
    for (size_t i = 0; i < 2; ++i) {
        if (candidate.starts_with('-')) {
            candidate = candidate.drop_front(1);
        }
    }

    const StringLiteral specialCases[] = {"h", "help", "help-hidden", "help-list", "help-list-hidden", "version"};
    return std::any_of(std::begin(specialCases), std::end(specialCases), [&](const StringLiteral& x) {
        return x == candidate;
    });
}
}  // namespace

std::optional<vpux::config::ArchKind> vpux::parseArchKind(int argc, char* argv[]) {
    static llvm::cl::OptionCategory vpuxOptOptions("NPU Options");
    // Please use this option to test pipelines only (DefaultHW, ReferenceSW, etc.)
    // This option allows us to avoid ambiguity here when the parameters contradict each other:
    // "vpux-opt --init-compiler="vpu-arch=NPU37XX compilation-mode=ReferenceSW" --default-hw-mode"
    // Instead you should only pass arch version:
    // "vpux-opt --vpu-arch=NPU37XX --default-hw-mode"
    static llvm::cl::opt<std::string> archOpt(VPU_ARCH_OPTION_NAME, llvm::cl::desc("VPU architecture to compile for"),
                                              llvm::cl::init(""), llvm::cl::cat(vpuxOptOptions));

    bool archKindCanBeEmpty = false;  // flags --help appearance in command-line
    StringRef archKindString{};
    for (int i = 1; i < argc; ++i) {  // Note: argv[0] is always the program name
        archKindCanBeEmpty = archKindCanBeEmpty || isSpecialCaseOption(argv[i]);

        auto maybeArchKind = parseVpuArch(argv[i]);
        if (bool notFound = maybeArchKind.empty(); notFound) {
            continue;
        }
        VPUX_THROW_WHEN(!archKindString.empty(),
                        "ArchKind value is ambiguous. Only one option can be used at a time, either \"vpu-arch\" or "
                        "\"init-compiler\"");
        archKindString = maybeArchKind;
    }

    // The logic is the following: if --help (or similar option) is set, arch
    // kind may be completely omitted (e.g. no --vpu-arch / --init-compiler)
    VPUX_THROW_WHEN(archKindString.empty() && !archKindCanBeEmpty,
                    "Can't get ArchKind value. Did you forget to specify \"vpu-arch\" or \"init-compiler\"?");
    auto archKind = vpux::config::symbolizeEnum<vpux::config::ArchKind>(archKindString);
    VPUX_THROW_UNLESS(archKind.has_value() || archKindCanBeEmpty, "Unknown VPU architecture : '{0}'", archKindString);
    return archKind;
}
