//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <optional>

namespace vpux {

std::optional<config::ArchKind> parseArchKind(int argc, char* argv[]);
std::optional<config::Platform> parseNpuPlatform(int argc, char* argv[]);

std::optional<config::ArchKind> parseParamsAndDeduceArch(int argc, char* argv[]);

}  // namespace vpux
