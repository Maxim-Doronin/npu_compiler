//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

std::string concatenatePath(StringRef baseName, StringRef suffix);
std::string getPerfDebugFilePath(StringRef fileName);
void createDirectory(StringRef pathName);

}  // namespace vpux
