//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/developer_path_utils.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

using namespace vpux;

std::string vpux::concatenatePath(StringRef baseName, StringRef suffix) {
    llvm::SmallString<128> concatStr(baseName);
    llvm::sys::path::append(concatStr, suffix);
    return concatStr.str().str();
}

std::string vpux::getPerfDebugFilePath(StringRef fileName) {
    return concatenatePath(perfDebugFilesRoot, fileName);
}

void vpux::createDirectory(StringRef pathName) {
    if (std::error_code EC = llvm::sys::fs::create_directories(pathName.str())) {
        VPUX_THROW("Failed to create directory '{0}': {1}", pathName.str(), EC.message());
    }
}
