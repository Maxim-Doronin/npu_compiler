//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/import.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/elf_importer.hpp"

using namespace vpux;

mlir::OwningOpRef<mlir::ModuleOp> vpux::ELFNPU37XX::importELF(mlir::MLIRContext* ctx, const std::string& elfFileName,
                                                              Logger log) {
    return vpux::ELFNPU37XX::ElfImporter(ctx, elfFileName, log).read();
}
