//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/tool_registration.hpp"
#include "vpux/compiler/tools/options.hpp"

#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        auto registry = vpux::createDialectRegistry(vpux::DummyOpMode::ENABLED);
        vpux::registerAllPassesGlobally();
        if (auto archKind = vpux::parseArchKind(argc, argv); archKind.has_value()) {
            vpux::registerAllHwSpecificComponents(registry, archKind.value());
        }

        return mlir::asMainReturnCode(mlir::MlirLspServerMain(argc, argv, registry));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
