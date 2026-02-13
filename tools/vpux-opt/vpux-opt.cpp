//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init.hpp"
#include "vpux/compiler/tool_registration.hpp"
#include "vpux/compiler/tools/options.hpp"

#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        // TODO: need to rework this unconditional replacement for dummy ops
        // there is an option for vpux-translate we can do it in the same way
        // Ticket: E#50937
        auto registry = vpux::createDialectRegistry(vpux::DummyOpMode::ENABLED);
        vpux::registerAllPassesGlobally();
        if (auto archKind = vpux::parseArchKind(argc, argv); archKind.has_value()) {
            vpux::registerAllHwSpecificComponents(registry, archKind.value());
        }

        return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "NPU Optimizer Testing Tool", registry));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
