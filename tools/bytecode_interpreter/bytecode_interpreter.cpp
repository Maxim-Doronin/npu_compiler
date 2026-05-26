//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/virtual_machine/virtual_machine.hpp"

#include <gflags/gflags.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

DEFINE_string(path, "", "[Required] Path to the bytecode file");
DEFINE_validator(path, [](const char* flagname, const std::string& value) {
    if (value.empty()) {
        std::cerr << "Error: the path to the bytecode file must be provided via the --" << flagname << " argument"
                  << std::endl;
        return false;
    }
    if (!std::filesystem::exists(value)) {
        std::cerr << "Error: the specified bytecode file does not exist: " << value << std::endl;
        return false;
    }
    return true;
});
DEFINE_string(mode, "run",
              "[Optional] Execution mode: 'run', 'print', 'print-full'. The 'print-full' mode includes the "
              "content of binary sections, such as constants or kernels");
DEFINE_validator(mode, [](const char* /*flagname*/, const std::string& value) {
    if (value != "run" && value != "print" && value != "print-full") {
        std::cerr << "Error: invalid execution mode '" << value << "'. Valid options are: run, print, print-full."
                  << std::endl;
        return false;
    }
    return true;
});

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const auto bytecodeFile = FLAGS_path;
    std::ifstream input(bytecodeFile, std::ios::binary);
    if (!input) {
        std::cerr << "Error: Failed to open bytecode file: " << bytecodeFile << std::endl;
        return 1;
    }
    std::cout << "Loading bytecode from file: " << bytecodeFile << std::endl;
    std::vector<uint8_t> bytecode((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::cout << "Bytecode size: " << bytecode.size() << " bytes" << std::endl;
    input.close();

    vpux::bytecode::VirtualMachine vm;
    const auto printOnly = FLAGS_mode == "print" || FLAGS_mode == "print-full";
    if (printOnly) {
        std::cout << "File content:" << std::endl;
        if (!vm.print(bytecode, FLAGS_mode == "print-full", /*indentLevel=*/1)) {
            std::cerr << "Error: Failed to print bytecode." << std::endl;
            return 1;
        }
        return 0;
    }

    if (!vm.parse(bytecode)) {
        std::cerr << "Error: Failed to parse bytecode." << std::endl;
        return 1;
    }
    try {
        vm.run();
    } catch (const std::exception& ex) {
        std::cerr << "Error during bytecode execution: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
