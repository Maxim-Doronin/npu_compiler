//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

using namespace vpux;

extern std::unordered_map<std::string, const std::pair<const uint8_t*, size_t>> shaveBinaryResourcesMap;

ShaveBinaryResources& ShaveBinaryResources::getInstance() {
    static ShaveBinaryResources instance;
    return instance;
}

vpux::SmallString ShaveBinaryResources::getSwKernelArchString(config::ArchKind archKind) {
    switch (archKind) {
    case config::ArchKind::NPU37XX:
        return vpux::SmallString("3720xx");
    case config::ArchKind::NPU40XX:
        return vpux::SmallString("4000xx");
    default:
        VPUX_THROW("unsupported archKind {0}", archKind);
        return vpux::SmallString("");
    }
}

llvm::ArrayRef<uint8_t> ShaveBinaryResources::getElf(llvm::StringRef kernelPath) const {
    auto symbolName = printToString("{0}_elf", kernelPath);
    const auto it = shaveBinaryResourcesMap.find(symbolName);

    VPUX_THROW_UNLESS(it != shaveBinaryResourcesMap.end(), "Can't find 'elf' for kernel symbol '{0}'", symbolName);

    const auto [symbolData, symbolSize] = it->second;
    return llvm::ArrayRef<uint8_t>(symbolData, symbolSize);
}

void ShaveBinaryResources::addCompiledElf(llvm::StringRef funcName, std::vector<uint8_t>& binary, llvm::StringRef arch,
                                          bool overwrite) {
    auto symbolName = printToString("{0}_{1}_elf", funcName, arch);
    auto data = shaveBinaryResourcesMap.find(symbolName);

    if (data != shaveBinaryResourcesMap.end()) {
        if (!overwrite) {
            return;
        }
        shaveBinaryResourcesMap.erase(symbolName);
    }

    // Store data in unique_ptr for memory leak prevention
    auto permArray = std::make_unique<uint8_t[]>(binary.size());
    // Transfer ownership to vector s.t. the unique_ptr is alive for the lifetime of ShaveBinaryResources
    _elfPermStorage.push_back(std::move(permArray));
    auto& ref = _elfPermStorage.back();

    memcpy(ref.get(), binary.data(), binary.size());
    shaveBinaryResourcesMap.insert(std::make_pair(symbolName, std::make_pair(ref.get(), binary.size())));
}

void ShaveBinaryResources::loadElfData(mlir::ModuleOp module) {
    ShaveBinaryResources& sbr = ShaveBinaryResources::getInstance();

    std::string line;

    std::ifstream ifileList("FileList.in", std::ifstream::in);
    if (!ifileList.is_open()) {
        return;
    }

    while (std::getline(ifileList, line)) {
        std::vector<uint8_t> binary;

        std::ifstream ifileElf(line, std::ifstream::in);
        VPUX_THROW_UNLESS(ifileElf.is_open(), "ELF file not found.");

        // Get length of file:
        ifileElf.seekg(0, std::ios::end);
        int length = ifileElf.tellg();
        ifileElf.seekg(0, std::ios::beg);

        auto buffer = std::vector<char>(length);
        ifileElf.read(buffer.data(), length);
        ifileElf.close();

        binary.insert(binary.end(), buffer.begin(), buffer.end());

        std::string funcName;
        std::getline(ifileList, funcName);

        config::ArchKind archKind = config::getArch(module.getOperation());
        auto kernelArch = getSwKernelArchString(archKind);

        sbr.addCompiledElf(funcName, binary, kernelArch, true);
    }

    ifileList.close();
}
