//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SERIALIZEELFTOBINARY
#define GEN_PASS_DEF_SERIALIZEELFTOBINARY
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
class SerializeELFToBinaryPass : public VPUIP::impl::SerializeELFToBinaryBase<SerializeELFToBinaryPass> {
public:
    explicit SerializeELFToBinaryPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void replaceModuleOpsWithBinaryOps(mlir::ModuleOp parentModuleOp, VPU::ArchKind arch);
    Logger _log;
};

void SerializeELFToBinaryPass::replaceModuleOpsWithBinaryOps(mlir::ModuleOp parentModuleOp, VPU::ArchKind arch) {
    for (auto moduleOp : parentModuleOp.getOps<mlir::ModuleOp>()) {
        net::NetworkInfoOp netInfo;
        mlir::func::FuncOp funcOp;
        net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
        mlir::OpBuilder moduleBuilder(moduleOp);

        // Serialize ELF module to binary
        std::vector<uint8_t> binaryBuffer;
        if (arch == VPU::ArchKind::NPU37XX) {
            binaryBuffer = vpux::ELFNPU37XX::exportToELF(moduleOp);

        } else {
            binaryBuffer = vpux::ELF::exportToELF(moduleOp);
        }
        auto object = moduleBuilder.getAttr<VPUIP::ObjectAttr>(moduleBuilder.getStringAttr(
                StringRef(reinterpret_cast<const char*>(binaryBuffer.data()), binaryBuffer.size())));

        // Store the serialized ELF data as binary data op
        auto binaryOp = moduleBuilder.create<VPUIP::BinaryOp>(moduleOp.getLoc(), moduleOp.getName().value());
        mlir::OpBuilder binaryOpBuilder(binaryOp.getBody());
        binaryOpBuilder.create<VPUIP::BinaryDataOp>(binaryOp.getLoc(), "serialized_" + funcOp.getName().str(), object);

        // Kernel functions do not return data/objects. All inputs and output ptrs are passed as function arguments
        // func op is set to private to indicate that function has no body just declaration
        const auto funcType = mlir::FunctionType::get(&getContext(), funcOp.getArgumentTypes(), {});
        auto newFuncOp = binaryOpBuilder.create<mlir::func::FuncOp>(binaryOp.getLoc(), funcOp.getName(), funcType);
        newFuncOp.setPrivate();

        moduleOp.erase();
    }
}

void SerializeELFToBinaryPass::safeRunOnFunc() {
    auto func = getOperation();
    auto arch = VPU::getArch(func);
    _log.debug("Serialize VPUIP Module to Binary {0}", func.getName());

    auto parentModuleOp = func->getParentOfType<mlir::ModuleOp>();
    replaceModuleOpsWithBinaryOps(parentModuleOp, arch);

    for (auto nestedCallOp : func.getOps<vpux::Core::NestedCallOp>()) {
        mlir::OpBuilder builder(func);
        builder.setInsertionPoint(nestedCallOp);

        // Match the Core.NestedCallOp with the new funcOp signature
        builder.create<vpux::Core::NestedCallOp>(nestedCallOp.getLoc(), nestedCallOp.getCalleeAttr(),
                                                 mlir::TypeRange({}), nestedCallOp.getOperands());
        nestedCallOp.erase();
    }
}
}  // namespace

//
// createSerializeELFToBinaryPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSerializeELFToBinaryPass(Logger log) {
    return std::make_unique<SerializeELFToBinaryPass>(log);
}
