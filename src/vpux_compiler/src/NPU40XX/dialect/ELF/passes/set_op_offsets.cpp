//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/SymbolTable.h>

namespace vpux::ELF::arch40xx {
#define GEN_PASS_DECL_SETOPOFFSETS
#define GEN_PASS_DEF_SETOPOFFSETS
#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF::arch40xx

using namespace vpux;

namespace {

class SetOpOffsetsPass : public ELF::arch40xx::impl::SetOpOffsetsBase<SetOpOffsetsPass> {
public:
    explicit SetOpOffsetsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    Logger _log;
};

mlir::LogicalResult SetOpOffsetsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

void SetOpOffsetsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    mlir::MLIRContext* ctx = &getContext();
    const auto arch = VPU::getArch(netFunc);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    auto u64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    SmallVector<ELF::ElfSectionInterface> metadataSecVec;

    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto block = section.getBlock();

        if (section.getSectionType().value_or(ELF::SectionTypeAttr::SHT_NULL) ==
            ELF::SectionTypeAttr::VPU_SHT_CMX_METADATA) {
            metadataSecVec.push_back(section);
            continue;
        }

        uint64_t tracker = 0;
        for (auto& operation : block->getOperations()) {
            auto offsetAttr = mlir::IntegerAttr::get(u64Type, mlir::APInt(64, tracker, false));
            if (auto binarySizeOperation = mlir::dyn_cast_or_null<ELF::BinarySizeOpInterface>(&operation)) {
                tracker += binarySizeOperation.getBinarySizeCached(symRefMap, arch);
                binarySizeOperation.setMemoryOffset(offsetAttr);
            }
        }
    }
}

}  // namespace

//
// createAddELFSymbolTablePass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createSetOpOffsetsPass(Logger log) {
    return std::make_unique<SetOpOffsetsPass>(log);
}
