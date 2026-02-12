//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_SHAVESTACKSRESERVEMEM
#define GEN_PASS_DEF_SHAVESTACKSRESERVEMEM
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
//  ShaveStacksReserveMemPass
//

class ShaveStacksReserveMemPass final : public VPU::impl::ShaveStacksReserveMemBase<ShaveStacksReserveMemPass> {
public:
    explicit ShaveStacksReserveMemPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ShaveStacksReserveMemPass::safeRunOnModule() {
    auto module = getOperation();
    auto shvPerTile = checked_cast<uint32_t>(
            config::getTileExecutor(module).getSubExecutor(config::ExecutorKind::SHAVE_ACT).getCount());
    auto* ctx = module->getContext();

    const size_t defaultStacksNum = 2;
    // Two stacks already reserved at the beginning of the CMX space, reserve resources for additional stacks if needed
    if (auto extraStacks = shvPerTile - defaultStacksNum; extraStacks > 0) {
        auto stackSize = static_cast<uint32_t>(CMX_SHAVE_STACK_SIZE.count());
        auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

        _log.trace("Shave stacks reserved CMX memory - size: '{0}'", extraStacks * stackSize);
        config::setShaveStacksReservedMemory(module, memSpaceAttr, extraStacks * stackSize, ELF::VPUX_SHAVE_ALIGNMENT);
    }
}

}  // namespace

//
// createShaveStacksReserveMemPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createShaveStacksReserveMemPass(Logger log) {
    return std::make_unique<ShaveStacksReserveMemPass>(log);
}
