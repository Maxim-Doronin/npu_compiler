//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"

#include <cstdint>

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDABIVERSION
#define GEN_PASS_DEF_ADDABIVERSION
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {
//
// AddABIVersionPass
//

class AddABIVersionPass : public ELF::impl::AddABIVersionBase<AddABIVersionPass> {
public:
    AddABIVersionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddABIVersionPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    mlir::OpBuilder builder(&(funcOp.getBody().front().back()));
    builder.create<ELF::ABIVersionOp>(builder.getUnknownLoc());
}

}  // namespace

//
// createAddABIVersionPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddABIVersionPass(Logger log) {
    return std::make_unique<AddABIVersionPass>(log);
}
