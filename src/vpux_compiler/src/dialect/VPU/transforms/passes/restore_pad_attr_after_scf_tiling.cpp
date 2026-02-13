//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_RESTOREPADATTRAFTERSCFTILING
#define GEN_PASS_DEF_RESTOREPADATTRAFTERSCFTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// RestorePadAttrAfterSCFTilingPass
//
class RestorePadAttrAfterSCFTilingPass final :
        public VPU::impl::RestorePadAttrAfterSCFTilingBase<RestorePadAttrAfterSCFTilingPass> {
public:
    explicit RestorePadAttrAfterSCFTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void processSCFForOp(mlir::scf::ForOp forOp);
};

/**
 * @brief Processes a single scf.for operation to restore padding attributes on convolution-like operations.
 *
 * This function walks through all operations within the given ForOp, identifies convolution-like operations
 * that have a tensor.pad operation as their parent, and restores the original padding attribute while
 * removing the pad operation. The function handles dynamic tensor shapes by inserting tensor.cast operations
 * when necessary to ensure type compatibility.
 *
 * @param forOp The scf.for operation to process
 */
void RestorePadAttrAfterSCFTilingPass::processSCFForOp(mlir::scf::ForOp forOp) {
    restorePaddingAttribute(forOp, _log.nest());
}

//
// safeRunOnFunc
//
void RestorePadAttrAfterSCFTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::OpBuilder builder(&ctx);

    _log.trace("Starting RestorePadAttrAfterSCFTiling pass on function: {0}", func.getName());

    // Iterate over all scf.for operations in the entry function
    func.walk([&](mlir::scf::ForOp forOp) {
        processSCFForOp(forOp);
    });
}

}  // namespace

//
// createRestorePadAttrAfterSCFTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createRestorePadAttrAfterSCFTilingPass(Logger log) {
    return std::make_unique<RestorePadAttrAfterSCFTilingPass>(log);
}
