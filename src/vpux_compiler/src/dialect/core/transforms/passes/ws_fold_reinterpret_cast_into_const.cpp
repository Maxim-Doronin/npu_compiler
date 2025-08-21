//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::Core {
#define GEN_PASS_DECL_WSFOLDREINTERPRETCASTINTOCONST
#define GEN_PASS_DEF_WSFOLDREINTERPRETCASTINTOCONST
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

class FoldReinterpretCastIntoConst final :
        public Core::impl::WsFoldReinterpretCastIntoConstBase<FoldReinterpretCastIntoConst> {
public:
    explicit FoldReinterpretCastIntoConst(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FoldReinterpretCastIntoConst::safeRunOnFunc() {
    auto funcOp = getOperation();

    OpBuilderLogger listener(_log);
    mlir::OpBuilder builder(&getContext(), &listener);

    funcOp.walk([&](Core::ReinterpretCastOp castOp) {
        auto constOp = castOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr) {
            return;
        }
        const auto outputType = mlir::cast<NDTypeInterface>(castOp.getOutput().getType());

        auto contentAttr = constOp.getContentAttr();

        // Note: this pass performs *eager* constant folding which means
        // potentially significant RAM usage during compilation. this is fine
        // here because the pass is *never* supposed to be used outside of
        // testing pipelines.
        auto content = contentAttr.fold();
        const auto tensorType = mlir::RankedTensorType::get(outputType.getShape(), outputType.getElementType());
        auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(tensorType, content.getRawStorageBuf());

        auto newContentAttr = Const::ContentAttr::get(denseAttr);

        builder.setInsertionPoint(castOp);
        auto newConstOp = builder.create<Const::DeclareOp>(castOp->getLoc(), newContentAttr.getType(), newContentAttr);
        castOp.replaceAllUsesWith(newConstOp.getResult());
        castOp.erase();
        if (constOp->getUses().empty()) {
            constOp.erase();
        }
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createWsFoldReinterpretCastIntoConstPass(const Logger& log) {
    return std::make_unique<FoldReinterpretCastIntoConst>(log);
}
