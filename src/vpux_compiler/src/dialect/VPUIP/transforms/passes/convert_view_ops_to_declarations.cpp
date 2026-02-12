//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTVIEWOPSTODECLARATIONS
#define GEN_PASS_DEF_CONVERTVIEWOPSTODECLARATIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ViewLikeRewrite
//

class ViewLikeRewrite final : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    ViewLikeRewrite(mlir::MLIRContext* ctx, const AliasesInfo* aliasInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx), _aliasInfo(aliasInfo), _log(log) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Byte calculateOffset(mlir::Value val) const;
    Shape calculateDimOffsets(mlir::Value val) const;

private:
    const AliasesInfo* _aliasInfo = nullptr;
    Logger _log;
};

Byte ViewLikeRewrite::calculateOffset(mlir::Value val) const {
    Byte offset(0);

    if (auto source = _aliasInfo->getSource(val)) {
        offset = calculateOffset(source);
    }

    if (auto declareOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(val.getDefiningOp())) {
        offset += Byte(declareOp.getByteOffset());
    }

    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(val.getDefiningOp())) {
        offset += subViewOp.getByteOffset();
    }
    if (auto extractFlatViewOp = mlir::dyn_cast_or_null<VPUIP::ExtractFlatSliceOp>(val.getDefiningOp())) {
        offset += extractFlatViewOp.getByteOffset();
    }

    return offset;
}

Shape ViewLikeRewrite::calculateDimOffsets(mlir::Value val) const {
    Shape viewOffsets{};
    Shape addendOffsets{};

    if (auto source = _aliasInfo->getSource(val)) {
        viewOffsets = calculateDimOffsets(source);
    }

    if (auto declareOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(val.getDefiningOp())) {
        if (declareOp->hasAttr(vpux::viewOffsetsAttrName)) {
            addendOffsets = Shape(parseIntArrayAttr<int64_t>(
                    mlir::dyn_cast_or_null<mlir::ArrayAttr>(declareOp->getAttr(vpux::viewOffsetsAttrName))));
        }
    }

    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(val.getDefiningOp())) {
        addendOffsets = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
    }

    auto viewOffsetsVec = to_small_vector(viewOffsets);
    if (!viewOffsetsVec.empty()) {
        auto tensorRank = checked_cast<size_t>(mlir::cast<vpux::NDTypeInterface>(val.getType()).getRank());
        if (tensorRank > viewOffsetsVec.size()) {
            viewOffsetsVec.insert(viewOffsetsVec.end(), tensorRank - viewOffsetsVec.size(), 0);
        }
    }
    if (!addendOffsets.empty()) {
        auto addendOffsetsVec = to_small_vector(addendOffsets);
        if (viewOffsetsVec.size() < addendOffsetsVec.size()) {
            viewOffsetsVec.insert(viewOffsetsVec.end(), addendOffsetsVec.size() - viewOffsetsVec.size(), 0);
        }
        std::transform(viewOffsetsVec.begin(), viewOffsetsVec.end(), addendOffsetsVec.begin(), viewOffsetsVec.begin(),
                       std::plus<>{});
    }

    return Shape(viewOffsetsVec);
}

mlir::LogicalResult ViewLikeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<Core::ReinterpretCastOp, VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::PermuteCastOp,
                   VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp, VPUIP::NonDistributedCastOp, VPUIP::ShapeCastOp,
                   VPUIP::StubOp, VPUIP::ViewOp, VPUIP::ExtractFlatSliceOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    const auto origVal = mlir::isa<VPUIP::NonDistributedCastOp>(origOp) ? origOp->getOperand(0) : origOp->getResult(0);
    const Byte offset = calculateOffset(origVal);
    auto dimOffsets = calculateDimOffsets(origVal);

    const auto rootVal = _aliasInfo->getRoot(origVal);

    auto declareOp = rootVal.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(declareOp == nullptr, "Unsupported source owner: '{0}'", rootVal);

    _log.nest().trace("It aliases internal buffer produced by '{0}'", declareOp->getLoc());

    auto section = declareOp.getSection();
    auto sectionIndex = declareOp.getSectionIndex();
    // TODO:#114687 -- section index is missed for CMX for some reason
    if (!sectionIndex.has_value()) {
        const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
        auto memSpaceIndex = outType.getMemSpace().getIndex();
        if (memSpaceIndex.has_value()) {
            sectionIndex = getIntArrayAttr(rewriter, ArrayRef({memSpaceIndex.value()}));
        }
    }

    const auto outType = origOp->getResult(0).getType();
    auto swizzlingScheme = VPUIP::getSwizzlingSchemeAttr(outType);
    mlir::IntegerAttr swizzlingKey;
    if (swizzlingScheme && swizzlingScheme.getKey().getInt() != 0) {
        swizzlingKey = swizzlingScheme.getKey();
    }

    mlir::ArrayAttr sectionIndexAttr = sectionIndex.has_value() ? sectionIndex.value() : nullptr;
    auto newDeclareOp = rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, outType, section, sectionIndexAttr,
                                                                            offset.count(), swizzlingKey);

    if (!dimOffsets.empty() &&
        (section == VPURT::BufferSection::NetworkInput || section == VPURT::BufferSection::NetworkOutput) &&
        vpux::net::isArgStrided(origOp->getParentOfType<mlir::ModuleOp>(),
                                declareOp.getNonEmptySectionIndex().front())) {
        newDeclareOp->setAttr(vpux::viewOffsetsAttrName, getIntArrayAttr(getContext(), dimOffsets));
    }

    return mlir::success();
}

//
// ConvertViewOpsToDeclarationsPass
//

class ConvertViewOpsToDeclarationsPass final :
        public VPUIP::impl::ConvertViewOpsToDeclarationsBase<ConvertViewOpsToDeclarationsPass> {
public:
    explicit ConvertViewOpsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertViewOpsToDeclarationsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();
    // The logic for ConcatView has been moved to BreakDataFlow pass
    // Leave ConcatView illegal here for sanity check
    target.addIllegalOp<Core::ReinterpretCastOp, VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::ConcatViewOp,
                        VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp,
                        VPUIP::NonDistributedCastOp, VPUIP::ShapeCastOp, VPUIP::StubOp, VPUIP::ViewOp,
                        VPUIP::ExtractFlatSliceOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ViewLikeRewrite>(&ctx, &aliasInfo, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertViewOpsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertViewOpsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertViewOpsToDeclarationsPass>(log);
}
