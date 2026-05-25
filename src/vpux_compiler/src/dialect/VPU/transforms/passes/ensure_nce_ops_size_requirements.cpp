//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/sparsity.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <numeric>

namespace vpux::VPU {
#define GEN_PASS_DECL_ENSURENCEOPSSIZEREQUIREMENTS
#define GEN_PASS_DEF_ENSURENCEOPSSIZEREQUIREMENTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

SmallVector<Dim> getDimsOverKHWLimit(ShapeRef shape, ArrayRef<int64_t> dimThresholds) {
    SmallVector<Dim> wrongDims = {};
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = Dim(i);
        if (shape[dim] > dimThresholds[i]) {
            wrongDims.push_back(dim);
        }
    }
    return wrongDims;
}

bool hasSplitOverKernelStrategy(mlir::Operation* op) {
    if (mlir::isa<VPU::ClusteredOpInterface>(op)) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
        const auto strategy = clusteredOp.getMultiClusterStrategy();
        if (!strategy.has_value()) {
            return false;
        }

        return strategy.value() == VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return false;
}

// Matches the subgraph pattern for NCEConvolutionOp:
//   weights <- Slice <- PermuteCast <- AffineReshape (single use) <- Concat (single use,
//   >= 3 inputs, single axis, arg0..argN-3 equal-sized on that axis).
// Returns the LCM of the per-input concat-axis size and the NCE channel alignment,
// or 0 if the pattern does not match.
static int64_t getAlignmentFromConcatSlicePattern(mlir::Operation* op) {
    auto nceConvOp = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op);
    if (nceConvOp == nullptr) {
        return 0;
    }
    auto weights = nceConvOp.getFilter();

    // Trace from weights
    auto sliceOp = weights.getDefiningOp<VPU::SliceOp>();
    if (sliceOp == nullptr) {
        return 0;
    }
    auto permuteCastOp = sliceOp.getInput().getDefiningOp<VPU::PermuteCastOp>();
    if (permuteCastOp == nullptr) {
        return 0;
    }
    auto affineReshapeOp = permuteCastOp.getInput().getDefiningOp<VPU::AffineReshapeOp>();
    if (affineReshapeOp == nullptr || !affineReshapeOp->hasOneUse()) {
        return 0;
    }
    auto concatOp = affineReshapeOp.getInput().getDefiningOp<VPU::ConcatOp>();
    if (concatOp == nullptr || !concatOp->hasOneUse()) {
        return 0;
    }
    // Require exactly one concat axis.
    const auto concatAxes = VPU::getConcatAxes(concatOp);
    if (concatAxes.size() != 1) {
        return 0;
    }
    const auto axis = Dim(*concatAxes.begin());
    const auto inputs = concatOp.getInputs();
    const int64_t firstSize = getShape(inputs.front())[axis];
    if (inputs.size() > 2) {
        // inputs[0] through inputs[N-3] must have equal size on the concat axis.
        // The last two inputs are allowed to differ (e.g. remainder tiles).
        for (auto input : inputs.drop_back(2)) {
            if (getShape(input)[axis] != firstSize) {
                return 0;
            }
        }
    }

    // Align to the LCM of the per-input concat size and the NCE channel alignment
    // so that each tile satisfies both the concat boundary and HW alignment requirements.
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
    const int64_t nceAlignment = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    const int64_t alignment = std::lcm(firstSize, nceAlignment);

    // If the alignment exceeds the HW dimension limit, tiles cannot be produced within
    // that limit, so fall back to the default alignment.
    if (alignment > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return 0;
    }
    return alignment;
}

class EnsureNCEOpSizeRequirements final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    EnsureNCEOpSizeRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("EnsureNCEOpSizeRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureNCEOpSizeRequirements::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    rewriter.setInsertionPoint(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto log = _log.nest();
    const auto tilingMode = TilingMode::ISOLATED;
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    _log.nest(4).trace("Tile Dim order is {0}", tileDimOrder);
    const auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = config::getTileExecutor(moduleOp).getCount();
    const int64_t concatInputAlignment = getAlignmentFromConcatSlicePattern(op);

    const auto getTilesWithOptionalConcatAlignment = [&](ShapeRef tilesOnDim) -> mlir::FailureOr<OutputTiling> {
        if (concatInputAlignment == 0) {
            return fillDividedTiles(op, tilesOnDim, outputShape);
        }

        // Preserve op-specific alignment and tile-unroll order, then LCM the C-dimension
        // alignment with the concat input alignment to respect concat boundaries.
        auto alignment = getAlignment(op, tilesOnDim, outputShape);
        alignment[Dims4D::Act::C.ind()] = std::lcm(alignment[Dims4D::Act::C.ind()], concatInputAlignment);
        auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
        auto unrollSpatialFirst = isSpatialFirstNestedTiling(op, tilesOnDim);
        return fillDividedTiles(tilesOnDim, outputShape, optionalAlignment, unrollSpatialFirst);
    };

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, Dim dimToTile, ArrayRef<int64_t> dimThresholds) -> bool {
        const auto tiles = getTilesWithOptionalConcatAlignment(nTilesOnDim);
        if (mlir::failed(tiles)) {
            return false;
        }
        for (auto tile : tiles.value()) {
            if (tile.shape[dimToTile] > dimThresholds[dimToTile.ind()]) {
                return false;
            }
            auto inputTiling = origOp.backInferTileInfo(tile, log);
            auto& inTiles = inputTiling.tiles;
            if ((dimToTile != Dims4D::Act::C) &&
                (inTiles.begin()->shape[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                return false;
            }
        }
        return true;
    };

    // Construct dim-specific thresholds for input and output shapes
    // In our test, extending the threshold on Dim C can improve performance by reducing workloads for SOK NCE
    // operations when the number of clusters is greater than 2
    SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
    if (hasSplitOverKernelStrategy(op) && numClusters > 2) {
        outputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
    }

    for (auto tileDimIter = tileDimOrder.begin(); tileDimIter < tileDimOrder.end(); ++tileDimIter) {
        auto dimToTile = *tileDimIter;
        while (!isSupportedTileSize(nTilesOnDim, dimToTile, outputDimThresholds) &&
               nTilesOnDim[dimToTile] <= outputShape[dimToTile]) {
            _log.nest(1).trace("Failed to tile {0} at {1} with {2}", op->getName(), dimToTile, nTilesOnDim);
            ++nTilesOnDim[dimToTile];
        }
    }

    // In case of single tile scheduled there is no need for tiling
    if (llvm::none_of(nTilesOnDim, [](int64_t tiles) {
            return tiles > 1;
        })) {
        return mlir::failure();
    }

    const auto tilesNew = getTilesWithOptionalConcatAlignment(nTilesOnDim);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Apply Tiling Strategy for {0} with {1}", op->getName(), nTilesOnDim);
    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, log.nest());
}

//
//  EnsureConvICRequirements
//

class EnsureConvICRequirements final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    EnsureConvICRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _log(log) {
        this->setDebugName("EnsureConvICRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureConvICRequirements::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    // Split over IC supported only for NCEConvolutionOp
    // TODO: E#70421

    // Get some of the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getShape(origOp.getInput());
    auto inputC = inputShape[Dims4D::Act::C];

    if (inputC <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return mlir::failure();
    }

    const auto kernelShape = getShape(origOp.getFilter());
    auto kernelW = kernelShape[Dims4D::Filter::KX];
    auto kernelH = kernelShape[Dims4D::Filter::KY];

    auto maxTiles = vpux::divUp(inputC, VPU::NCEInvariant::VPU_DIMENSION_LIMIT);

    if (maxTiles == 1) {
        return mlir::failure();
    }

    Shape nTilesOnDim(inputShape.size(), 1);
    nTilesOnDim[Dims4D::Act::C] = maxTiles;
    SmallVector<int64_t> alignment(inputShape.size(), 1);
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    auto inAlignment = VPU::NCEInvariant::getAlignment(inType.getElementType());
    auto weightsAlignment = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    // Weights alignment requirement is IC * KH * KW aligned with weightsAlignment. For
    // int4 case, weightsAlignment = 32, if KH = 2, then IC = 16 can meet the requirement.
    // So here we fist check if inAlignment can meet the requirement or not.
    if ((inAlignment * kernelW * kernelH) % weightsAlignment == 0) {
        alignment[Dims4D::Act::C.ind()] = inAlignment;
    } else {
        alignment[Dims4D::Act::C.ind()] = weightsAlignment;
    }

    const int64_t icAlign = getAlignmentFromConcatSlicePattern(origOp);
    if (icAlign != 0) {
        alignment[Dims4D::Act::C.ind()] = std::lcm(alignment[Dims4D::Act::C.ind()], icAlign);
    }

    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    const auto tiles = fillDividedTiles(nTilesOnDim, inputShape, optionalAlignment);

    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    auto weightInput = origOp.getFilter();
    // check for parent weight shave dequantize op
    auto weightDequantizeOp = weightInput.getDefiningOp<VPU::DequantizeOp>();
    if (weightDequantizeOp != nullptr) {
        weightInput = weightDequantizeOp.getInput();
    }

    SmallVector<VPU::NCEConvolutionOp> convOps;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    SmallVector<VPU::DequantizeOp> dequantizeOps;
    mlir::Value result = VPU::splitNCEConvolutionOverIC(origOp, weightInput, convOps, addOps, dequantizeOps,
                                                        tiles.value(), weightDequantizeOp, rewriter, _log.nest());

    rewriter.replaceOp(origOp, result);

    return mlir::success();
}

//
// EnsureNCEOpsSizeRequirementsPass
//

class EnsureNCEOpsSizeRequirementsPass final :
        public VPU::impl::EnsureNCEOpsSizeRequirementsBase<EnsureNCEOpsSizeRequirementsPass> {
public:
    explicit EnsureNCEOpsSizeRequirementsPass(bool enableOutputEnsurance,
                                              bool enableDequantWeightEnsuranceBeforeStrategy, StringRef skipConvOC,
                                              StringRef skipEltwiseOC, Logger log) {
        this->enableOutputEnsurance = enableOutputEnsurance;
        this->enableDequantWeightEnsuranceBeforeStrategy = enableDequantWeightEnsuranceBeforeStrategy;
        this->skipConvOC = skipConvOC.str();
        this->skipEltwiseOC = skipEltwiseOC.str();
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void EnsureNCEOpsSizeRequirementsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();

    const auto validateSkipOCMode = [](StringRef mode, StringRef optionName) {
        VPUX_THROW_WHEN(mode != "SKIP_NONE" && mode != "SKIP_LARGE_SPATIAL" && mode != "SKIP_ALL",
                        "Unknown {0} mode '{1}': expected SKIP_NONE, SKIP_LARGE_SPATIAL, or SKIP_ALL", optionName,
                        mode);
    };
    validateSkipOCMode(skipConvOC, "skip-conv-oc");
    validateSkipOCMode(skipEltwiseOC, "skip-eltwise-oc");

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        // TODO: #-196283 There is no pattern rewriter for the VPU.NCEMatMulOp,
        // it is better to catch the illegal operation and abort compilation process as soon as possible
        if (!mlir::isa<VPU::NCEConvolutionOp, VPU::NCEMatMulOp>(op)) {
            return true;
        }

        const auto inputShape = getShape(op->getOperand(0));
        auto channelIndex = Dims4D::Act::C;
        if (mlir::isa<VPU::NCEMatMulOp>(op)) {
            channelIndex = DimsGroups5D::Act::C;
        }
        return inputShape[channelIndex] <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    });

    patterns.add<EnsureConvICRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }

    // If output shape ensurance is disabled, skip the rest of the pass
    // OC will be split at multi-cluster and tiling pass if needed
    if (!enableOutputEnsurance) {
        return;
    }

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op)) {
            return true;
        }

        if (mlir::isa<VPU::TilingInfoOpInterface>(op)) {
            const auto inputShape = getShape(op->getOperand(0));
            const auto outputShape = getShape(op->getResult(0));
            const auto numClusters = config::getTileExecutor(moduleOp).getCount();

            // Construct dim-specific thresholds for input and output shapes
            // In our test, extending the threshold on Dim C can improve performance by reducing workloads for SOK NCE
            // operations when the number of clusters is greater than 2
            SmallVector<int64_t> inputDimThresholds(inputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
            SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
            if (hasSplitOverKernelStrategy(op) && numClusters > 2) {
                inputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
                outputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
            }

            auto inSizeWrongDims = getDimsOverKHWLimit(inputShape, inputDimThresholds);
            if (!inSizeWrongDims.empty()) {
                _log.nest(2).debug("Input size has dims greater than HW requirements: {0}", inSizeWrongDims);
            }
            auto outSizeWrongDims = getDimsOverKHWLimit(outputShape, outputDimThresholds);
            if (!outSizeWrongDims.empty()) {
                _log.nest(2).debug("Output size has dims greater than HW requirements: {0}", outSizeWrongDims);
            }
            // Skip slicing conv with dequant weight input before strategy is assigned : this allows for more vertical
            // fusion for large convs
            if (enableDequantWeightEnsuranceBeforeStrategy) {
                if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
                    auto weightDequantizeOp = convOp.getFilter().getDefiningOp<VPU::DequantizeOp>();
                    if (weightDequantizeOp != nullptr) {
                        _log.nest(2).debug("Allow op {0} with dequant weights to skip dimension check before strategy "
                                           "assignment",
                                           op->getLoc());
                        return true;
                    }
                }
            }

            // Skip slicing C for per-channel based NCE ops, which will be handled later in tiling pass
            // This will benefit vertical fusion
            const auto eraseChannel = [&](SmallVector<Dim>& wrongDims) {
                wrongDims.erase(std::remove(wrongDims.begin(), wrongDims.end(), Dims4D::Act::C), wrongDims.end());
            };
            if (mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op)) {
                _log.nest(2).debug("Skip checking C dimension for per-channel based NCE op {0} at {1}", op->getName(),
                                   op->getLoc());
                eraseChannel(inSizeWrongDims);
                eraseChannel(outSizeWrongDims);
            }

            // For NCEConvolutionOp and NCEEltwiseOp, conditionally skip slicing OC based on mode:
            //   SKIP_NONE: always enforce OC limit.
            //   SKIP_LARGE_SPATIAL: skip OC check when H or W > 4.
            //   SKIP_ALL: always skip OC check.
            // TODO: Fix all regressions (E#209583, E#209685, E#210083) to skip all OC checks
            const auto applySkipOCMode = [&](StringRef mode, bool isEltwise) {
                if (mode == "SKIP_NONE") {
                    return;
                }
                const bool ocIsInWrongDims = llvm::is_contained(outSizeWrongDims, Dims4D::Act::C);
                if (!ocIsInWrongDims) {
                    return;
                }
                constexpr int64_t kSpatialLimit = 4;
                const bool largeSpatial =
                        outputShape[Dims4D::Act::H] > kSpatialLimit || outputShape[Dims4D::Act::W] > kSpatialLimit;
                const bool doSkip = (mode == "SKIP_ALL") || (mode == "SKIP_LARGE_SPATIAL" && largeSpatial);
                if (doSkip) {
                    _log.nest(2).debug("Skip checking OC dimension for {0} at {1} (mode={2})", op->getName(),
                                       op->getLoc(), mode);
                    eraseChannel(outSizeWrongDims);
                    if (isEltwise) {
                        eraseChannel(inSizeWrongDims);
                    }
                }
            };
            if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
                applySkipOCMode(skipConvOC, /*isEltwise=*/false);
            } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
                applySkipOCMode(skipEltwiseOC, /*isEltwise=*/true);
            }

            return inSizeWrongDims.empty() && outSizeWrongDims.empty();
        }

        return true;
    });

    patterns.clear();
    patterns.add<EnsureNCEOpSizeRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createEnsureNCEOpsSizeRequirementsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEnsureNCEOpsSizeRequirementsPass(
        bool enableOutputEnsurance, bool enableDequantWeightEnsuranceBeforeStrategy, StringRef skipConvOC,
        StringRef skipEltwiseOC, Logger log) {
    return std::make_unique<EnsureNCEOpsSizeRequirementsPass>(
            enableOutputEnsurance, enableDequantWeightEnsuranceBeforeStrategy, skipConvOC, skipEltwiseOC, log);
}
