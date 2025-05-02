//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"

namespace vpux::VPU {

//
// SCFTilingCommonModelOp
//

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingCommonModelOp : public mlir::TilingInterface::ExternalModel<ConcreteModel, ConcreteOp> {
protected:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        return static_cast<const ConcreteModel*>(this)->backInferSCFTileInfo(operation, builder, outputTile);
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, Dim dim, SCFShapeRef origShape,
                                          mlir::Operation* origOperation, ShapeRef tiling) const {
        return static_cast<const ConcreteModel*>(this)->createTiledOperation(opGenerator, operandsGenerator, builder,
                                                                             inputTiling, outputTile, dim, origShape,
                                                                             origOperation, tiling);
    }

    mlir::Value generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                             const SCFTileInfo& inputTileInfo) const {
        auto origType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());

        auto staticNewShape = mlir::getConstantIntValues(inputTileInfo.shape);
        if (origType.getShape().isStatic() && staticNewShape.has_value() &&
            llvm::equal(origType.getShape().raw(), staticNewShape.value())) {
            return origInput;
        }

        SmallVector<mlir::OpFoldResult> defaultStrides(inputTileInfo.offsets.size(), builder.getIndexAttr(1));

        auto extractTile = builder.create<mlir::tensor::ExtractSliceOp>(
                appendLoc(loc, "extractSlice"), origInput, inputTileInfo.offsets, inputTileInfo.shape, defaultStrides);

        auto newShape = getShape(extractTile.getResult());
        auto newType = origType.changeShape(ShapeRef(newShape));

        // by default output type loses NPU-specific attributes so we have to set it manually
        extractTile->getResult(0).setType(newType);

        return extractTile;
    }

    mlir::Type extractResultType(mlir::Type origType, SCFShapeRef newShape) const {
        auto ndTensorType = mlir::cast<vpux::NDTypeInterface>(origType);
        auto origElemType = ndTensorType.getElementType();

        VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(origElemType),
                        "Per axis quantized types are not supported in scf");

        const auto tensorDesc =
                vpux::getTensorAttr(origElemType.getContext(), ndTensorType.getDimsOrder(), ndTensorType.getMemSpace());
        SmallVector<mlir::Value> dynamicDims;  // unused cause for shape static dims are enough
        SmallVector<int64_t> staticDims;
        mlir::dispatchIndexOpFoldResults(newShape, dynamicDims, staticDims);
        return mlir::RankedTensorType::get(staticDims, origElemType, tensorDesc);
    }

public:
    SmallVector<mlir::Range> getIterationDomain(mlir::Operation* operation, mlir::OpBuilder& builder) const {
        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

        auto tilingDims = getNonOneDim(strategy);
        auto loc = operation->getLoc();

        auto tilingRank = tilingDims.size();
        SmallVector<mlir::Range> loops(tilingRank);
        mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

        const auto outputValue = operation->getResult(0);
        const auto outputType = mlir::cast<mlir::ShapedType>(outputValue.getType());

        const auto getDimValue = [&](int64_t dim) -> mlir::OpFoldResult {
            // E-162627 support dynamic shapes in upper bound calculation
            VPUX_THROW_WHEN(outputType.isDynamicDim(dim), "Dynamic case is not supported for tiling yet");
            return builder.getIndexAttr(outputType.getDimSize(dim));
        };

        for (auto dim : llvm::seq<int64_t>(0, tilingRank)) {
            loops[dim].offset = zero;
            loops[dim].size = getDimValue(tilingDims[dim].ind());
        }
        return loops;
    }

    mlir::FailureOr<mlir::TilingResult> getTiledImplementation(mlir::Operation* operation, mlir::OpBuilder& builder,
                                                               ArrayRef<mlir::OpFoldResult> offsets,
                                                               ArrayRef<mlir::OpFoldResult> sizes) const {
        SmallVector<mlir::OpFoldResult> resultOffsets;
        SmallVector<mlir::OpFoldResult> resultSizes;

        // E-162801 extend to multiple results
        unsigned resultNumber = 0;
        if (mlir::failed(getResultTilePosition(operation, builder, resultNumber, offsets, sizes, resultOffsets,
                                               resultSizes))) {
            return mlir::failure();
        }

        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getNonOneDim(strategy);

        SmallVector<mlir::Operation*> results;
        SmallVector<mlir::Value> resultValues;
        results.reserve(resultOffsets.size());
        resultValues.reserve(resultOffsets.size());
        SmallVector<mlir::OpFoldResult> axis(resultOffsets.size(), builder.getIndexAttr(1));
        auto origShape = mlir::getAsIndexOpFoldResult(
                builder.getContext(), mlir::cast<mlir::ShapedType>(operation->getResult(0).getType()).getShape());
        for (auto index : irange(tilingDims.size())) {
            axis[tilingDims[index].ind()] = builder.getIndexAttr(strategy[tilingDims[index]]);
            auto outputTile = SCFTileInfo(resultSizes, resultOffsets, axis);
            auto inputTiling = backInferSCFTileInfo(operation, builder, outputTile);
            SmallVector<mlir::Value> tiledOperands;
            tiledOperands.reserve(operation->getNumOperands());

            const OpTilingOperandsFunc createTiledOperands = [&](auto& tiling) {
                tiledOperands.clear();
                for (auto p : operation->getOperands() | indexed) {
                    auto origInput = p.value();
                    auto inputIdx = p.index();

                    if (tiling.size() <= inputIdx) {
                        tiledOperands.emplace_back(origInput);
                        continue;
                    }

                    auto inputTileInfo = tiling[inputIdx];
                    auto tiledInput = generateTile(operation->getLoc(), builder, origInput, inputTileInfo);

                    tiledOperands.emplace_back(tiledInput);
                }
            };

            const OpGeneratorFunc generatorFunc = [&]() {
                auto resultDenseTile = extractResultType(operation->getResult(0).getType(), resultSizes);

                auto* tiledOp = mlir::clone(builder, operation, {resultDenseTile}, tiledOperands);
                tiledOp->removeAttr(tilingStrategy);
                return tiledOp;
            };

            auto* resultOp = createTiledOperation(generatorFunc, createTiledOperands, builder, inputTiling, outputTile,
                                                  tilingDims[index], origShape, operation, strategy);

            results.emplace_back(resultOp);
            resultValues.emplace_back(resultOp->getResult(0));
        }

        return mlir::TilingResult{results, resultValues};
    }

    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(resultNumber).getType());

        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getNonOneDim(strategy);
        resultOffsets = SmallVector<mlir::OpFoldResult>(outputType.getRank(), builder.getIndexAttr(0));
        resultSizes.reserve(outputType.getRank());
        resultSizes = mlir::getAsIndexOpFoldResult(builder.getContext(), outputType.getShape());

        for (auto dimIndex : irange(tilingDims.size())) {
            resultOffsets[tilingDims[dimIndex].ind()] = offsets[dimIndex];
            resultSizes[tilingDims[dimIndex].ind()] = sizes[dimIndex];
        }

        return mlir::success();
    }
};

class SCFTilingEltwiseModelOp : public SCFTilingCommonModelOp<SCFTilingEltwiseModelOp, VPU::NCEEltwiseOp> {
public:
    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& tiling, const SCFTileInfo&, Dim, SCFShapeRef,
                                          mlir::Operation*, ShapeRef) const {
        operandsGenerator(tiling);
        return opGenerator();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        SmallVector<SCFTileInfo> inputTiles;
        for (auto origInput : operation->getOperands()) {
            const auto inputType = mlir::cast<mlir::ShapedType>(origInput.getType());
            const auto curShape = inputType.getShape();

            auto curTile = outputTile;
            for (auto ind : irange(curShape.size())) {
                if (curShape[ind] == 1) {
                    curTile.shape[ind] = builder.getIndexAttr(1);
                    curTile.offsets[ind] = builder.getIndexAttr(0);
                }
            }

            inputTiles.push_back(curTile);
        }
        return SCFTilingInfo{inputTiles};
    }
};

template <typename ConcreteOp>
class SCFTilingPoolingModelOp : public SCFTilingCommonModelOp<SCFTilingPoolingModelOp<ConcreteOp>, ConcreteOp> {
private:
    SCFTilingInfo backInferPoolTile(mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile,
                                    SCFShapeRef origInputShape, mlir::ArrayAttr kernelSize, mlir::ArrayAttr strides,
                                    const PadInfo& origPadding) const {
        SCFTileInfo inputTile(origInputShape, builder);

        auto axes = outputTile.axis;

        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];

        inputTile.shape[Dims4D::Act::C.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
        inputTile.offsets[Dims4D::Act::C.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];

        auto padMap = origPadding.toPadByDims();

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto dim = Dims4D::Act::getSpatialDim(index);
            if (mlir::isConstantIntValue(axes[dim.ind()], 1)) {
                continue;
            }

            const auto stride = mlir::cast<mlir::IntegerAttr>(strides[index]).getValue().getSExtValue();
            const auto kernel = mlir::cast<mlir::IntegerAttr>(kernelSize[index]).getValue().getSExtValue();

            mlir::Range inputRange =
                    solutionForOutputRange(loc, builder, outputTile, dim, kernel, stride, padMap[dim.ind()]);
            inputTile.offsets[dim.ind()] = inputRange.offset;
            inputTile.shape[dim.ind()] = inputRange.size;
        }

        return {inputTile};
    }

public:
    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, Dim dim, SCFShapeRef origShape,
                                          mlir::Operation* origOperation, ShapeRef tiling) const {
        return createTiledPaddedOperation<ConcreteOp>(opGenerator, operandsGenerator, builder, inputTiling, outputTile,
                                                      dim, origShape, origOperation, tiling);
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto poolingOperation = mlir::cast<ConcreteOp>(operation);
        const auto origInputShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(poolingOperation.getInput()).raw());
        const auto origPadding = toPadInfo(poolingOperation.getPad());

        auto inputTiling =
                backInferPoolTile(operation->getLoc(), builder, outputTile, origInputShape,
                                  poolingOperation.getKernelSize(), poolingOperation.getStrides(), origPadding);

        auto nceOp = mlir::dyn_cast<NCEOpInterface>(operation);
        if (nceOp != nullptr && nceOp.getWeightsTableOperand() != nullptr &&
            !mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1)) {
            inputTiling.emplace_back(
                    getWeightsTableSCFTile(nceOp.getWeightsTableOperand().getType(), builder, outputTile));
        }

        return inputTiling;
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingConvModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
protected:
    SCFTileInfo backInferConvInputTile(mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile,
                                       SCFShapeRef origInputShape, const std::array<int64_t, 2> kernel_size,
                                       mlir::ArrayAttr strides, const PadInfo& origPadding) const {
        SCFTileInfo inputTile(origInputShape, builder);

        auto axes = outputTile.axis;

        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];

        auto padMap = origPadding.toPadByDims();

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto dim = Dims4D::Act::getSpatialDim(index);
            if (mlir::isConstantIntValue(axes[dim.ind()], 1)) {
                continue;
            }

            const auto stride = strides[index].cast<mlir::IntegerAttr>().getValue().getSExtValue();
            const auto kernel = kernel_size[index];

            mlir::Range inputRange =
                    solutionForOutputRange(loc, builder, outputTile, dim, kernel, stride, padMap[dim.ind()]);

            inputTile.offsets[dim.ind()] = inputRange.offset;
            inputTile.shape[dim.ind()] = inputRange.size;
        }

        return inputTile;
    }

public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto convOperation = mlir::cast<ConcreteOp>(operation);
        const auto origInputShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(convOperation.getInput()).raw());
        const auto origFilterShape = getShape(convOperation.getFilter());
        const auto origPadding = toPadInfo(convOperation.getPad());

        const std::array<int64_t, 2> kernel = {origFilterShape[Dims4D::Filter::KX],
                                               origFilterShape[Dims4D::Filter::KY]};

        auto inputTileTiling = backInferConvInputTile(operation->getLoc(), builder, outputTile, origInputShape, kernel,
                                                      convOperation.getStrides(), origPadding);
        SCFTilingInfo tilingInfo = {inputTileTiling};

        const auto tileOverChannels = !mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1);

        if (tileOverChannels) {
            SCFTileInfo filterTile(origFilterShape, builder);

            filterTile.shape[Dims4D::Filter::OC.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
            filterTile.offsets[Dims4D::Filter::OC.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];

            tilingInfo.emplace_back(filterTile);

            auto nceOp = mlir::dyn_cast<NCEOpInterface>(operation);
            if (nceOp != nullptr && nceOp.getWeightsTableOperand() != nullptr) {
                tilingInfo.emplace_back(
                        getWeightsTableSCFTile(nceOp.getWeightsTableOperand().getType(), builder, outputTile));
            }
        }

        return tilingInfo;
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, Dim dim, SCFShapeRef origShape,
                                          mlir::Operation* origOperation, ShapeRef tiling) const {
        auto generator = opGenerator;
        auto newChannelValue = mlir::getConstantIntValue(outputTile.shape[Dims4D::Act::C.ind()]);
        if (!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1) && newChannelValue.has_value()) {
            const OpGeneratorFunc adjustFilterGenerator = [&]() -> mlir::Operation* {
                auto newOperation = mlir::cast<ConcreteOp>(opGenerator());
                auto newRawFilterShape = Shape(parseIntArrayAttr<int64_t>(newOperation.getRawFilterShape()));
                newRawFilterShape[Dims4D::Filter::OC] = newChannelValue.value();
                newOperation.setRawFilterShapeAttr(getIntArrayAttr(newOperation->getContext(), newRawFilterShape));

                return newOperation.getOperation();
            };

            generator = adjustFilterGenerator;
        }
        return createTiledPaddedOperation<ConcreteOp>(generator, operandsGenerator, builder, inputTiling, outputTile,
                                                      dim, origShape, origOperation, tiling);
    }
};

class SCFConvOpModel : public SCFTilingConvModelOp<SCFConvOpModel, NCEConvolutionOp> {};

class SCFTilingDepthConvModelOp : public SCFTilingConvModelOp<SCFTilingDepthConvModelOp, NCEDepthConvolutionOp> {
public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto inputConvTiling =
                SCFTilingConvModelOp<SCFTilingDepthConvModelOp, NCEDepthConvolutionOp>::backInferSCFTileInfo(
                        operation, builder, outputTile);

        const auto tileOverChannels = !mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1);

        if (tileOverChannels) {
            auto depthOperation = mlir::cast<NCEDepthConvolutionOp>(operation);
            const auto origFilterShape = Shape(parseIntArrayAttr<int64_t>(depthOperation.getRawFilterShape()));
            const auto origInputShape = getShape(depthOperation.getInput());
            auto& inputTiles = inputConvTiling[0];

            mlir::AffineExpr d0;
            bindDims(builder.getContext(), d0);

            const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / origInputShape[Dims4D::Act::C];
            auto loc = operation->getLoc();

            auto groupMap = mlir::AffineMap::get(
                    1, 0, {d0.floorDiv(numOutChannelsPerGroup) * origFilterShape[Dims4D::Filter::IC]},
                    builder.getContext());
            inputTiles.offsets[Dims4D::Act::C.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, loc, groupMap, {outputTile.offsets[Dims4D::Act::C.ind()]});

            inputTiles.shape[Dims4D::Act::C.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, loc, groupMap, {outputTile.shape[Dims4D::Act::C.ind()]});
        }

        return inputConvTiling;
    }
};

}  // namespace vpux::VPU
