//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/interpolate_bound.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::InterpolateDMAOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::InterpolateDMAOpAdaptor adaptor(operands, attrs, prop);
    if (mlir::failed(adaptor.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(adaptor.getInput().getType());
    const auto inShape = getBoundedShape(adaptor.getInput());
    const auto axesVal = parseIntArrayAttr<int64_t>(adaptor.getAxesAttr());
    const auto beginPads = IE::extractIntVector(loc, nullptr, adaptor.getAttr().getPadsBegin());
    const auto endPads = IE::extractIntVector(loc, nullptr, adaptor.getAttr().getPadsEnd());

    // Scales are runtime parameters — use bounded scales for compile-time shape inference
    auto scalesBound = SmallVector<double>(axesVal.size(), 1.0);
    scalesBound[scalesBound.size() - 1] = INTERPOLATE_SCALES_BOUND;
    scalesBound[scalesBound.size() - 2] = INTERPOLATE_SCALES_BOUND;

    const auto scalesElemType = mlir::cast<vpux::NDTypeInterface>(adaptor.getScales().getType()).getElementType();

    const auto outShapeVec =
            IE::inferInterpOutShape(loc, axesVal, inShape, beginPads, endPads, IE::InterpolateCalcMode::SCALES,
                                    mlir::FailureOr<ArrayRef<int64_t>>(mlir::failure()), ArrayRef<double>(scalesBound),
                                    scalesElemType, Logger::global());

    auto [outDesc, outShape] = callOnShapeOf(inputType, [&](const auto& shape) {
        using ShapeT = std::decay_t<decltype(shape)>;
        if constexpr (std::is_same_v<ShapeT, BoundedShape>) {
            auto desc =
                    vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), BoundsRef(outShapeVec));
            // Interpolated axes are always dynamic; non-interpolated axes preserve input dynamism
            auto staticShape = outShapeVec;
            for (const auto& axis : axesVal) {
                staticShape[axis] = mlir::ShapedType::kDynamic;
            }
            for (int64_t i = 0; i < inputType.getRank(); ++i) {
                if (llvm::find(axesVal, i) == axesVal.end() &&
                    inputType.getShape()[Dim(i)] == mlir::ShapedType::kDynamic) {
                    staticShape[i] = mlir::ShapedType::kDynamic;
                }
            }
            return std::make_pair(desc, staticShape);
        } else if constexpr (std::is_same_v<ShapeT, DimsMaskedShape>) {
            auto mask = mlir::cast<Core::DynamicDimsMaskTensorType>(inputType).getDynamicDimsMask();
            auto outMask = SmallVector<int64_t>(mask.begin(), mask.end());
            for (const auto& axis : axesVal) {
                outMask[axis] = 1;
            }
            auto desc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), {},
                                            DynamicDimsMaskRef(outMask));
            return std::make_pair(desc, outShapeVec);
        } else {
            // Static input — use assignDynamicTypeComponents via bounds_representation
            const auto outDynShape = Shape(SmallVector<int64_t>(inputType.getRank(), mlir::ShapedType::kDynamic));
            auto typeComponents =
                    TypeComponents().setDimsOrder(inputType.getDimsOrder()).setElementType(inputType.getElementType());
            assignDynamicTypeComponents(typeComponents, adaptor.getBoundsRepresentation(), outDynShape.raw(),
                                        outShapeVec);
            auto outType = inputType.changeTypeComponents(typeComponents);
            return std::make_pair(
                    mlir::dyn_cast_or_null<vpux::TensorAttr>(mlir::cast<mlir::RankedTensorType>(outType).getEncoding()),
                    SmallVector<int64_t>(mlir::cast<mlir::ShapedType>(outType).getShape()));
        }
    });

    auto outputType = mlir::RankedTensorType::get(outShape, inputType.getElementType(), outDesc);
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// SWOpInterface
//

bool vpux::VPU::InterpolateDMAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    // This op uses DDR buffers exclusively.
    // The SHAVE kernel will manage its own DMA transfers from DDR to CMX in runtime.
    VPUX_UNUSED(buffers);
    VPUX_UNUSED(reservedMem);
    return false;
}

bool vpux::VPU::InterpolateDMAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::InterpolateDMAOp::supportCycleCostCalculation() {
    return false;
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::InterpolateDMAOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto loc = getLoc();
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    const auto axesVal = parseIntArrayAttr<int64_t>(getAxesAttr());

    return reifyInterpolateResultShape(builder, loc, getInput(), getScales(), std::nullopt, axesVal, outputShapedType,
                                       reifiedReturnShapes);
}
