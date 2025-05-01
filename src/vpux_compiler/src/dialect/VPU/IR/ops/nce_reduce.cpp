// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEReduceOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NCEReduceOpAdaptor reduce(operands, attrs, prop);
    if (mlir::failed(reduce.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduce.getInput();
    auto axes = parseIntArrayAttr<int64_t>(reduce.getAxesAttr());

    return VPU::inferReduceReturnTypes(loc, input, /*keep_dims*/ true, /*axes*/ axes, inferredReturnTypes,
                                       reduce.getInputPaddingAttr(), reduce.getOutputPaddingAttr());
}

mlir::LogicalResult vpux::VPU::NCEReduceOp::verify() {
    const auto op = getOperation();

    if (mlir::failed(IE::checkPadding(getInputPaddingAttr(), getInput().getType()))) {
        return errorAt(op, "Input padding {0} incompatible with input type {1}", getInputPaddingAttr(),
                       getInput().getType());
    }
    if (mlir::failed(IE::checkPadding(getOutputPaddingAttr(), getOutput().getType()))) {
        return errorAt(op, "Output padding {0} incompatible with output type {1}", getOutputPaddingAttr(),
                       getOutput().getType());
    }

    return mlir::success();
}

//
// isSupported
//

bool vpux::VPU::NCEReduceOp::isSupported(mlir::Operation* op, LogCb logCb, bool checkLayout,
                                         bool checkChannelAlignment) {
    if (!isReduceOpSupportedOnNCE(op) || !vpux::VPU::isNCEReduceSupported(op, logCb)) {
        return false;
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

    if (inputType.getRank() != 4 || outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (checkChannelAlignment) {
        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            if (!NCEInvariant::isInputActTypeSupported(inputType, iface.getInputChannelAlignment(), false) ||
                !NCEInvariant::isOutputActTypeSupported(outputType, iface.getOutputChannelAlignment())) {
                logCb(formatv("Misaligned tensor shape"));
                return false;
            }
        }
    }

    if (checkLayout) {
        if (!NCEInvariant::checkLayouts({inputType}, {outputType}, getArch(op), 1, logCb)) {
            return false;
        }
    }
    return true;
}
