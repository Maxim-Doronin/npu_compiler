//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

bool hasAutoPaddingOption(mlir::ModuleOp module, StringRef paddingMode) {
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch auto padding mode");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(paddingMode);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find config.OptionOp to fetch auto padding mode");
        return false;
    }
    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attrValue.getOptionValue());
    if (boolAttr == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to cast config.OptionOp to BoolAttr");
        return false;
    }
    return boolAttr.getValue();
}

bool VPU::hasAutoPadding(mlir::ModuleOp module) {
    return hasAutoPaddingOption(module, AUTO_PADDING_IDU) || hasAutoPaddingOption(module, AUTO_PADDING_ODU);
}

bool VPU::hasAutoPaddingIDU(mlir::ModuleOp module) {
    return hasAutoPaddingOption(module, AUTO_PADDING_IDU);
}

bool VPU::hasAutoPaddingODU(mlir::ModuleOp module) {
    return hasAutoPaddingOption(module, AUTO_PADDING_ODU);
}

bool VPU::areChannelsCompatibleWithIDUAutoPad(int64_t inputChannels, int64_t elemTypeBitWidth) {
    return elemTypeBitWidth >= CHAR_BIT &&
           ((elemTypeBitWidth < FP16_WIDTH && inputChannels < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) ||
            (elemTypeBitWidth >= FP16_WIDTH && inputChannels < WIDTH16_CHANNEL_LIMIT));
}

bool VPU::areChannelsCompatibleWithODUAutoPad(int64_t outputChannels, int64_t elemTypeBitWidth) {
    return outputChannels < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT && elemTypeBitWidth >= CHAR_BIT;
}

bool VPU::inputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    if (type.getRank() != 4) {
        return false;
    }
    const auto inShape = type.getShape();
    const auto elemTypeBitWidth = type.getElemTypeSize().count();
    return areChannelsCompatibleWithIDUAutoPad(inShape[Dims4D::Act::C], elemTypeBitWidth);
}

bool VPU::outputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    if (type.getRank() != 4) {
        return false;
    }
    const auto outShape = type.getShape();
    const auto elemTypeBitWidth = type.getElemTypeSize().count();
    return areChannelsCompatibleWithODUAutoPad(outShape[Dims4D::Act::C], elemTypeBitWidth);
}

bool VPU::canAutopadInput(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<VPU::NCEConvolutionOp>(op)) {
        return false;
    }
    const auto inputType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
    return inputCompatibleWithAutoPad(inputType) && hasAutoPaddingIDU(getModuleOp(op));
}

bool VPU::canAutopadOutput(mlir::Operation* op, std::optional<vpux::NDTypeInterface> optType) {
    const auto outputType = optType.value_or(mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()));
    return outputCompatibleWithAutoPad(outputType) && hasAutoPaddingODU(getModuleOp(op));
}
