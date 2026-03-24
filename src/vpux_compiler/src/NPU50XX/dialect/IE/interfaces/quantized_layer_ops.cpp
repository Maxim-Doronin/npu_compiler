//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"

using namespace vpux;
using namespace IE;

namespace {

//
// QuantizedLayerOpModel for arch50xx
//

template <typename OpType>
class QuantizedLayerOpModel50XX :
        public IE::QuantizedLayerOpInterface::ExternalModel<QuantizedLayerOpModel50XX<OpType>, OpType> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        vpux::Logger log = vpux::Logger::global();
        return IE::arch50xx::isMixPrecisionSupported(op, isPReLUSupported, log);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (!layerWithPostOp) {
            return true;
        }
        return IE::arch50xx::checkPostOp(layerWithPostOp, isPerAxisQuantizedOutput, isFloatInput);
    }
};

}  // namespace

void vpux::IE::arch50xx::registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        // Register the interface for operations that support mixed precision and can be lowered to NCE
        // Note: arch50xx checks for LayerWithPostOpInterface in isMixPrecisionSupported, so we register
        // for all operations that have that interface
        IE::ConvolutionOp::attachInterface<QuantizedLayerOpModel50XX<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<QuantizedLayerOpModel50XX<IE::GroupConvolutionOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<QuantizedLayerOpModel50XX<IE::TransposedConvolutionOp>>(*ctx);
        IE::GroupTransposedConvolutionOp::attachInterface<QuantizedLayerOpModel50XX<IE::GroupTransposedConvolutionOp>>(
                *ctx);
        IE::AddOp::attachInterface<QuantizedLayerOpModel50XX<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<QuantizedLayerOpModel50XX<IE::MultiplyOp>>(*ctx);
        IE::SubtractOp::attachInterface<QuantizedLayerOpModel50XX<IE::SubtractOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<QuantizedLayerOpModel50XX<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<QuantizedLayerOpModel50XX<IE::AvgPoolOp>>(*ctx);
        IE::MatMulOp::attachInterface<QuantizedLayerOpModel50XX<IE::MatMulOp>>(*ctx);
    });
}
