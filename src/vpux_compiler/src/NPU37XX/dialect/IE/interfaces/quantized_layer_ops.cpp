//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"

using namespace vpux;
using namespace IE;

namespace {

//
// QuantizedLayerOpModel for arch37xx
//

template <typename OpType>
class QuantizedLayerOpModel37XX :
        public IE::QuantizedLayerOpInterface::ExternalModel<QuantizedLayerOpModel37XX<OpType>, OpType> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        vpux::Logger log = vpux::Logger::global();
        return IE::arch37xx::isMixPrecisionSupported(op, isPReLUSupported, log);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (layerWithPostOp == nullptr) {
            return true;
        }
        return IE::arch37xx::checkPostOp(layerWithPostOp, isPerAxisQuantizedOutput, isFloatInput);
    }
};

}  // namespace

void vpux::IE::arch37xx::registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        // Register the interface for operations that support mixed precision and can be lowered to NCE
        IE::ConvolutionOp::attachInterface<QuantizedLayerOpModel37XX<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<QuantizedLayerOpModel37XX<IE::GroupConvolutionOp>>(*ctx);
        IE::AddOp::attachInterface<QuantizedLayerOpModel37XX<IE::AddOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<QuantizedLayerOpModel37XX<IE::AvgPoolOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<QuantizedLayerOpModel37XX<IE::TransposedConvolutionOp>>(*ctx);
        IE::MatMulOp::attachInterface<QuantizedLayerOpModel37XX<IE::MatMulOp>>(*ctx);
    });
}
