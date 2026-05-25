//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/bounded_buffer_pass_utils.hpp"

#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <intel_npu/prefix.hpp>

#include <mlir/IR/Builders.h>

#include <iterator>

using namespace vpux;

vpux::VPUIP::BoundedBufferComponents vpux::VPUIP::unpackBoundedBufferType(VPUIP::BoundedBufferType type) {
    // TODO: support for other buffer types will be added separately
    // Track E#118173
    return {mlir::cast<mlir::MemRefType>(type.getData()), mlir::cast<mlir::MemRefType>(type.getDynamicShape())};
}

void vpux::VPUIP::addShapeTensorDataInfo(mlir::func::FuncOp funcOp, mlir::MemRefType dynamicShapeMemRef,
                                         mlir::Block& infoBlock, mlir::StringRef dataInfoName, size_t dataBufferCount) {
    const auto type = mlir::RankedTensorType::get(dynamicShapeMemRef.getShape(), dynamicShapeMemRef.getElementType());
    const auto name = std::string(intel_npu::SHAPE_TENSOR_PREFIX) + dataInfoName.str();

    auto insertionPointAfter = std::next(infoBlock.begin(), static_cast<int64_t>(dataBufferCount));
    auto infoBuilder = mlir::OpBuilder(&infoBlock, insertionPointAfter);
    infoBuilder.create<net::DataInfoOp>(takeOpLoc(funcOp, "{0}", funcOp.getName()), name, type);
}
