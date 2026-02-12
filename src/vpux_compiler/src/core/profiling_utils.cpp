//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/profiling_utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ScopedPrinter.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

using namespace vpux;

namespace {

std::string stringifyStorageType(mlir::Type type) {
    return llvm::TypeSwitch<mlir::Type, std::string>(type)
            .Case<mlir::MemRefType>([](const auto& ndType) {
                return llvm::to_string(ndType.getElementType());
            })
            .Case<mlir::quant::QuantizedType>([](const auto& quantizedType) {
                return llvm::to_string(quantizedType.getStorageType());
            })
            .Default([](const auto& otherType) {
                return llvm::to_string(otherType);
            });
}

// ActShave Tasks: we export every input and output and it is up to the end user to interpret them.
profiling::TensorShapeInfo getTensorInfo(vpux::ShapeRef shape, mlir::Type elementType) {
    auto dimVec = shape | transformed([](auto x) {
                      return static_cast<uint32_t>(x);
                  });
    profiling::TensorShapeInfo tensorShapeInfo;
    tensorShapeInfo.dimensions = std::vector<uint32_t>(dimVec.begin(), dimVec.end());
    tensorShapeInfo.elemType = stringifyStorageType(elementType);
    return tensorShapeInfo;
}

profiling::TensorShapeInfo getTensorInfoFromType(mlir::Type type) {
    auto ndType = mlir::cast<NDTypeInterface>(type);
    return getTensorInfo(ndType.getShape(), ndType.getElementType());
}

profiling::TensorShapeInfo getTensorInfo(const SmallVector<int64_t>& dims, mlir::Type elemType) {
    auto dimVec = dims | transformed([](auto x) {
                      return static_cast<uint32_t>(x);
                  });

    profiling::TensorShapeInfo tensorShapeInfo;
    tensorShapeInfo.dimensions = std::vector<uint32_t>(dimVec.begin(), dimVec.end());
    tensorShapeInfo.elemType = stringifyStorageType(elemType);
    return tensorShapeInfo;
}

}  // namespace

namespace vpux {

VariantInfoArray extractVariantInfoFromOp(VPUIP::NCEClusterTaskOp op) {
    auto& variantsRegion = op.getVariants();
    const auto dpuTasks = variantsRegion.getOps<VPUIP::DPUTaskOp>();
    const auto empty = mlir::ArrayAttr::get(op->getContext(), {});
    return VariantInfoArray(dpuTasks | transformed([&](VPUIP::DPUTaskOp dpuTask) {
                                auto inStartAttr = dpuTask.getInStart().value_or(empty);
                                auto inEndAttr = dpuTask.getInEnd().value_or(empty);
                                profiling::DPUVariantInfo variantInfo;
                                variantInfo.inStart = to_std_vector(parseIntArrayAttr<uint32_t>(inStartAttr));
                                variantInfo.inEnd = to_std_vector(parseIntArrayAttr<uint32_t>(inEndAttr));
                                variantInfo.outStart =
                                        to_std_vector(parseIntArrayAttr<uint32_t>(dpuTask.getOutStart()));
                                variantInfo.outEnd = to_std_vector(parseIntArrayAttr<uint32_t>(dpuTask.getOutEnd()));
                                return variantInfo;
                            }));
}

// DPU Tasks: we export only the input tensor and weights tensor (the latter if available).
profiling::TensorInfo extractTensorInfoFromOp(VPUIP::NCEClusterTaskOp op) {
    auto inputTypes = llvm::SmallVector{op.getInput().getType()};
    if (auto weights = op.getWeights()) {
        inputTypes.push_back(weights.getType());
    }
    profiling::TensorInfo tensorInfo;
    tensorInfo.inputs = to_std_vector(llvm::map_range(inputTypes, getTensorInfoFromType));
    tensorInfo.outputs = std::vector{getTensorInfoFromType(op.getOutput().getType())};
    return tensorInfo;
}

profiling::TensorInfo extractTensorInfoFromOp(VPUIP::SwKernelOp op) {
    auto inputTypes = op.getInputs() | transformed([](const mlir::Value& x) {
                          return getTensorInfoFromType(x.getType());
                      });
    auto outputTypes = op.getOutputs() | filtered([&op](const mlir::Value& x) -> bool {
                           // do not expose compiler profiling internals to the end user
                           return x != op.getProfilingData();
                       }) |
                       transformed([](const mlir::Value& x) -> profiling::TensorShapeInfo {
                           return getTensorInfoFromType(x.getType());
                       });
    profiling::TensorInfo tensorInfo;
    tensorInfo.inputs = to_std_vector(inputTypes);
    tensorInfo.outputs = to_std_vector(outputTypes);
    return tensorInfo;
}

std::pair<profiling::TensorInfo, profiling::TensorInfo> extractTensorInfoFromOp(VPUIP::DMATypeOpInterface op) {
    auto inType = mlir::cast<NDTypeInterface>(op.getInput().getType());
    auto outType = mlir::cast<NDTypeInterface>(op.getOutput().getType());

    auto inMemDims = to_small_vector(inType.getMemShape());
    auto inMemStrides = to_small_vector(inType.getMemStrides());
    auto inDmaTransaction = reduceDimsForDma(std::move(inMemDims), std::move(inMemStrides),
                                             inType.getElemTypeSize().count(), op->hasAttr(vpux::stridedInputAttrName));

    auto outMemDims = to_small_vector(outType.getMemShape());
    auto outMemStrides = to_small_vector(outType.getMemStrides());
    auto outDmaTransaction =
            reduceDimsForDma(std::move(outMemDims), std::move(outMemStrides), outType.getElemTypeSize().count(),
                             op->hasAttr(vpux::stridedOutputAttrName));

    auto u8Type = mlir::IntegerType::get(op->getContext(), 8, mlir::IntegerType::SignednessSemantics::Unsigned);

    profiling::TensorInfo tensorShapeInfo;
    tensorShapeInfo.inputs = {getTensorInfo(inDmaTransaction.dims, u8Type)};
    tensorShapeInfo.outputs = {getTensorInfo(outDmaTransaction.dims, u8Type)};

    profiling::TensorInfo tensorStrideInfo;
    tensorStrideInfo.inputs = {getTensorInfo(inDmaTransaction.strides, u8Type)};
    tensorStrideInfo.outputs = {getTensorInfo(outDmaTransaction.strides, u8Type)};

    return std::make_pair(tensorShapeInfo, tensorStrideInfo);
}

}  // namespace vpux
