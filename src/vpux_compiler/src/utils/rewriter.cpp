//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/profiling/location.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// updateFunctionSignature
//

mlir::LogicalResult vpux::updateFunctionSignature(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                                  ArrayRef<mlir::Type> newResultTypes, Logger log) {
    const auto origFuncType = funcOp.getFunctionType();

    if (newArgTypes.size() != origFuncType.getNumInputs()) {
        log.trace("New inputs size '{0}' doesn't match original prototype", newArgTypes.size());
        return mlir::failure();
    }
    if (newResultTypes.size() != origFuncType.getNumResults()) {
        log.trace("New results size '{0}' doesn't match original prototype", newResultTypes.size());
        return mlir::failure();
    }

    const auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);

    if (newFuncType == origFuncType) {
        log.trace("Nothing to change");
        return mlir::success();
    }

    log.trace("Update Function signature : '{0}' -> '{1}'", origFuncType, newFuncType);
    funcOp.setType(newFuncType);

    return mlir::success();
}

namespace {

// tensors -> buffers

mlir::BaseMemRefType tensorWithBoundsToBoundedBuffer(mlir::RankedTensorType tensorType) {
    VPUX_THROW_UNLESS(mlir::isa<Core::DynamicDimsMaskTensorType>(tensorType),
                      "Expected to have dynamic tensor, got {0}", tensorType);

    const auto ndType = mlir::cast<vpux::NDTypeInterface>(tensorType);

    const auto dataMemShape = ndType.getShape();
    const auto dataMemOrder = ndType.getDimsOrder();
    const auto dataMemType = ndType.getElementType();
    const auto dataMemSpace = ndType.getMemSpace();
    const auto dataMemRef = getMemRefType(dataMemShape, dataMemType, dataMemOrder, dataMemSpace);

    const auto rank = checked_cast<int32_t>(dataMemShape.size());
    const auto si32 = getSInt32Type(tensorType.getContext());
    const auto dynamicShapeMemRef = getMemRefType({rank}, si32, DimsOrder::C, dataMemSpace);

    return VPUIP::BoundedBufferType::get(dataMemRef, dynamicShapeMemRef);
}

mlir::BaseMemRefType tensorToBuffer(mlir::RankedTensorType tensorType) {
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(tensorType)) {
        return tensorWithBoundsToBoundedBuffer(tensorType);
    }
    const auto type = mlir::cast<vpux::NDTypeInterface>(tensorType);
    const auto shape = type.getShape();
    const auto elemType = type.getElementType();
    const auto order = type.getDimsOrder();
    const auto memSpace = type.getMemSpace();
    return getMemRefType(shape, elemType, order, memSpace);
}

mlir::BaseMemRefType distributedTensorToBuffer(VPU::DistributedTensorType type) {
    mlir::MLIRContext* ctx = type.getContext();
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(type)) {
        const auto data =
                VPUIP::DistributedBufferType::get(ctx, type.getShape().raw(), type.getElementType(), type.getOrder(),
                                                  type.getMemSpace(), type.getDistribution());
        const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

        const auto dataMemShape = ndType.getShape();
        const auto dataMemSpace = ndType.getMemSpace();
        const auto rank = checked_cast<int32_t>(dataMemShape.size());
        const auto si32 = getSInt32Type(ctx);

        const DimsOrder order = DimsOrder::C;
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto memRefLayoutAttr = vpux::MemRefAttr::get(orderAttr, nullptr, nullptr, ctx);

        const auto duplicatedMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        auto shapeDistribution = VPU::DistributionInfoAttr::get(ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr,
                                                                type.getDistribution().getNumClusters(), nullptr,
                                                                type.getDistribution().getUniformDistributedSegments(),
                                                                nullptr, nullptr, nullptr, nullptr, nullptr);

        const auto dynamicShapeMemRef =
                VPUIP::DistributedBufferType::get(ctx, {rank}, si32, memRefLayoutAttr, dataMemSpace, shapeDistribution);

        return VPUIP::BoundedBufferType::get(data, dynamicShapeMemRef);
    }

    return VPUIP::DistributedBufferType::get(ctx, type.getShape().raw(), type.getElementType(), type.getOrder(),
                                             type.getMemSpace(), type.getDistribution());
}

mlir::Type bufferizeTensor(mlir::Type tensorType) {
    if (tensorType == nullptr) {
        return nullptr;
    }

    if (auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tensorType)) {
        return distributedTensorToBuffer(distributedType);
    } else if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(tensorType)) {
        return tensorToBuffer(rankedType);
    }
    VPUX_THROW("Unsupported type for bufferization '{0}'", tensorType);
    return nullptr;
}

VPUIP::SparseBufferType sparseTensorToBuffer(VPU::SparseTensorType type) {
    const auto data = bufferizeTensor(type.getData());
    const auto sparsityMap = bufferizeTensor(type.getSparsityMap());
    const auto storageElementTable = bufferizeTensor(type.getStorageElementTable());
    const auto seAttr = type.getSeAttr();

    VPUIP::SparsityCompressionAttr sparsityCompression = nullptr;
    if (auto origCompression = type.getSparsityCompression()) {
        sparsityCompression =
                VPUIP::SparsityCompressionAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                    origCompression.getNumElems(), origCompression.getAlignment());
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, type.getIsWeights(),
                                        sparsityCompression, seAttr);
}

// tensors <- buffers

mlir::RankedTensorType bufferToTensor(mlir::MemRefType memrefType) {
    const auto type = mlir::cast<NDTypeInterface>(memrefType);
    const auto encoding = getTensorAttr(type.getContext(), type.getDimsOrder(), type.getMemSpace());
    return mlir::RankedTensorType::get(type.getShape().raw(), type.getElementType(), encoding);
}

VPU::DistributedTensorType bufferToTensor(VPUIP::DistributedBufferType type) {
    // Note: distributed tensor is special: during bufferization its layout is
    // always an affine map (never vpux::MemRefAttr)
    const auto order = mlir::AffineMapAttr::get(type.getLayout().getAffineMap());
    return VPU::DistributedTensorType::get(type.getContext(), type.getShape().raw(), type.getElementType(), order,
                                           type.getMemSpace(), type.getDistribution());
}

mlir::Type tensorizeBuffer(mlir::Type bufferType) {
    if (bufferType == nullptr) {
        return nullptr;
    }

    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(bufferType)) {
        return bufferToTensor(distributedType);
    } else if (auto rankedType = mlir::dyn_cast<mlir::MemRefType>(bufferType)) {
        return bufferToTensor(rankedType);
    }
    VPUX_THROW("Unsupported buffer type for tensor conversion '{0}'", bufferType);
    return nullptr;
}

VPU::SparseTensorType bufferToTensor(VPUIP::SparseBufferType type) {
    const auto data = tensorizeBuffer(type.getData());
    const auto sparsityMap = tensorizeBuffer(type.getSparsityMap());
    const auto storageElementTable = tensorizeBuffer(type.getStorageElementTable());
    const auto seAttr = type.getSeAttr();

    VPU::SparsityCompressionAttr sparsityCompression = nullptr;
    if (auto origCompression = type.getSparsityCompression()) {
        sparsityCompression =
                VPU::SparsityCompressionAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                  origCompression.getNumElems(), origCompression.getAlignment());
    }

    return VPU::SparseTensorType::get(data, sparsityMap, storageElementTable, type.getIsWeights(), sparsityCompression,
                                      seAttr);
}

mlir::RankedTensorType bufferToTensor(VPUIP::BoundedBufferType type) {
    auto dataType = tensorizeBuffer(type.getData());
    auto ndType = mlir::cast<NDTypeInterface>(dataType);
    auto dimsMask = SmallVector<int64_t>(ndType.getShape().size(), 1);
    dataType = Core::DynamicDimsMaskTensorType::get(ndType, dimsMask);
    return mlir::cast<mlir::RankedTensorType>(dataType);
}

mlir::Value materializeToTensor(mlir::OpBuilder& builder, mlir::TensorType type, mlir::ValueRange inputs,
                                mlir::Location loc) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "Expected only one input");
    VPUX_THROW_UNLESS(mlir::isa<mlir::BaseMemRefType>(inputs[0].getType()), "Expected input type is BaseMemRefType");
    return builder.create<mlir::bufferization::ToTensorOp>(loc, type, inputs[0]);
};

mlir::Value materializeToMemref(mlir::OpBuilder& builder, mlir::BaseMemRefType type, mlir::ValueRange inputs,
                                mlir::Location loc) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "Expected only one input");
    VPUX_THROW_UNLESS(mlir::isa<mlir::TensorType>(inputs[0].getType()), "Expected input type is TensorType");
    return builder.create<mlir::bufferization::ToMemrefOp>(loc, type, inputs[0]);
};

}  // namespace

//
// convertFunc
//

mlir::LogicalResult vpux::convertFunc(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                      ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder, Logger log) {
    log.trace("Convert Function '@{0}' prototype", funcOp.getSymName());
    log = log.nest();

    if (funcOp.isExternal()) {
        log.trace("Can't convert external Function '@{0}'", funcOp.getSymName());
        return mlir::failure();
    }

    if (updateFunctionSignature(funcOp, newArgTypes, newResultTypes, log).failed()) {
        return mlir::failure();
    }

    //
    // Convert arguments
    //

    log.trace("Convert arguments");

    for (const auto& p : funcOp.getArguments() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());
        auto val = p.value();

        log.nest().trace("Process argument #{0}", ind);

        const auto origType = mlir::cast<vpux::NDTypeInterface>(val.getType());
        const auto newType = newArgTypes[ind];

        if (newType == origType) {
            log.nest(2).trace("Nothing to change");
            continue;
        }

        log.nest(2).trace("Convert the argument type : '{0}' -> '{1}'", origType, newType);

        val.setType(newType);

        auto* firstUser = getFirstUser(val);
        if (firstUser == nullptr) {
            log.nest(2).trace("The argument has no users");
            continue;
        }

        OpBuilderLogger builderLog(log.nest(2));
        mlir::OpBuilder argBuilder(firstUser, &builderLog);

        auto* cvtOp = cvtOpBuilder(argBuilder, IE::getValueLocation(val), val, origType);

        val.replaceAllUsesExcept(cvtOp->getResult(0), llvm::SmallPtrSet<mlir::Operation*, 1>{cvtOp});
    }

    //
    // Convert results
    //

    log.trace("Convert results");
    auto moduleOp = getModuleOp(funcOp);
    auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
    SmallVector<net::DataInfoOp> outputsInfo;
    if (netInfoOps.size() == 1) {
        outputsInfo = to_small_vector(netInfoOps.front().getOutputsInfo().getOps<net::DataInfoOp>());
    } else {
        log.warning("Can't get location for output. If it isn't a test, please, debug this.");
    }

    funcOp.walk([&](mlir::func::ReturnOp retOp) {
        log.nest().trace("Process return Operation '{0}'", retOp.getLoc());

        OpBuilderLogger builderLog(log.nest(3));
        mlir::OpBuilder resBuilder(retOp, &builderLog);

        for (const auto& p : retOp->getOperands() | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());
            auto val = p.value();

            log.nest(2).trace("Process result #{0}", ind);

            const auto origType = val.getType();
            const auto newType = mlir::cast<vpux::NDTypeInterface>(newResultTypes[ind]);

            if (newType == origType) {
                log.nest(3).trace("Nothing to change");
                continue;
            }

            log.nest(3).trace("Convert the result type : '{0}' -> '{1}'", newType, origType);

            mlir::Location cvtLoc = mlir::UnknownLoc::get(retOp.getContext());
            if (outputsInfo.empty()) {
                cvtLoc = appendLoc(IE::getValueLocation(val), "out_{0}", p.index());
            } else {
                cvtLoc = outputsInfo[p.index()]->getLoc();
            }
            auto* cvtOp = cvtOpBuilder(resBuilder, cvtLoc, val, newType);

            retOp.setOperand(ind, cvtOp->getResult(0));
        }
    });

    return mlir::success();
}

//
// getDefaultGreedyRewriteConfig
//

mlir::GreedyRewriteConfig vpux::getDefaultGreedyRewriteConfig() {
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Normal;
    config.maxIterations = 10;
    return config;
}

//
// appendLoc
//

mlir::Location vpux::appendLoc(mlir::Location baseLoc, StringRef suffix) {
    const auto suffixIdentifier = mlir::StringAttr::get(baseLoc.getContext(), suffix);
    return appendLoc(baseLoc, suffixIdentifier);
}

mlir::Location vpux::appendLoc(mlir::Location baseLoc, const formatv_object_base& suffix) {
    const auto suffixIdentifier = mlir::StringAttr::get(baseLoc.getContext(), suffix);
    return appendLoc(baseLoc, suffixIdentifier);
}

mlir::Location vpux::appendLoc(mlir::Location baseLoc, mlir::StringAttr suffix) {
    VPUX_THROW_WHEN(suffix.getValue().find(LOCATION_ORIGIN_SEPARATOR) != std::string::npos,
                    "'{0}' character is reserved inside locations", LOCATION_ORIGIN_SEPARATOR);
    const mlir::Location suffixLoc = mlir::NameLoc::get(suffix);
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(baseLoc)) {
        const auto metadata = fusedLoc.getMetadata();
        auto locations = fusedLoc.getLocations().vec();
        locations.push_back(suffixLoc);
        return mlir::FusedLoc::get(baseLoc.getContext(), locations, metadata);
    }
    return mlir::FusedLoc::get(baseLoc.getContext(), {baseLoc, suffixLoc});
}

//
// BufferizeTypeConverterBase
//

vpux::BufferizeTypeConverterBase::BufferizeTypeConverterBase() {
    addConversion([](mlir::Type type) {
        return type;
    });

    addConversion(tensorToBuffer);
    addConversion(distributedTensorToBuffer);
    addConversion(sparseTensorToBuffer);
}

//
// BufferizeTypeConverter
//

vpux::BufferizeTypeConverter::BufferizeTypeConverter() {
    addTargetMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addArgumentMaterialization(dummyConverter<mlir::BaseMemRefType>);
    addSourceMaterialization(dummyConverter<mlir::TensorType>);
}

//
// BufferizeOneShotTypeConverter
//

vpux::BufferizeOneShotTypeConverter::BufferizeOneShotTypeConverter() {
    addArgumentMaterialization(materializeToTensor);
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization(materializeToMemref);
}

namespace {
// NPU compiler's wrapper around preferred unknown type bufferization function
mlir::BaseMemRefType getMemRefTypeForUnknownTensorType(mlir::Type type, mlir::Attribute memorySpace) {
    auto tensorType = mlir::cast<mlir::TensorType>(type);
    return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace);
}
}  // namespace

mlir::bufferization::OneShotBufferizationOptions vpux::getOneShotBufferizationOptions() {
    mlir::bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.allowUnknownOps = true;
    options.copyBeforeWrite = false;
    // E#118032: Setting testAnalysisOnly as true does not introduce `bufferization.alloc_tensor`
    // but only adding `__inplace_operands_attr__`. Need to investigate whether it could be set to false,
    // and eliminate the introduced `bufferization.alloc_tensor, which will reduce the insertion of VPUIP.Copy
    options.testAnalysisOnly = true;
    options.unknownTypeConverterFn = [](mlir::Value value, mlir::Attribute memorySpace,
                                        const mlir::bufferization::BufferizationOptions& /*options*/) {
        return getMemRefTypeForUnknownTensorType(value.getType(), memorySpace);
    };
    options.opFilter.allowDialect<mlir::bufferization::BufferizationDialect, mlir::memref::MemRefDialect,
                                  mlir::func::FuncDialect, VPU::VPUDialect, Const::ConstDialect,
                                  mlir::linalg::LinalgDialect, Core::CoreDialect>();

    return options;
}

//
// getBufferType
//

vpux::NDTypeInterface vpux::getBufferType(mlir::Type tensorType) {
    const bool isAlreadyABufferType = isBufferType(tensorType);
    if (isAlreadyABufferType) {
        return mlir::cast<vpux::NDTypeInterface>(tensorType);
    }

    return llvm::TypeSwitch<mlir::Type, mlir::Type>(tensorType)
            .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType rankedTensorType) {
                return tensorToBuffer(rankedTensorType);
            })
            .Case<VPU::DistributedTensorType>([&](VPU::DistributedTensorType distributedTensorType) {
                return distributedTensorToBuffer(distributedTensorType);
            })
            .Case<VPU::SparseTensorType>([&](VPU::SparseTensorType sparseTensorType) {
                return sparseTensorToBuffer(sparseTensorType);
            })
            .Default([&](mlir::Type type) {
                // this is likely an unranked tensor type and we don't know how
                // to get a memSpace for it.
                const mlir::Attribute unknownMemSpace = nullptr;
                // E#108407: use UnknownTypeConverterFn directly, once it
                // accepts mlir::Type
                return getMemRefTypeForUnknownTensorType(type, unknownMemSpace);
            });
}

vpux::NDTypeInterface vpux::getBufferType(mlir::Value value) {
    return vpux::getBufferType(value.getType());
}

mlir::Type vpux::reconstructTensorType(mlir::Type type) {
    VPUX_THROW_UNLESS(isBufferType(type), "Provided a non-buffer type: {0}", type);

    // Note: enumerates types that are of interest to NPU compiler
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
            .Case<mlir::MemRefType>([](mlir::MemRefType memref) {
                return bufferToTensor(memref);
            })
            .Case<mlir::UnrankedMemRefType>([](mlir::UnrankedMemRefType memref) {
                return mlir::UnrankedTensorType::get(memref.getElementType());
            })
            .Case<VPUIP::DistributedBufferType>([](VPUIP::DistributedBufferType memref) {
                return bufferToTensor(memref);
            })
            .Case<VPUIP::SparseBufferType>([](VPUIP::SparseBufferType memref) {
                return bufferToTensor(memref);
            })
            .Case<VPUIP::BoundedBufferType>([](VPUIP::BoundedBufferType memref) {
                return bufferToTensor(memref);
            })
            .Default([](mlir::Type memref) {
                VPUX_THROW("Unknown memref type: {0}", memref);
                return mlir::NoneType::get(memref.getContext());
            });
}

//
// getBuffer
//

mlir::Value vpux::getBuffer(mlir::RewriterBase& rewriter, mlir::Value value) {
    if (auto toTensorOp = value.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
        return toTensorOp.getMemref();
    }

    const auto tensorType = value.getType();
    const bool isAlreadyABufferType = isBufferType(tensorType);
    if (isAlreadyABufferType) {
        return value;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(value);

    auto bufferType = vpux::getBufferType(value);
    auto origType = mlir::cast<vpux::NDTypeInterface>(value.getType());
    VPUX_THROW_WHEN(origType.hasRank() && origType.getRank() != bufferType.getRank(),
                    "Incompatible ranks: original rank {0}, buffer rank {1}", origType.getRank(), bufferType.getRank());

    // E#109609: replace with getResult()/getMemref() once we can convert
    //           VPUIP::{Distributed, Sparse}BufferType to mlir::BaseMemRefType
    return rewriter.create<mlir::bufferization::ToMemrefOp>(value.getLoc(), bufferType, value)->getResult(0);
}

//
// bufferizeOperands
//

SmallVector<mlir::Value> vpux::bufferizeOperands(mlir::RewriterBase& rewriter, mlir::OperandRange operands) {
    if (operands.size() == 0) {
        return {};
    }
    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(llvm::size(operands));
    for (const auto& operand : operands) {
        auto buffer = vpux::getBuffer(rewriter, operand);
        newOperands.push_back(buffer);
    }
    return newOperands;
}

//
// populateBufferizeMaterializationLegality
//

void vpux::populateBufferizeMaterializationLegality(mlir::ConversionTarget& target) {
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
}

//
// inferReturnTypes
//

void vpux::inferReturnTypes(mlir::Operation* op, InferShapedTypeMode mode) {
    auto iface = mlir::dyn_cast<mlir::InferTypeOpInterface>(op);
    VPUX_THROW_WHEN(iface == nullptr, "Operation '{0}' doesn't implement InferTypeOpInterface", op->getName());

    SmallVector<mlir::Type> newTypes;
    VPUX_THROW_WHEN(iface.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                           op->getPropertiesStorage(), op->getRegions(), newTypes)
                            .failed(),
                    "Failed to infer return types for operation '{0}'", op->getName());

    for (auto p : zip(op->getResults(), newTypes)) {
        auto val = std::get<0>(p);
        auto newType = mlir::dyn_cast<vpux::NDTypeInterface>(std::get<1>(p));
        VPUX_THROW_UNLESS(newType != nullptr, "newType has non vpux::NDTypeInterface type '{0}'", std::get<1>(p));

        if (!bitEnumContains(mode, InferShapedTypeMode::SHAPE)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeShape(oldType.getShape());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::ELEM_TYPE)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeElemType(oldType.getElementType());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::LAYOUT)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeDimsOrder(oldType.getDimsOrder());
            }
        }
        if (!bitEnumContains(mode, InferShapedTypeMode::MEM_SPACE)) {
            if (const auto oldType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
                newType = newType.changeMemSpace(oldType.getMemSpace());
            }
        }

        val.setType(newType);
    }
}
