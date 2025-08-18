//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

namespace vpux::HostExec {
#define GEN_PASS_DECL_CONVERTTOLLVMUMDCALLS
#define GEN_PASS_DEF_CONVERTTOLLVMUMDCALLS
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {
void addFuncParamsForUmdFuncCall(mlir::func::FuncOp& funcOp) {
    // Update the function's return type to NoneType
    auto funcType = funcOp.getFunctionType();
    auto ctx = funcOp.getContext();

    SmallVector<mlir::Type, 8> newInputTypes;
    for (auto input : funcType.getInputs()) {
        newInputTypes.push_back(input);
    }

    // add handle to command list
    mlir::Type contextHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type deviceHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type ddiTableHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type commandListHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type commandQueueHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type fenceHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type eventHandlePtrType = mlir::LLVM::LLVMPointerType::get(ctx);

    newInputTypes.push_back(contextHandlePtrType);
    newInputTypes.push_back(deviceHandlePtrType);
    newInputTypes.push_back(ddiTableHandlePtrType);
    newInputTypes.push_back(commandListHandlePtrType);
    newInputTypes.push_back(commandQueueHandlePtrType);
    newInputTypes.push_back(fenceHandlePtrType);
    newInputTypes.push_back(eventHandlePtrType);

    for (auto& block : funcOp.getBody()) {
        // Find the current terminator
        if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator())) {
            // Replace the return operation with a new one that has no arguments
            mlir::OpBuilder builder(returnOp);
            builder.create<mlir::func::ReturnOp>(returnOp.getLoc());
            returnOp.erase();
        }
    }

    auto newFuncType =
            mlir::FunctionType::get(funcOp.getContext(), newInputTypes, mlir::TypeRange{});  // No return types
    funcOp.setType(newFuncType);
    auto& entryBlock = funcOp.getBody().front();
    entryBlock.addArgument(contextHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(deviceHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(ddiTableHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(commandListHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(commandQueueHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(fenceHandlePtrType, funcOp->getLoc());
    entryBlock.addArgument(eventHandlePtrType, funcOp->getLoc());
}
mlir::LogicalResult areAllLLVMTypes(mlir::Operation* op, mlir::ValueRange operands,
                                    mlir::ConversionPatternRewriter& rewriter) {
    if (!llvm::all_of(operands, [](mlir::Value value) {
            return mlir::LLVM::isCompatibleType(value.getType());
        })) {
        return rewriter.notifyMatchFailure(op, "Cannot convert if operands aren't of LLVM type.");
    }
    return mlir::success();
}

mlir::LLVM::CallOp createLLVMFuncCallOp(mlir::OpBuilder& builder, mlir::ModuleOp module, StringRef name,
                                        ArrayRef<mlir::Value> args, mlir::Type returnType) {
    SmallVector<mlir::Type> argTypes;
    argTypes.reserve(args.size());
    for (auto arg : args) {
        argTypes.push_back(arg.getType());
    }
    auto funcType = mlir::LLVM::LLVMFunctionType::get(returnType, argTypes);
    auto funcOp = [&] {
        if (auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
            return function;
        }
        return mlir::OpBuilder::atBlockBegin(module.getBody())
                .create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, funcType);
    }();

    return builder.create<mlir::LLVM::CallOp>(builder.getUnknownLoc(), funcOp, args);
}

class LvlZeroAllocLowering final : public mlir::ConvertOpToLLVMPattern<mlir::memref::AllocOp> {
public:
    LvlZeroAllocLowering(const mlir::LLVMTypeConverter& typeConverter)
            : mlir::ConvertOpToLLVMPattern<mlir::memref::AllocOp>(typeConverter, vpux::benefitHigh) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult LvlZeroAllocLowering::matchAndRewrite(mlir::memref::AllocOp origOp, OpAdaptor adaptor,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MemRefType memrefType = origOp.getType();
    if (mlir::failed(areAllLLVMTypes(origOp, adaptor.getOperands(), rewriter)) ||
        !isConvertibleAndHasIdentityMaps(memrefType)) {
        return mlir::failure();
    }
    // Get shape of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.
    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;

    auto loc = origOp.getLoc();
    getMemRefDescriptorSizes(loc, memrefType, adaptor.getDynamicSizes(), rewriter, shape, strides, sizeBytes);
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMPointerType::get(ctx));
    mlir::func::FuncOp funcOp;
    auto moduleOp = vpux::getModuleOp(origOp);
    vpux::net::NetworkInfoOp netInfo;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    auto numArgs = funcOp.getNumArguments();
    auto context = funcOp.getArgument(GET_ARG_INDEX_CONTEXT(numArgs));
    auto allocatedPtr =
            createLLVMFuncCallOp(rewriter, moduleOp, "npu_level_zero_alloc", {sizeBytes, context}, returnType)
                    .getResult();
    // No alignment.
    mlir::Value alignedPtr = allocatedPtr;
    auto memrefDescriptor =
            this->createMemRefDescriptor(loc, memrefType, allocatedPtr, alignedPtr, shape, strides, rewriter);

    rewriter.replaceOp(origOp, {memrefDescriptor});

    return mlir::success();
}

class LvlZeroMemoryCopyLowering final : public mlir::ConvertOpToLLVMPattern<mlir::memref::CopyOp> {
public:
    LvlZeroMemoryCopyLowering(const mlir::LLVMTypeConverter& typeConverter)
            : mlir::ConvertOpToLLVMPattern<mlir::memref::CopyOp>(typeConverter, vpux::benefitHigh) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::memref::CopyOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult LvlZeroMemoryCopyLowering ::matchAndRewrite(mlir::memref::CopyOp origOp, OpAdaptor adaptor,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    auto loc = origOp.getLoc();
    auto srcType = mlir::cast<mlir::BaseMemRefType>(origOp.getSource().getType());
    auto targetType = mlir::cast<mlir::BaseMemRefType>(origOp.getTarget().getType());

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](mlir::Value ranked, mlir::MemRefType type) {
        auto rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, getIndexType(), type.getRank());
        auto* typeConverter = getTypeConverter();
        auto ptr = typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);

        auto unrankedType = mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
        return mlir::UnrankedMemRefDescriptor::pack(rewriter, loc, *typeConverter, unrankedType, {rank, ptr});
    };

    // Save stack position before promoting descriptors
    auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(loc, getVoidPtrType());

    auto srcMemRefType = mlir::dyn_cast<mlir::MemRefType>(srcType);
    mlir::Value unrankedSource = srcMemRefType ? makeUnranked(adaptor.getSource(), srcMemRefType) : adaptor.getSource();
    auto targetMemRefType = mlir::dyn_cast<mlir::MemRefType>(targetType);
    mlir::Value unrankedTarget =
            targetMemRefType ? makeUnranked(adaptor.getTarget(), targetMemRefType) : adaptor.getTarget();

    // Now promote the unranked descriptors to the stack.
    auto one = rewriter.create<mlir::LLVM::ConstantOp>(loc, getIndexType(), rewriter.getIndexAttr(1));
    auto promote = [&](mlir::Value desc) {
        auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
        auto allocated = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, desc.getType(), one);
        rewriter.create<mlir::LLVM::StoreOp>(loc, desc, allocated);
        return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);
    auto elemSize = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    auto module = vpux::getModuleOp(origOp);
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
    mlir::func::FuncOp funcOp;
    auto moduleOp = vpux::getModuleOp(origOp);
    vpux::net::NetworkInfoOp netInfo;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    auto numArgs = funcOp.getNumArguments();
    auto cmdlist = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));

    createLLVMFuncCallOp(rewriter, module, "npu_level_zero_append_memory_copy",
                         {sourcePtr, targetPtr, elemSize, cmdlist}, returnType);
    // Restore stack used for descriptors
    rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSaveOp);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

template <typename AsyncOp>
class AsyncOpRewriter final : public mlir::OpRewritePattern<AsyncOp> {
public:
    AsyncOpRewriter(mlir::MLIRContext* ctx, const mlir::LLVMTypeConverter& typeConverter, mlir::PatternBenefit benefit,
                    Logger log)
            : mlir::OpRewritePattern<AsyncOp>(ctx, benefit), _typeConverter(typeConverter), _log(std::move(log)) {
        this->setDebugName("AsyncOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const final;
    const mlir::LLVMTypeConverter& _typeConverter;
    Logger _log;
};

template <typename AsyncOp>
mlir::LogicalResult AsyncOpRewriter<AsyncOp>::matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const {
    auto submitCommandList = [&](mlir::Operation* origOp) {
        mlir::MLIRContext* ctx = rewriter.getContext();
        mlir::func::FuncOp funcOp;
        auto moduleOp = vpux::getModuleOp(origOp);
        vpux::net::NetworkInfoOp netInfo;
        vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
        auto numArgs = funcOp.getNumArguments();
        auto cmdlist = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));
        auto cmdQueue = funcOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));
        auto fence = funcOp.getArgument(GET_ARG_INDEX_COMMAND_FENCE(numArgs));
        auto event = funcOp.getArgument(GET_ARG_INDEX_COMMAND_EVENT(numArgs));
        auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
        createLLVMFuncCallOp(rewriter, getModuleOp(origOp), "npu_level_zero_submit_commandlist",
                             {cmdlist, cmdQueue, fence, event}, returnType);
        rewriter.eraseOp(origOp);
        return mlir::success();
    };

    if (mlir::isa<mlir::async::AwaitOp>(origOp)) {
        if (origOp->template getParentOfType<mlir::scf::ForOp>()) {
            rewriter.eraseOp(origOp);
            return mlir::success();
        }

        mlir::async::AwaitOp awaitOp = mlir::cast<mlir::async::AwaitOp>(*origOp);
        auto users = awaitOp->getUsers();
        if (users.empty()) {
            return submitCommandList(origOp);
        }

        // Async.AwaitOp
        // Replace operand of uses with operand of async.awaitop
        // as AwaitOp will be removed
        mlir::Value awaitOpOperand = awaitOp.getOperand();
        auto op = awaitOpOperand.getDefiningOp();
        if (auto executeOp = mlir::cast<mlir::async::ExecuteOp>(op)) {
            const auto results = executeOp.getResults();
            int index = -1;
            for (size_t i = 0; i < results.size(); ++i) {
                if (awaitOp.getOperand() == results[i]) {
                    if (i == 0) {
                        _log.error("Invalid index {0} as the first result is token", i);
                        return mlir::failure();
                    }

                    // decrease index as the first result of ExecuteOp is token
                    index = static_cast<int>(i) - 1;
                    break;
                }
            }

            if (index == -1) {
                _log.error("Invalid async.AwaitOp");
                return mlir::failure();
            }

            auto moduleOp = vpux::getModuleOp(executeOp);
            for (auto& op : executeOp.getBody()->getOperations()) {
                if (auto callOp = mlir::dyn_cast<Core::NestedCallOp>(op)) {
                    auto callee = callOp->getAttrOfType<mlir::SymbolRefAttr>("callee");
                    auto root = callee.getRootReference();
                    auto fnModule = moduleOp.lookupSymbol<HostExec::BinaryOp>(root);
                    if (fnModule == nullptr) {
                        _log.error("Could not find binary op for subgraph: {0}", root.str());
                        return mlir::failure();
                    }
                    auto funcOp = fnModule.lookupSymbol<mlir::func::FuncOp>(callee.getLeafReference().str());
                    if (funcOp == nullptr) {
                        _log.error("Could not find function declaration: {0}", callee.getLeafReference().str());
                        return mlir::failure();
                    }

                    auto resultCount = funcOp.getNumResults();
                    auto inputCount = funcOp.getNumArguments();
                    auto operand = *(callOp.getOperands().begin() + (inputCount - resultCount) +
                                     static_cast<unsigned int>(index));
                    for (auto u : users) {
                        if (auto viewOp = mlir::dyn_cast<mlir::memref::SubViewOp>(u)) {
                            viewOp.setOperand(0, operand);
                        } else if (auto copyOp = mlir::dyn_cast<mlir::memref::CopyOp>(u)) {
                            copyOp.setOperand(0, operand);
                        } else if (mlir::isa<mlir::UnrealizedConversionCastOp, Core::NestedCallOp,
                                             mlir::async::YieldOp>(u)) {
                            continue;
                        } else {
                            _log.error("Not supported user type: {0}", u->getName().getStringRef().str());
                            return mlir::failure();
                        }
                    }
                }
            }
        }

        mlir::Operation* nextOp = origOp->getNextNode();

        // If there are multiple AwaitOp for one operation with multiple outputs
        // there will be multiple AwaitOp for synchronization.
        // The last AwaitOp will be replaced with submitCommandList.
        if (mlir::isa<mlir::async::AwaitOp>(nextOp)) {
            rewriter.eraseOp(origOp);
            return mlir::success();
        }
        return submitCommandList(origOp);
    } else if (mlir::isa<mlir::async::AwaitAllOp>(origOp)) {
        return submitCommandList(origOp);
    } else {
        _log.warning("Unsupported async operation: {0}", origOp->getName());
    }
    return mlir::failure();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::AddToGroupOp>::matchAndRewrite(mlir::async::AddToGroupOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    rewriter.eraseOp(origOp);
    return mlir::success();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::CreateGroupOp>::matchAndRewrite(
        mlir::async::CreateGroupOp origOp, mlir::PatternRewriter& rewriter) const {
    mlir::func::FuncOp funcOp;
    auto moduleOp = vpux::getModuleOp(origOp);
    vpux::net::NetworkInfoOp netInfo;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    auto numArgs = funcOp.getNumArguments();
    auto cmdlist = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));

    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
    createLLVMFuncCallOp(rewriter, getModuleOp(origOp), "npu_level_zero_reset_commandlist", {cmdlist}, returnType);
    rewriter.eraseOp(origOp);
    return mlir::success();
}

template <>
mlir::LogicalResult AsyncOpRewriter<mlir::async::ExecuteOp>::matchAndRewrite(mlir::async::ExecuteOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto loc = origOp.getLoc();
    mlir::func::FuncOp funcOp;
    auto moduleOp = vpux::getModuleOp(origOp);
    vpux::net::NetworkInfoOp netInfo;
    vpux::net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);

    auto numArgs = funcOp.getNumArguments();
    auto umdContext = funcOp.getArgument(GET_ARG_INDEX_CONTEXT(numArgs));
    auto device = funcOp.getArgument(GET_ARG_INDEX_DEVICE(numArgs));
    auto ddiTable = funcOp.getArgument(GET_ARG_INDEX_DDI_TABLE(numArgs));
    auto cmdList = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));
    auto cmdQueue = funcOp.getArgument(GET_ARG_INDEX_COMMAND_QUEUE(numArgs));
    // needs to calculate the size of the kernel function after serialization of the core.NestedModule

    auto voidPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    std::map<std::string, std::pair<HostExec::BinaryOp, mlir::Value>> kernels;
    for (auto& op : origOp.getBody()->getOperations()) {
        if (auto callOp = mlir::dyn_cast<Core::NestedCallOp>(op)) {
            mlir::SmallVector<mlir::Value, 1> kernelInputs, kernelOutputs;
            auto callee = callOp->getAttrOfType<mlir::SymbolRefAttr>("callee");
            auto root = callee.getRootReference();
            auto calleeNameAttr = callee.getLeafReference();
            auto kernelBinary = moduleOp.lookupSymbol<HostExec::BinaryOp>(root);
            auto rootStr = root.str();
            if (!kernelBinary) {
                _log.error("BinaryOp not found for {0}", root.str());
                return mlir::failure();
            }
            auto binaryDataOp =
                    kernelBinary.lookupSymbol<HostExec::BinaryDataOp>("serialized_" + callee.getLeafReference().str());
            if (!binaryDataOp) {
                _log.error("BinaryDataOp not found for {0}", callee.getLeafReference().str());
                return mlir::failure();
            }

            auto object = binaryDataOp.getObject();
            if (!object) {
                _log.error("Object not found in BinaryDataOp for {0}",
                           callOp.getOperation()->getName().getStringRef().str());
                return mlir::failure();
            }

            llvm::StringRef rawBytes = object.getObject().getValue();
            size_t dataSize = rawBytes.size();
            auto kernelSize = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), dataSize);

            auto resultCount = callOp.getNumResults();
            auto inputCount = callOp.getNumOperands() - resultCount;

            mlir::Value kernelGlobal;
            auto iter = kernels.find(calleeNameAttr.str());
            if (iter != kernels.end()) {
                kernelGlobal = iter->second.second;
            } else {
                auto name = callee.getLeafReference().getValue();
                auto nameAttr = mlir::StringAttr::get(origOp.getContext(), std::string(name) + "_kernel");
                kernelGlobal = mlir::LLVM::createGlobalString(loc, rewriter, nameAttr.getValue(), object.getObject(),
                                                              mlir::LLVM::Linkage::Internal);
                kernels[calleeNameAttr.str()] = std::make_pair(kernelBinary, kernelGlobal);
            }

            kernelInputs.insert(kernelInputs.begin(), callOp.getArgOperands().begin(),
                                callOp.getArgOperands().begin() + inputCount);
            kernelOutputs.insert(kernelOutputs.begin(), callOp.getArgOperands().begin() + inputCount,
                                 callOp.getArgOperands().end());

            auto numInputs =
                    rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), kernelInputs.size());
            auto numOutputs =
                    rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), kernelOutputs.size());

            auto inputs = rewriter.create<mlir::LLVM::AllocaOp>(loc, voidPtrTy, voidPtrTy, numInputs);
            auto outputs = rewriter.create<mlir::LLVM::AllocaOp>(loc, voidPtrTy, voidPtrTy, numOutputs);

            // Store each output pointer as void* in the input array
            for (size_t i = 0; i < kernelInputs.size(); ++i) {
                auto idx = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), i);
                auto gep = rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, voidPtrTy, inputs, mlir::ValueRange{idx});
                auto llvmInput = kernelInputs[i];
                if (!mlir::LLVM::isCompatibleType(llvmInput.getType())) {
                    if (auto converted = _typeConverter.materializeTargetConversion(
                                rewriter, loc, _typeConverter.convertType(llvmInput.getType()),
                                mlir::ValueRange{llvmInput})) {
                        llvmInput = converted;
                    } else {
                        _log.error("Could not convert input type: {0}", llvmInput.getType());
                        return mlir::failure();
                    }
                }

                auto desc = mlir::MemRefDescriptorView(mlir::ValueRange{llvmInput});
                rewriter.create<mlir::LLVM::StoreOp>(loc, desc.allocatedPtr(), gep);
            }

            // Store each output pointer as void* in the output array
            for (size_t i = 0; i < kernelOutputs.size(); ++i) {
                auto idx = rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), i);
                auto gep =
                        rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, voidPtrTy, outputs, mlir::ValueRange{idx});
                auto llvmOutput = kernelOutputs[i];
                if (!mlir::LLVM::isCompatibleType(llvmOutput.getType())) {
                    if (auto converted = _typeConverter.materializeTargetConversion(
                                rewriter, loc, _typeConverter.convertType(llvmOutput.getType()),
                                mlir::ValueRange{llvmOutput})) {
                        llvmOutput = converted;
                    } else {
                        _log.error("Could not convert output type: {0}", llvmOutput.getType());
                        return mlir::failure();
                    }
                }
                auto desc = mlir::MemRefDescriptorView(mlir::ValueRange{llvmOutput});
                rewriter.create<mlir::LLVM::StoreOp>(loc, desc.allocatedPtr(), gep);
            }

            auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
            createLLVMFuncCallOp(rewriter, getModuleOp(origOp), "npu_level_zero_execute_graph",
                                 {inputs, numInputs, outputs, numOutputs, kernelGlobal, kernelSize, umdContext, device,
                                  ddiTable, cmdList, cmdQueue},
                                 returnType);
        }
    }

    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// ConvertToLLVMUMDCallsPass
//

class ConvertToLLVMUMDCallsPass final : public HostExec::impl::ConvertToLLVMUMDCallsBase<ConvertToLLVMUMDCallsPass> {
public:
    explicit ConvertToLLVMUMDCallsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertToLLVMUMDCallsPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = &getContext();

    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);
    mlir::LowerToLLVMOptions options(ctx);
    mlir::LLVMTypeConverter typeConverter(ctx, options);

    for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
        if (funcOp.getName() == "main") {
            addFuncParamsForUmdFuncCall(funcOp);
        }
    }

    mlir::populateConversionTargetFromOperation(module, target, typeConverter, patterns);
    target.addIllegalOp<mlir::memref::AllocOp>();
    target.addIllegalOp<mlir::memref::CopyOp>();
    target.addIllegalOp<mlir::async::CreateGroupOp>();
    target.addIllegalOp<mlir::async::AddToGroupOp>();
    target.addIllegalOp<mlir::async::AwaitAllOp>();
    target.addIllegalOp<mlir::async::AwaitOp>();
    target.addIllegalOp<mlir::async::ExecuteOp>();
    target.addLegalOp<vpux::HostExec::BinaryOp>();
    target.addLegalOp<vpux::HostExec::BinaryDataOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    patterns.add<LvlZeroMemoryCopyLowering>(typeConverter);
    patterns.add<LvlZeroAllocLowering>(typeConverter);
    patterns.add<AsyncOpRewriter<mlir::async::AddToGroupOp>>(ctx, typeConverter, vpux::benefitHigh, _log);
    patterns.add<AsyncOpRewriter<mlir::async::CreateGroupOp>>(ctx, typeConverter, vpux::benefitHigh, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitAllOp>>(ctx, typeConverter, vpux::benefitHigh, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitOp>>(ctx, typeConverter, vpux::benefitHigh, _log);
    // Note: ExecuteOp is a special case, a few conditions apply which is why it is the last pattern,
    // 1 npu_level_zero_execute_graph that all inputs and outputs are converted to LLVM types.
    // 2.It will have successor and predecessor dependencies on the other async and memref operations therefore those
    // operations should be removed or converted before this pattern is applied.

    patterns.add<AsyncOpRewriter<mlir::async::ExecuteOp>>(ctx, typeConverter, vpux::benefitLow, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }

    // Remove all BinaryOps as global variables for the ops were defined
    auto binaryOps = to_small_vector(module.getOps<HostExec::BinaryOp>());
    for (auto binaryOp : binaryOps) {
        binaryOp.getOperation()->erase();
    }
}

}  // namespace

//
// createConvertToLLVMUMDCallsPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createConvertToLLVMUMDCallsPass(Logger log) {
    return std::make_unique<ConvertToLLVMUMDCallsPass>(log);
}
