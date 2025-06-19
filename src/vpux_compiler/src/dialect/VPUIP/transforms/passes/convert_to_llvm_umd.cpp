//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include "vpux/compiler/utils/analysis.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTTOLLVMUMDCALLS
#define GEN_PASS_DEF_CONVERTTOLLVMUMDCALLS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
#define GET_ARG_INDEX_CONTEXT(numArgs) ((numArgs)-7)
#define GET_ARG_INDEX_DEVICE(numArgs) ((numArgs)-6)
#define GET_ARG_INDEX_DDI_TABLE(numArgs) ((numArgs)-5)
#define GET_ARG_INDEX_COMMAND_LIST(numArgs) ((numArgs)-4)
#define GET_ARG_INDEX_COMMAND_QUEUE(numArgs) ((numArgs)-3)
#define GET_ARG_INDEX_COMMAND_FENCE(numArgs) ((numArgs)-2)
#define GET_ARG_INDEX_COMMAND_EVENT(numArgs) ((numArgs)-1)

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
            : mlir::ConvertOpToLLVMPattern<mlir::memref::AllocOp>(typeConverter) {
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
    auto moduleOp = vpux::getModuleOp(origOp);
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMPointerType::get(ctx));
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();
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
            : mlir::ConvertOpToLLVMPattern<mlir::memref::CopyOp>(typeConverter) {
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
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();
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
    AsyncOpRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<AsyncOp>(ctx), _log(std::move(log)) {
        this->setDebugName("AsyncOpRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const final;
    Logger _log;
};

template <typename AsyncOp>
mlir::LogicalResult AsyncOpRewriter<AsyncOp>::matchAndRewrite(AsyncOp origOp, mlir::PatternRewriter& rewriter) const {
    auto submitCommandList = [&](mlir::Operation* origOp) {
        mlir::MLIRContext* ctx = rewriter.getContext();
        auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();
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
    auto funcOp = origOp->getParentOfType<mlir::func::FuncOp>();
    auto numArgs = funcOp.getNumArguments();
    auto cmdlist = funcOp.getArgument(GET_ARG_INDEX_COMMAND_LIST(numArgs));

    mlir::MLIRContext* ctx = rewriter.getContext();
    auto returnType = mlir::Type(mlir::LLVM::LLVMVoidType::get(ctx));
    createLLVMFuncCallOp(rewriter, getModuleOp(origOp), "npu_level_zero_reset_commandlist", {cmdlist}, returnType);
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// ConvertToLLVMUMDCallsPass
//

class ConvertToLLVMUMDCallsPass final : public VPUIP::impl::ConvertToLLVMUMDCallsBase<ConvertToLLVMUMDCallsPass> {
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

    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();

    patterns.add<LvlZeroMemoryCopyLowering>(typeConverter);
    patterns.add<LvlZeroAllocLowering>(typeConverter);
    patterns.add<AsyncOpRewriter<mlir::async::AddToGroupOp>>(ctx, _log);
    patterns.add<AsyncOpRewriter<mlir::async::CreateGroupOp>>(ctx, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitAllOp>>(ctx, _log);
    patterns.add<AsyncOpRewriter<mlir::async::AwaitOp>>(ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToLLVMUMDCallsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertToLLVMUMDCallsPass(Logger log) {
    return std::make_unique<ConvertToLLVMUMDCallsPass>(log);
}
