//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace vpux {
#define GEN_PASS_DECL_CONVERTHOSTCODETOBYTECODE
#define GEN_PASS_DEF_CONVERTHOSTCODETOBYTECODE
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

class ArithConstantRewriter final : public mlir::OpConversionPattern<mlir::arith::ConstantOp> {
public:
    ArithConstantRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::ConstantOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::ConstantOp origOp, OpAdaptor /*adaptor*/,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        int64_t value = 0;
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(origOp.getValueAttr())) {
            value = intAttr.getInt();
        } else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(origOp.getValueAttr())) {
            value = llvm::bit_cast<int64_t>(floatAttr.getValueAsDouble());
        } else {
            _log.error("Unsupported constant attribute type: {0}", origOp.getValueAttr().getType());
            return mlir::failure();
        }

        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::SetImmOp>(origOp.getLoc(), dstRegOp.getResult(), rewriter.getI64IntegerAttr(value));
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithAddIRewriter final : public mlir::OpConversionPattern<mlir::arith::AddIOp> {
public:
    ArithAddIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::AddIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::AddIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::AddI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMulIRewriter final : public mlir::OpConversionPattern<mlir::arith::MulIOp> {
public:
    ArithMulIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MulIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MulIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MulI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMinSIRewriter final : public mlir::OpConversionPattern<mlir::arith::MinSIOp> {
public:
    ArithMinSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MinSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MinSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MinI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ArithMaxSIRewriter final : public mlir::OpConversionPattern<mlir::arith::MaxSIOp> {
public:
    ArithMaxSIRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::arith::MaxSIOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::arith::MaxSIOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto dstRegOp = rewriter.create<bytecode::VirtualGeneralRegisterOp>(origOp.getLoc());
        rewriter.create<bytecode::MaxI64Op>(origOp.getLoc(), dstRegOp.getResult(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(origOp, dstRegOp.getResult());
        return mlir::success();
    }

private:
    Logger _log;
};

class ReturnRewriter final : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
public:
    ReturnRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::func::ReturnOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp origOp, OpAdaptor /*adaptor*/,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        rewriter.replaceOpWithNewOp<bytecode::RetOp>(origOp);
        return mlir::success();
    }

private:
    Logger _log;
};

class AssertRewriter final : public mlir::OpConversionPattern<mlir::cf::AssertOp> {
public:
    AssertRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpConversionPattern<mlir::cf::AssertOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::cf::AssertOp origOp, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        rewriter.replaceOpWithNewOp<bytecode::ExtAssertOp>(origOp, adaptor.getArg(), adaptor.getMsg());
        return mlir::success();
    }

private:
    Logger _log;
};

}  // namespace

namespace vpux {

class ConvertHostcodeToBytecodePass final : public impl::ConvertHostcodeToBytecodeBase<ConvertHostcodeToBytecodePass> {
public:
    explicit ConvertHostcodeToBytecodePass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        auto moduleOp = getOperation();

        const auto hostCompileFunctions = [&]() {
            SmallVector<mlir::func::FuncOp> hostCompileFunctions;
            for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
                if (config::isPureHostCompileFunc(funcOp)) {
                    hostCompileFunctions.push_back(funcOp);
                }
            }
            return hostCompileFunctions;
        }();
        if (hostCompileFunctions.empty()) {
            _log.debug("No host compile functions found.");
            return;
        }

        auto funcSection = prepareFuncSection(moduleOp);
        for (auto funcOp : hostCompileFunctions) {
            if (mlir::failed(convertFuncToBytecode(funcOp, funcSection))) {
                _log.error("Failed to convert function {0} to bytecode", funcOp.getName());
                signalPassFailure();
                return;
            }
        }
    }

    // Introduce an empty function sections into the module operation
    bytecode::FuncSectionOp prepareFuncSection(mlir::ModuleOp moduleOp) {
        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToEnd(moduleOp.getBody());
        const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), bytecode::FUNCTION_SECTION_NAME));
        auto funcSection = bytecode::FuncSectionOp::create(builder, loc, bytecode::FUNCTION_SECTION_NAME);
        funcSection.getContent().emplaceBlock();
        return funcSection;
    }

    // Convert a function to bytecode operations and store the new bytecode function in the provided function section
    // The original function is erased after conversion
    mlir::LogicalResult convertFuncToBytecode(mlir::func::FuncOp funcOp, bytecode::FuncSectionOp funcSection) {
        mlir::TypeConverter typeConverter;
        typeConverter.addConversion([](mlir::IntegerType type) -> mlir::Type {
            return bytecode::RegisterType::get(type.getContext());
        });
        const auto materialize = [&](mlir::OpBuilder& builder, mlir::Type resultType, mlir::ValueRange inputs,
                                     mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1) {
                return mlir::Value();
            }
            if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(inputs.front())) {
                return builder.create<bytecode::VirtualParameterRegisterOp>(loc, blockArg.getArgNumber());
            }
            return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
        };
        typeConverter.addTargetMaterialization(materialize);
        typeConverter.addSourceMaterialization(materialize);

        auto ctx = &getContext();
        mlir::ConversionTarget target(*ctx);
        target.addIllegalDialect<mlir::arith::ArithDialect>();
        target.addIllegalDialect<mlir::cf::ControlFlowDialect>();
        target.addIllegalOp<mlir::func::ReturnOp>();
        target.addLegalDialect<bytecode::BytecodeDialect>();

        mlir::RewritePatternSet patterns(ctx);
        patterns.add<ArithConstantRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithAddIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMulIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMinSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ArithMaxSIRewriter>(typeConverter, ctx, _log);
        patterns.add<ReturnRewriter>(typeConverter, ctx, _log);
        patterns.add<AssertRewriter>(typeConverter, ctx, _log);

        if (mlir::failed(mlir::applyPartialConversion(funcOp, target, std::move(patterns)))) {
            return errorAt(funcOp, "Failed to apply conversion patterns");
        }

        mlir::OpBuilder builder(&getContext());
        builder.setInsertionPointToEnd(&funcSection.getContent().getBlocks().front());
        auto bytecodeFuncOp =
                bytecode::ExtFuncOp::create(builder, funcOp->getLoc(), funcOp.getName(), funcOp.getFunctionType());
        // Take the body of the original function (which now contains bytecode operations) and move it to the new
        // bytecode function, then erase the block arguments, as these are now represented by virtual parameter register
        // operations
        bytecodeFuncOp.getBody().takeBody(funcOp.getBody());
        for (auto arg : bytecodeFuncOp.getBody().getArguments() | reversed) {
            bytecodeFuncOp.getBody().eraseArgument(arg.getArgNumber());
        }
        funcOp->erase();

        return mlir::success();
    }
};

}  // namespace vpux

std::unique_ptr<mlir::Pass> vpux::bytecode::createConvertHostcodeToBytecodePass(const Logger& log) {
    return std::make_unique<ConvertHostcodeToBytecodePass>(log);
}
