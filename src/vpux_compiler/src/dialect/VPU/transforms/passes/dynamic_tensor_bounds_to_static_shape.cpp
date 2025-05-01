//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_BOUNDEDTENSORSTODYNAMICDIMSMASK
#define GEN_PASS_DEF_BOUNDEDTENSORSTODYNAMICDIMSMASK
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class BoundedTensorsToDynamicDimsMask final :
        public VPU::impl::BoundedTensorsToDynamicDimsMaskBase<BoundedTensorsToDynamicDimsMask> {
public:
    explicit BoundedTensorsToDynamicDimsMask(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    Logger _log;
};

void BoundedTensorsToDynamicDimsMask::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();

    module.walk([&](VPU::BoundsRepresentationInterface op) {
        op.setBoundsRepresentation(VPU::BoundsRepresentation::DYNAMIC_DIMS_MASK);
    });

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([&](NDTypeInterface ndType) {
        auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(ndType);
        if (boundedType == nullptr) {
            return ndType;
        }

        const auto shape = ndType.getShape();
        const auto dimsMask = to_small_vector(shape | transformed([&](auto dim) -> int64_t {
                                                  return (dim == mlir::ShapedType::kDynamic) ? 1 : 0;
                                              }));

        const auto bounds = boundedType.getBounds().raw();
        auto typeComponents =
                vpux::TypeComponents().setShape(Shape(bounds)).setDynamicDimsMask(DynamicDimsMask(dimsMask));

        auto outType = ndType.changeTypeComponents(typeComponents);
        return outType;
    });

    const auto convert = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        auto newLocation = appendLoc(loc, "casted");

        auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(newLocation, type, inputs[0]);
        return castOp->getResult(0);
    };

    typeConverter.addSourceMaterialization(convert);
    typeConverter.addTargetMaterialization(convert);
    typeConverter.addArgumentMaterialization(convert);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    // Mark dialects used by ShaveCodeGen as legal.
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::math::MathDialect, mlir::linalg::LinalgDialect>();

    target.markUnknownOpDynamicallyLegal(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    if (mlir::failed(vpux::IE::runConvertOpTypes(module, typeConverter, target, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDynamicTensorBoundsToStaticShapePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createBoundedTensorsToDynamicDimsMaskPass(Logger log) {
    return std::make_unique<BoundedTensorsToDynamicDimsMask>(log);
}
