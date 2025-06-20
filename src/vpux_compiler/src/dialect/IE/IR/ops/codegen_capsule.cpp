//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/asm.hpp"

using namespace vpux;

//
// CodeGenCapsuleOp
//

mlir::DenseMap<mlir::Value, mlir::Value> IE::CodeGenCapsuleOp::getYieldToResultMapping() {
    auto block = getBody();
    auto yieldOp = mlir::cast<IE::CGCYieldOp>(block->getTerminator());

    assert(yieldOp->getNumOperands() == getNumResults());
    auto yieldOperands = yieldOp->getOperands();
    auto opResults = getResults();

    mlir::DenseMap<mlir::Value, mlir::Value> map;
    for (auto [yieldOperand, opResult] : llvm::zip(yieldOperands, opResults)) {
        map.insert({yieldOperand, opResult});
    }

    return map;
}

mlir::DenseMap<mlir::Value, mlir::Value> IE::CodeGenCapsuleOp::getResultToYieldMapping() {
    auto block = getBody();
    auto yieldOp = mlir::cast<IE::CGCYieldOp>(block->getTerminator());

    assert(yieldOp->getNumOperands() == getNumResults());
    auto yieldOperands = yieldOp->getOperands();
    auto opResults = getResults();

    mlir::DenseMap<mlir::Value, mlir::Value> map;
    for (auto [opResult, yieldOperand] : llvm::zip(opResults, yieldOperands)) {
        map.insert({opResult, yieldOperand});
    }

    return map;
}

mlir::LogicalResult IE::CodeGenCapsuleOp::verify() {
    auto blockTerminator = getBody()->getTerminator();
    if (!mlir::isa<IE::CGCYieldOp>(blockTerminator)) {
        return mlir::failure();
    }
    auto yieldOperands = blockTerminator->getOperands();
    if (yieldOperands.getTypes() != getResultTypes()) {
        return mlir::failure();
    }
    return mlir::success();
}

void IE::CodeGenCapsuleOp::print(mlir::OpAsmPrinter& p) {
    p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{});

    auto block = getBody();
    unsigned argInd = 0;
    printGroupOfOperands(p, block, "inputs", getInputs(), argInd);
    p << ' ';

    p.printRegion(getRegion(), false);
    p.printOptionalArrowTypeList(getResultTypes());
}

mlir::ParseResult IE::CodeGenCapsuleOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    SmallVector<mlir::OpAsmParser::Argument> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    // Parse inputs
    int32_t inCount = 0;
    if (mlir::failed(parseGroupOfOperands(parser, result, blockArgs, blockTypes, "inputs", inCount))) {
        return mlir::failure();
    }

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs)) {
        return mlir::failure();
    }

    // Parse outputs
    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    return mlir::success();
}

//
// CGCYieldOp
//

void IE::CGCYieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, mlir::ValueRange());
}
