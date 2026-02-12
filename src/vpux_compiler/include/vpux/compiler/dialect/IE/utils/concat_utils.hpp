//
// Copyright (C) 2024-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {
mlir::DenseSet<int64_t> getConcatAxes(IE::ConcatOp concatOp);
std::optional<std::pair<Dim, Shape>> inferOutputShapeAfterAffineReshapeBeforeConcat(mlir::Value curInput,
                                                                                    IE::ConcatOp concatOp,
                                                                                    IE::AffineReshapeOp reshapeOp);
mlir::ArrayAttr inferConcatOffsets(ArrayRef<ShapeRef> concatInShapes, const Dim concatDim, mlir::MLIRContext* ctx);

// TODO: E#159557 refactor initiative
mlir::Value createPaddingConstForConcat(ArrayRef<int64_t> constShape, mlir::Location loc,
                                        vpux::NDTypeInterface inputType, double padValue,
                                        mlir::PatternRewriter& rewriter);
const mlir::ArrayAttr inferOffsetsAttrWithAxis(IE::ConcatOp origOp, int64_t& axis);
std::optional<vpux::Dim> getConcatAxis(IE::ConcatOp concatOp);
mlir::FailureOr<SmallVector<Dim>> getConcatDimWithShape1(IE::ConcatOp concatOp, bool supportAdjacentDims);

template <class ConvolutionType>
mlir::FailureOr<mlir::Operation*> getConcatOpConsumer(mlir::Operation* op, bool requireAffineReshape,
                                                      bool requireConvolution) {
    if (op == nullptr || op->getUsers().empty()) {
        return mlir::failure();
    }

    mlir::Operation* concatOp = nullptr;

    for (auto user : op->getUsers()) {
        mlir::Operation* operation = user;

        if (requireAffineReshape) {
            if (!mlir::isa<IE::AffineReshapeOp>(operation) || operation->getUsers().empty() ||
                (!requireConvolution && !operation->hasOneUse())) {
                return mlir::failure();
            }
            operation = *(operation->getUsers().begin());
        }

        if (requireConvolution) {
            if (!mlir::isa<ConvolutionType>(operation) || operation->getUsers().empty() || !operation->hasOneUse()) {
                return mlir::failure();
            }

            auto convOp = mlir::dyn_cast<ConvolutionType>(operation);
            auto constFilter = mlir::dyn_cast<Const::DeclareOp>(convOp.getFilter().getDefiningOp());
            if (constFilter == nullptr) {
                return mlir::failure();
            }

            auto isConst = [](mlir::Value value) {
                return mlir::isa<Const::DeclareOp>(value.getDefiningOp());
            };

            if (!isConst(convOp.getFilter())) {
                return mlir::failure();
            }

            operation = *(operation->getUsers().begin());
        }

        if (!mlir::isa<IE::ConcatOp>(operation)) {
            return mlir::failure();
        }

        if (concatOp == nullptr) {
            concatOp = operation;
            continue;
        } else if (concatOp != operation) {
            return mlir::failure();
        }
    }

    return concatOp;
}

}  // namespace IE
}  // namespace vpux
