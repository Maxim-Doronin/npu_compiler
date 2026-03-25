//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace mlir {
class RewritePatternSet;
class TypeConverter;
}  // namespace mlir

namespace vpux::ShaveCodeGen {
void populateIEReduceToLinalgPatterns(mlir::RewritePatternSet& patternSet, mlir::TypeConverter& typeConverter);
void populateIEDataMovementToTensorPatterns(mlir::RewritePatternSet& patternSet, mlir::TypeConverter& typeConverter);
void populateIEShapeManipulationToTensorPatterns(mlir::RewritePatternSet& patternSet,
                                                 mlir::TypeConverter& typeConverter);
void populateIESoftmaxToLinalgPatterns(mlir::RewritePatternSet& patternSet, mlir::TypeConverter& typeConverter);
}  // namespace vpux::ShaveCodeGen
