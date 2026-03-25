//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace IE {

void setupConvertPrecision(mlir::TypeConverter& typeConverter, FuncRef<mlir::Type(mlir::Type)> elemTypeConversionCb);

mlir::LogicalResult runConvertPrecision(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                        mlir::ConversionTarget& target, Logger& log);
mlir::LogicalResult runConvertOpTypes(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                      mlir::ConversionTarget& target, Logger& log);

}  // namespace IE
}  // namespace vpux
