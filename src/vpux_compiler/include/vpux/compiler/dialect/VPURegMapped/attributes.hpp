//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

mlir::Attribute getVPURegMapped_RegisterFieldAttr(::mlir::MLIRContext* context, vpux::VPURegMapped::RegFieldType value);
mlir::ArrayAttr getVPURegMapped_RegisterFieldArrayAttr(mlir::OpBuilder builder,
                                                       mlir::ArrayRef<vpux::VPURegMapped::RegFieldType> values);

mlir::Attribute getVPURegMapped_RegisterAttr(::mlir::MLIRContext* context, vpux::VPURegMapped::RegisterType value);
mlir::ArrayAttr getVPURegMapped_RegisterArrayAttr(mlir::OpBuilder builder,
                                                  mlir::ArrayRef<vpux::VPURegMapped::RegisterType> values);

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/attributes.hpp.inc>
#undef GET_ATTRDEF_CLASSES
