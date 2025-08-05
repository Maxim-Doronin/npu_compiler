//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace vpux;

namespace {
struct VPUInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

}  // namespace

//
// initialize
//

void vpux::VPU::VPUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops.cpp.inc>
            >();

    registerAttributes();
    registerTypes();

    addInterface<VPUInlinerInterface>();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/dialect.cpp.inc>
