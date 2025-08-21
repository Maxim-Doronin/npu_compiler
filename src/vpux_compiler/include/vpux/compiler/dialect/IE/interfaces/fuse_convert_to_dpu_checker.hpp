//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {

/*
   Class for fusion of Convert to DPU op checker
*/
class FuseConvertToDPUCheckerBase {
public:
    FuseConvertToDPUCheckerBase() = default;
    FuseConvertToDPUCheckerBase(const FuseConvertToDPUCheckerBase&) = default;
    FuseConvertToDPUCheckerBase(FuseConvertToDPUCheckerBase&&) = default;
    virtual ~FuseConvertToDPUCheckerBase() = default;

    FuseConvertToDPUCheckerBase& operator=(const FuseConvertToDPUCheckerBase&) = default;
    FuseConvertToDPUCheckerBase& operator=(FuseConvertToDPUCheckerBase&&) = default;

    virtual bool isFusionToParentDPUOpSupported(mlir::Operation* /*dpuOp*/, Logger /*log*/) const {
        return true;
    };
};

/*
   Find right class to verify whether fusion of Convert F16 -> F32 to parent DPU is feasible
*/
std::unique_ptr<FuseConvertToDPUCheckerBase> createFuseConvertToDPUChecker(vpux::config::ArchKind arch);

}  // namespace IE
}  // namespace vpux
