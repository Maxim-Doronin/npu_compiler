//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/MLIRContext.h>

#include "vpux/utils/core/func_ref.hpp"

namespace vpux {

enum class LoopExecPolicy {
    Sequential,
    Parallel,
};

void loop_1d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, vpux::FuncRef<void(int64_t)> func);

void loop_2d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1,
             FuncRef<void(int64_t, int64_t)> func);

void loop_3d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1, int64_t dim2,
             FuncRef<void(int64_t, int64_t, int64_t)> func);

void loop_4d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
             FuncRef<void(int64_t, int64_t, int64_t, int64_t)> func);

/**
 * @brief Find the first index where the predicate is true. Each index between [0, size) will be passed
 * to the predicate function. The first one that is true will be returned.
 * @warning This should only be used in case the predicate is computationally expensive.
 * @param  ctx      The MLIRContext that contains the thread pool which will execute the predicate
 * @param  size     The total number of indices that will be tested with the predicate
 * @return  The first index in the range [0, size) such that the predicate is true, or failure if no such index exists.
 */
mlir::FailureOr<size_t> parallel_find_index(mlir::MLIRContext* ctx, size_t size, vpux::FuncRef<bool(size_t)> pred);

}  // namespace vpux
