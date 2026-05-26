//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>
#include <mlir/IR/BuiltinTypes.h>

#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace {
mlir::Type localGetInt8Type(mlir::MLIRContext* ctx) {
    return vpux::getInt8Type(ctx);
}
mlir::Type localGetFp32Type(mlir::MLIRContext* ctx) {
    return mlir::Float32Type::get(ctx);
}
}  // namespace

template <typename ResultType>
static void BM_ContentGetValues(benchmark::State& state, vpux::FuncRef<mlir::Type(mlir::MLIRContext*)> getInputType) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const auto inElemType = getInputType(&ctx);
    const auto elemCount = state.range(0);
    const auto inTensorType = mlir::RankedTensorType::get(mlir::ArrayRef(elemCount), inElemType);

    auto content = vpux::Const::Content::allocTempBuffer(inTensorType, inElemType, false);
    std::vector<ResultType> output(inTensorType.getNumElements());

    for (auto _ : state) {
        auto values = content.template getValues<ResultType>();
        for (size_t i = 0; i < values.size(); ++i) {
            output[i] = values[i];
        }
        benchmark::ClobberMemory();
    }
}

template <typename ResultType>
static void BM_ContentRead(benchmark::State& state, vpux::FuncRef<mlir::Type(mlir::MLIRContext*)> getInputType) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const auto inElemType = getInputType(&ctx);
    const auto elemCount = state.range(0);
    const auto inTensorType = mlir::RankedTensorType::get(mlir::ArrayRef(elemCount), inElemType);

    auto content = vpux::Const::Content::allocTempBuffer(inTensorType, inElemType, false);
    std::vector<ResultType> output(inTensorType.getNumElements());

    for (auto _ : state) {
        content.read([&](auto rawValues) {
            for (size_t i = 0; i < rawValues.size(); ++i) {
                // Note: pessimize by having checked_cast, not static_cast
                output[i] = vpux::checked_cast<ResultType>(rawValues[i]);
            }
        });
        benchmark::ClobberMemory();
    }
}

template <typename ResultType>
static void BM_ContentGetTempBuf(benchmark::State& state, vpux::FuncRef<mlir::Type(mlir::MLIRContext*)> getInputType) {
    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::Const::ConstDialect>();

    const auto inElemType = getInputType(&ctx);
    const auto elemCount = state.range(0);
    const auto inTensorType = mlir::RankedTensorType::get(mlir::ArrayRef(elemCount), inElemType);

    auto content = vpux::Const::Content::allocTempBuffer(inTensorType, inElemType, false);
    std::vector<ResultType> output(inTensorType.getNumElements());

    for (auto _ : state) {
        auto rawValues = content.template getTempBuf<ResultType>();
        for (size_t i = 0; i < rawValues.size(); ++i) {
            // Note: pessimize by having checked_cast, not static_cast
            output[i] = vpux::checked_cast<ResultType>(rawValues[i]);
        }
        benchmark::ClobberMemory();
    }
}

BENCHMARK_CAPTURE(BM_ContentGetValues<float>, si8_to_fp32, &localGetInt8Type)->Range(8, 8 << 10);
BENCHMARK_CAPTURE(BM_ContentRead<float>, si8_to_fp32, &localGetInt8Type)->Range(8, 8 << 10);

BENCHMARK_CAPTURE(BM_ContentGetValues<float>, fp32_to_fp32, &localGetFp32Type)->Range(8, 8 << 10);
BENCHMARK_CAPTURE(BM_ContentRead<float>, fp32_to_fp32, &localGetFp32Type)->Range(8, 8 << 10);
BENCHMARK_CAPTURE(BM_ContentGetTempBuf<float>, fp32_to_fp32, &localGetFp32Type)->Range(8, 8 << 10);
