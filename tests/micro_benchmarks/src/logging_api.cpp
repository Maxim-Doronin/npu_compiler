//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser/Parser.h>

#include <benchmark/benchmark.h>

namespace llvm {
template <>
struct format_provider<mlir::NamedAttribute> {
    static void format(const mlir::NamedAttribute& val, llvm::raw_ostream& stream, StringRef style) {
        stream << '(';
        llvm::format_provider<mlir::StringAttr>::format(val.getName(), stream, style);
        stream << ", ";
        val.getValue().print(stream);  // Note: there's no formatter for this!
        stream << ')';
    }
};
}  // namespace llvm

namespace {

struct TestState {
    mlir::DialectRegistry registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx;
    vpux::Logger log = vpux::Logger::global();

    // cached elements
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
    mlir::func::FuncOp funcOp;
    vpux::Const::DeclareOp constOp;

    TestState(mlir::StringRef source) {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<vpux::Const::ConstDialect>();

        log.setLevel(vpux::LogLevel::Error);

        moduleOp = mlir::parseSourceString<mlir::ModuleOp>(source, &ctx);
        VPUX_THROW_WHEN(moduleOp.get() == nullptr, "Failed to create module from source");

        moduleOp.get()->setAttr("intAttr", vpux::getIntAttr(&ctx, 42));
        moduleOp.get()->setAttr("flag", mlir::UnitAttr::get(&ctx));
        moduleOp.get()->setAttr("arrayAttr", vpux::getIntArrayAttr(&ctx, mlir::ArrayRef<int>({0, 1, 2, 3})));

        auto funcs = moduleOp->getOps<mlir::func::FuncOp>();
        VPUX_THROW_WHEN(funcs.empty(), "Failed to get the main function");
        funcOp = *funcs.begin();

        auto constOps = funcOp.getOps<vpux::Const::DeclareOp>();
        VPUX_THROW_WHEN(constOps.empty(), "Failed to get const operation");
        constOp = *constOps.begin();
    }
};

constexpr llvm::StringLiteral TEST_IR = R"(
module @main {
    func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>, [#const.Add<1.0>]
        return %arg0 : tensor<2x2xf16>
    }
})";

}  // namespace

// Note: logging is super fast, so repeat the same operation multiple times to
// average-out the noisy results.
constexpr size_t INNER_ITER_COUNT = 1000;

template <typename Callable>
static void BM_LogCppFunction(benchmark::State& state, Callable getLogValue) {
    TestState testState(TEST_IR);

    for (auto _ : state) {
        for (size_t i = 0; i < INNER_ITER_COUNT; ++i) {
            testState.log.trace("{0}", getLogValue(testState));
        }
    }
}

namespace {
#define LOG_TRACE(log, format, ...)            \
    if (log.isActive(vpux::LogLevel::Trace)) { \
        log.trace(format, __VA_ARGS__);        \
    }
}  // namespace

template <typename Callable>
static void BM_LogMacro(benchmark::State& state, Callable getLogValue) {
    TestState testState(TEST_IR);

    for (auto _ : state) {
        for (size_t i = 0; i < INNER_ITER_COUNT; ++i) {
            LOG_TRACE(testState.log, "{0}", getLogValue(testState));
        }
    }
}

// benchmarks:

mlir::StringLiteral emptyLiteral(TestState&) {
    return "";
}

BENCHMARK_CAPTURE(BM_LogCppFunction, nothing, emptyLiteral);
BENCHMARK_CAPTURE(BM_LogMacro, nothing, emptyLiteral);

std::string longStr(TestState&) {
    return "this is a relatively long string of smth. lorem ipsum dolor sit amet...";
}

BENCHMARK_CAPTURE(BM_LogCppFunction, long_std_str, longStr);
BENCHMARK_CAPTURE(BM_LogMacro, long_std_str, longStr);

auto symbolName(TestState& state) {
    return state.moduleOp->getSymName();
}

BENCHMARK_CAPTURE(BM_LogCppFunction, module_name, symbolName);
BENCHMARK_CAPTURE(BM_LogMacro, module_name, symbolName);

mlir::Location location(TestState& state) {
    return state.moduleOp->getLoc();
}

BENCHMARK_CAPTURE(BM_LogCppFunction, module_loc, location);
BENCHMARK_CAPTURE(BM_LogMacro, module_loc, location);

auto moduleAttrs(TestState& state) {
    return state.moduleOp.get()->getAttrs();
}

BENCHMARK_CAPTURE(BM_LogCppFunction, module_attrs_user_set, moduleAttrs);
BENCHMARK_CAPTURE(BM_LogMacro, module_attrs_user_set, moduleAttrs);

mlir::FunctionType funcType(TestState& state) {
    return state.funcOp.getFunctionType();
}

BENCHMARK_CAPTURE(BM_LogCppFunction, func_op_type, funcType);
BENCHMARK_CAPTURE(BM_LogMacro, func_op_type, funcType);

mlir::func::FuncOp funcOp(TestState& state) {
    return state.funcOp;
}

BENCHMARK_CAPTURE(BM_LogCppFunction, func_op, funcOp);
BENCHMARK_CAPTURE(BM_LogMacro, func_op, funcOp);

mlir::Type constType(TestState& state) {
    return state.constOp.getType();
}

BENCHMARK_CAPTURE(BM_LogCppFunction, declare_op_type, constType);
BENCHMARK_CAPTURE(BM_LogMacro, declare_op_type, constType);

auto constOpTransforms(TestState& state) {
    return state.constOp.getContentAttr().getTransformations();
}

BENCHMARK_CAPTURE(BM_LogCppFunction, declare_op_transforms, constOpTransforms);
BENCHMARK_CAPTURE(BM_LogMacro, declare_op_transforms, constOpTransforms);

vpux::Const::DeclareOp constOp(TestState& state) {
    return state.constOp;
}

BENCHMARK_CAPTURE(BM_LogCppFunction, declare_op, constOp);
BENCHMARK_CAPTURE(BM_LogMacro, declare_op, constOp);
