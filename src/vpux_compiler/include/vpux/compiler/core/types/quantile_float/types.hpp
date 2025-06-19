// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include "vpux/compiler/core/types/quantile_float/type_detail.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/array_ref.hpp"

using namespace mlir;
using namespace llvm;

namespace vpux {
namespace type {

//===----------------------------------------------------------------------===//
// QuantileFloatType
//===----------------------------------------------------------------------===//

class QuantileFloatType : public mlir::Type {
public:
    using ImplType = vpux::detail::QuantileFloatTypeStorage;
    using Type::Type;

    // Return the bitwidth of the storage type.
    unsigned getStorageTypeIntegralWidth() const;

    // Get the underlying type used for to store raw values.
    Type getStorageType() const;

    // Get primitive expressed type of data in quantiles.
    // Note that we may convert FP8 data to FP16 for storage,
    // but we should treat its expressed type as FP8 rather than FP16.
    Type getQuantileType() const;

    /// Return the quantile table of this float type.
    ArrayRef<double> getQuantiles() const;

    // Get a quantile float type with specified quantile table.
    static QuantileFloatType get(MLIRContext* ctx, Type storageType, Type quantileType,
                                 ArrayRef<double> quantiles = {});

    // Get NF4 instance.
    static QuantileFloatType getNF4(mlir::MLIRContext* ctx, Type storageType, Type quantileType,
                                    ArrayRef<double> quantiles = {});

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool classof(mlir::Type type);

    // Printer
    void print(mlir::AsmPrinter& printer) const;

    // Parser
    static mlir::Type parse(mlir::AsmParser& parser);

    static constexpr llvm::StringLiteral getMnemonic() {
        return {"quantileFloat"};
    }
};

class NF4Type : public mlir::Type::TypeBase<NF4Type, QuantileFloatType, vpux::detail::QuantileFloatTypeStorage> {
private:
    static const SmallVector<double> specQuantiles;

public:
    using Base::Base;
    static NF4Type get(mlir::MLIRContext* ctx, Type storageType, Type quantileType, ArrayRef<double> quantiles = {});
    static constexpr llvm::StringLiteral name = "nf4";
    static ArrayRef<double> getSpecQuantiles();
    static bool classof(mlir::Type type);
};

}  // namespace type
}  // namespace vpux
