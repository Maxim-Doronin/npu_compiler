//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/core/types/quantile_float/dialect.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace type {

QuantileType QuantileType::get(mlir::MLIRContext* ctx, mlir::Type storageType, mlir::Type quantileType,
                               ArrayRef<double> quantiles) {
    return Base::get(ctx, storageType, quantileType, quantiles);
}

bool QuantileType::classof(mlir::Type type) {
    return mlir::isa<NF4Type>(type) || type.getTypeID() == mlir::TypeID::get<QuantileType>();
}

mlir::Type QuantileType::getStorageType() const {
    return static_cast<ImplType*>(impl)->getStorageType();
}

mlir::Type QuantileType::getQuantileType() const {
    return static_cast<ImplType*>(impl)->getQuantileType();
}

ArrayRef<double> QuantileType::getQuantiles() const {
    return static_cast<ImplType*>(impl)->getQuantiles();
}

void QuantileType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer << getStorageType();
    printer << ":";
    printer << getQuantileType();
    printer << ", ";

    ArrayRef<double> quantiles = this->getQuantiles();
    printer << "{";
    llvm::interleave(
            llvm::seq<size_t>(0, quantiles.size()), printer,
            [&](size_t index) {
                printer << quantiles[index];
            },
            ",");
    printer << "}";

    printer << ">";
}

mlir::Type QuantileType::parse(mlir::AsmParser& parser) {
    Type storageType;
    Type quantileType;
    SmallVector<double, 1> quantiles;

    if (parser.parseLess()) {
        return nullptr;
    }

    if (parser.parseType(storageType)) {
        return nullptr;
    }

    if (parser.parseColon()) {
        return nullptr;
    }

    if (parser.parseType(quantileType)) {
        return nullptr;
    }

    if (parser.parseComma()) {
        return nullptr;
    }

    if (parser.parseLBrace()) {
        return nullptr;
    }

    do {
        quantiles.emplace_back();
        if (parser.parseFloat(quantiles.back())) {
            return nullptr;
        }
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
        return nullptr;
    }

    if (parser.parseGreater()) {
        return nullptr;
    }

    return get(parser.getContext(), storageType, quantileType, quantiles);
}

bool QuantileType::shouldDefaultToSigned() const {
    Type storageType = getStorageType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
        return !intType.isUnsigned();
    }
    // Always true for float types
    return true;
}

unsigned QuantileType::getStorageWidth() const {
    return getStorageType().getIntOrFloatBitWidth();
}

int64_t QuantileType::getDefaultMaximum([[maybe_unused]] bool isSigned) const {
    if (isSigned) {
        return (1LL << (getStorageWidth() - 1)) - 1;
    }
    return (1LL << getStorageWidth()) - 1;
}

int64_t QuantileType::getDefaultMinimum([[maybe_unused]] bool isSigned) const {
    if (isSigned) {
        return -(1LL << (getStorageWidth() - 1));
    }
    return 0;
}

std::string QuantileType::getStorageTypeName([[maybe_unused]] bool isSigned) const {
    std::string result = "!QuantileType.quantile<";
    llvm::raw_string_ostream os(result);
    os << getStorageType() << ":" << getQuantileType() << ", {";

    ArrayRef<double> quantiles = this->getQuantiles();
    llvm::interleave(
            llvm::seq<size_t>(0, quantiles.size()), os,
            [&](size_t index) {
                os << quantiles[index];
            },
            ",");

    os << "}>";
    os.flush();
    return result;
}

bool QuantileType::isPacked() const {
    return getStorageWidth() <= 4;
}

unsigned QuantileType::getLogicalBitWidth() const {
    return getStorageWidth();
}

unsigned QuantileType::getElementsPerByte() const {
    unsigned width = getStorageWidth();
    return width > 0 ? 8 / width : 0;
}

std::optional<unsigned> QuantileType::getPreferredAlignmentBytes() const {
    return std::nullopt;
}

//===----------------------------------------------------------------------===//
// NF4Type
//===----------------------------------------------------------------------===//

const SmallVector<double> NF4Type::specQuantiles{-1.0,
                                                 -0.6961928009986877,
                                                 -0.5250730514526367,
                                                 -0.39491748809814453,
                                                 -0.28444138169288635,
                                                 -0.18477343022823334,
                                                 -0.09105003625154495,
                                                 0.0,
                                                 0.07958029955625534,
                                                 0.16093020141124725,
                                                 0.24611230194568634,
                                                 0.33791524171829224,
                                                 0.44070982933044434,
                                                 0.5626170039176941,
                                                 0.7229568362236023,
                                                 1.0};

NF4Type NF4Type::get(mlir::MLIRContext* ctx, Type storageType, Type quantileType, ArrayRef<double> quantiles) {
    if (!quantiles.empty()) {
        VPUX_THROW_UNLESS(quantiles.size() == 16, "quantiles array size of nf4 type needs to be equal to 16");
        return Base::get(ctx, storageType, quantileType, quantiles);
    }

    return Base::get(ctx, storageType, quantileType, NF4Type::specQuantiles);
}

ArrayRef<double> NF4Type::getSpecQuantiles() {
    return NF4Type::specQuantiles;
}

bool NF4Type::classof(mlir::Type type) {
    return type.getTypeID() == mlir::TypeID::get<NF4Type>();
}

}  // namespace type
}  // namespace vpux
