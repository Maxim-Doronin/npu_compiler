// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace type {

QuantileFloatType QuantileFloatType::getNF4(mlir::MLIRContext* ctx, ArrayRef<double> quantiles) {
    return NF4Type::get(ctx, 4, quantiles);
}

QuantileFloatType QuantileFloatType::getQuantileFloat(MLIRContext* ctx, unsigned bitWidth, ArrayRef<double> quantiles) {
    if (bitWidth == 4) {
        return NF4Type::get(ctx, bitWidth, quantiles);
    }
    llvm_unreachable("unexpected quantile float type");
}

bool QuantileFloatType::classof(mlir::Type type) {
    return mlir::isa<NF4Type>(type);
}

unsigned QuantileFloatType::getWidth() const {
    if (mlir::isa<NF4Type>(*this)) {
        return 4;
    }
    llvm_unreachable("unexpected quantile float type");
}

ArrayRef<double> QuantileFloatType::getQuantiles() const {
    if (auto nf4 = mlir::dyn_cast<NF4Type>(*this)) {
        return nf4.getQuantiles();
    }
    llvm_unreachable("unexpected quantile float type");
}

void QuantileFloatType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer << getWidth();
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

mlir::Type QuantileFloatType::parse(mlir::AsmParser& parser) {
    uint32_t width = 0;
    SmallVector<double, 1> quantiles;

    if (parser.parseLess()) {
        return nullptr;
    }

    if (parser.parseInteger(width)) {
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

    return getQuantileFloat(parser.getContext(), width, quantiles);
}

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

NF4Type NF4Type::get(mlir::MLIRContext* ctx, unsigned width, ArrayRef<double> quantiles) {
    if (!quantiles.empty()) {
        VPUX_THROW_UNLESS(quantiles.size() == std::pow(2, width), "quantiles array size needs to be equal to 2^width");
        return Base::get(ctx, width, quantiles);
    }

    return Base::get(ctx, width, NF4Type::specQuantiles);
}

unsigned NF4Type::getWidth() const {
    return 4;
}

ArrayRef<double> NF4Type::getSpecQuantiles() {
    return NF4Type::specQuantiles;
}

ArrayRef<double> NF4Type::getQuantiles() const {
    return getImpl()->getQuantiles();
}

}  // namespace type
}  // namespace vpux
