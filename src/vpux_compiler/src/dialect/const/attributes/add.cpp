//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// AddAttr::print
//

void vpux::Const::AddAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    if (getBias()) {
        printer.printAttribute(getBias());
    } else if (getBiasArray()) {
        printer.printAttribute(getBiasArray());
    }
    printer << ">";
}

//
// AddAttr::parse
//

mlir::Attribute vpux::Const::AddAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::Attribute attr;
    if (mlir::failed(parser.parseAttribute(attr))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
        return Const::AddAttr::get(floatAttr);
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
        return Const::AddAttr::get(arrayAttr);
    }

    return nullptr;
}

//
// AddAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::AddAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::AddAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// AddAttr::transform
//

Const::Content vpux::Const::AddAttr::transform(vpux::Const::Content& input) const {
    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()), mlir::Float32Type::get(getContext()),
                                            inferOutputSplat(input.isSplat(), input.getType()));

    llvm::MutableArrayRef<float> shiftedVals = output.getTempBuf<float>();

    if (getBias()) {
        // Single bias value applied to all elements
        const auto bias = static_cast<float>(getBias().getValue().convertToDouble());
        input.read([&](auto values) {
            for (size_t i = 0; i < shiftedVals.size(); ++i) {
                shiftedVals[i] = checked_cast<float>(values[i]) + bias;
            }
        });
    } else if (getBiasArray()) {
        // Array of bias values - broadcast across all elements
        const auto biasArray = getBiasArray().getValue();
        input.read([&](auto values) {
            for (size_t i = 0; i < shiftedVals.size(); ++i) {
                const auto biasIdx = checked_cast<size_t>(i % biasArray.size());
                const auto bias = static_cast<float>(
                        mlir::cast<mlir::FloatAttr>(biasArray[biasIdx]).getValue().convertToDouble());
                shiftedVals[i] = checked_cast<float>(values[i]) + bias;
            }
        });
    }

    return output;
}
