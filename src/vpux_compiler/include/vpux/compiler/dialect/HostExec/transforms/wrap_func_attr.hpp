//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include "vpux/compiler/core/layers.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <cassert>
#include <map>
#include <optional>
#include <vector>

namespace vpux {
struct WrapFuncData final {
    WrapFuncData(StringRef wrapperFunctionName,
                 const mlir::DenseMap<StringRef, StringRef>& optionalParams = mlir::DenseMap<StringRef, StringRef>(0));
    WrapFuncData(StringRef wrapperFunctionName);
    WrapFuncData(StringRef wrapperFunctionName, bool needToDeleteWrappedFunc);

    StringRef getWrapperFunctionNameValue() const;
    bool needToDeleteWrapped() const;

    static WrapFuncData deserialize(const SmallVector<StringRef>& array);
    SmallVector<std::string> serialize() const;
    std::string to_string() const;

private:
    WrapFuncData(const mlir::DenseMap<StringRef, StringRef>& params);
    void init(StringRef wrapperFunctionName, const mlir::DenseMap<StringRef, StringRef>& optionalParams);

    static constexpr StringRef getWrapperFunctionFieldName() {
        return "wrapper";
    }

    static constexpr StringRef getNeedToDeleteWrapperDeclarationFieldName() {
        return "deleteWrapped";
    }

    std::string wrapperFuncName;
    std::optional<bool> deleteWrapped;
};

class WrapFuncDataAttributeView {
    WrapFuncData data;

public:
    WrapFuncDataAttributeView(const WrapFuncDataAttributeView&) = default;
    WrapFuncDataAttributeView(WrapFuncDataAttributeView&&) = default;
    ~WrapFuncDataAttributeView() = default;
    WrapFuncDataAttributeView& operator=(const WrapFuncDataAttributeView&) = default;
    WrapFuncDataAttributeView& operator=(WrapFuncDataAttributeView&&) = default;

    static constexpr std::string_view name() {
        return "wrapFunctionAttr";
    }

    const WrapFuncData& getFuncData() const;
    static std::optional<WrapFuncDataAttributeView> extract(mlir::func::FuncOp funcOp);

    template <class... Args>
    static WrapFuncDataAttributeView inject(mlir::func::FuncOp funcOp, Args&&... args) {
        WrapFuncDataAttributeView view{std::forward<Args>(args)...};
        view.injectImpl(funcOp);
        return view;
    }

private:
    template <class... Args>
    WrapFuncDataAttributeView(Args&&... args): data(std::forward<Args>(args)...) {
    }

    void injectImpl(mlir::func::FuncOp funcOp) const;
};
}  // namespace vpux
