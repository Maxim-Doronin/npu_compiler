
//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0.
//

#ifndef VPUX_COMPILER_CORE_TYPES
#define VPUX_COMPILER_CORE_TYPES

include "mlir/IR/OpBase.td"

include "vpux/compiler/dialect/core/dialect.td"

def Bool8 : I<8>;
def I4 : I<4>;

class Core_Type<string name, list<Trait> traits = []>
    : TypeDef<Core_Dialect, name, traits> {
  let mnemonic = name;
}

def BoundedTensorType : Core_Type<"BoundedTensor"> {
    let extraClassDeclaration = [{
        static bool classof(mlir::Type type);
        static BoundedTensorType get(mlir::Type type, mlir::ArrayRef<int64_t> bounds);
        [[nodiscard]] Bounds getBounds() const;
        [[nodiscard]] BoundedTensorType changeBounds(mlir::ArrayRef<int64_t> bounds) const;
    }];
}

def DynamicDimsMaskTensorType : Core_Type<"DynamicDimsMaskTensor"> {
    let extraClassDeclaration = [{
        static bool classof(mlir::Type type);
        static DynamicDimsMaskTensorType get(mlir::Type type, mlir::ArrayRef<int64_t> dynamicDimsMask);
        [[nodiscard]] DynamicDimsMask getDynamicDimsMask() const;
        [[nodiscard]] DynamicDimsMaskTensorType changeDynamicDimsMask(mlir::ArrayRef<int64_t> dynamicDimsMask) const;
    }];
}

#endif
