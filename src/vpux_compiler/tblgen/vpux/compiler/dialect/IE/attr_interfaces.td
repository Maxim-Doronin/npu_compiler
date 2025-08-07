//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_IE_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

def IE_ChannelAgnosticAttr : NativeAttrTrait<"ChannelAgnosticAttr"> {
    let cppNamespace = "vpux::IE";
}

//
// PostOpAttrInterface
//

def IE_PostOpAttrInterface : AttrInterface<"PostOpAttr"> {
    let description = [{
        Interface PostOp attributes.
    }];

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Returns the name of the post-op attribute.",
            "::mlir::StringLiteral", "getName", (ins),
            [{}],
            [{
                return ConcreteAttr::name;
            }]
        >,

        InterfaceMethod<
            "Checks if the post-op has the same effect on all the channels of its input.",
            "bool", "isChannelAgnostic", (ins),
            [{}],
            [{
                return mlir::cast<mlir::Attribute>($_attr).template hasTrait<vpux::IE::ChannelAgnosticAttr>();
            }]
        >,
    ];
}

#endif
