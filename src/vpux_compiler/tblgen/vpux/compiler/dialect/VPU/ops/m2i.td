//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_M2I
#define VPUX_COMPILER_DIALECT_VPU_OPS_M2I

include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

def VPU_M2IColorConvertOp :
        VPU_LayerOp<
            "M2I.ColorConvert"
        > {
    let summary = "M2I version for color-convert operations";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,
        IE_ColorFmtAttr:$inFmt,
        IE_ColorFmtAttr:$outFmt
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::YuvToRgbOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;
}


def VPU_M2IResizeOp :
        VPU_LayerOp<
            "M2I.Resize"
        > {
    let summary = "M2I version for resize operations";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,

        I64ArrayAttr:$sizes,
        I64ArrayAttr:$axes,
        VPU_M2iInterpAttr:$interp
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::InterpolateOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;
}


def VPU_M2INormOp :
        VPU_LayerOp<
            "M2I.Norm"
        > {
    let summary = "M2I version for BatchNormInference";

    let arguments = (ins
        4DTensorOf<[F16]>:$input,

        F64ArrayAttr:$gamma_value,
        F64ArrayAttr:$beta_value,
        F64ArrayAttr:$mean_value,
        F64ArrayAttr:$variance_value,

        F64Attr:$eps
    );

    let results = (outs
         4DTensorOf<[F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];

    let extraClassDeclaration = [{
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem);
        static bool fitIntoCMX(mlir::Operation* op, vpux::NDTypeInterface input, vpux::NDTypeInterface output);
        static bool isSupported(vpux::IE::BatchNormInferenceOp origOp, vpux::LogCb logCb, bool checkLayout = false,
                                bool checkChannelAlignment = false);
    }] # baseExtraClassDeclaration;
}


def VPU_M2ITaskOp :
        VPU_LayerOp<
            "M2I.Task"
        > {
    let summary = "M2I full task op";

    let arguments = (ins
        4DTensorOf<[UI8, F16]>:$input,

        BoolAttr:$do_csc,
        BoolAttr:$do_norm,
        VPU_M2iColorFmtAttr:$inFmt,
        VPU_M2iColorFmtAttr:$outFmt,
        UnitAttr:$chroma_in_reverse_channels,
        UnitAttr:$chroma_out_reverse_channels,
        UnitAttr:$luma_in_reverse_channels,
        UnitAttr:$luma_out_reverse_channels,
        OptionalAttr<I64ArrayAttr>:$sizes,
        OptionalAttr<I64ArrayAttr>:$axes,
        OptionalAttr<F64ArrayAttr>:$norm,
        DefaultValuedAttr<VPU_M2iInterpAttr, "vpux::VPU::M2iInterp::NEAREST">:$interp
    );

    let results = (outs
         4DTensorOf<[UI8, F16]>:$output
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        custom<OptionalTypes>(type($input)) ``
        `->` type(results)
    }];
}

#endif
