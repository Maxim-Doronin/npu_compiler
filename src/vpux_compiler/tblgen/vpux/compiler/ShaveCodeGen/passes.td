//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_SHAVE_CODE_GEN_PASSES
#define VPUX_COMPILER_SHAVE_CODE_GEN_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// ShaveCodeGen Passes
//=================================================================================

//
// OutlineCodeGenCapsules
//

def OutlineCodeGenCapsules : PassBase<"outline-codegen-capsules", "vpux::ModulePass"> {
    let summary = "Outline the CodeGenCapsules contents to external functions";

    let description = [{
        The pass outlines the Shave-directed CodeGenCapsules to external functions.
        In place of the capsules, it introduces VPU::GenericSwLayer ops that now represent calls to the shave functions.
    }];

    let constructor = "vpux::ShaveCodeGen::createOutlineCodeGenCapsulesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::func::FuncDialect"
    ];
}

//
// LowerMathToShaveIntrinsics
//

def LowerMathToShaveIntrinsics : PassBase<"lower-math-to-shave-intrinsics", "vpux::ModulePass"> {
    let summary = "Lower math operations to shave intrinsics";

    let description = [{
        Some math operations directly map to shave intrinsics in cases where there are no
        equivalent generic LLVM operations (for example tanh). In these cases there is no
        generic lowering path to the LLVM dialect.
        This pass inserts the required declarations for the shave intrisics in the software
        module and replaces the math operations with function calls.
    }];

    let constructor = "vpux::ShaveCodeGen::createLowerMathToShaveIntrinsicsPass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::math::MathDialect"
    ];
}


//
// AdaptLLVMFuncsForShave
//

def AdaptLLVMFuncsForShave : PassBase<"adapt-LLVM-funcs-for-shave", "vpux::ModulePass"> {
    let summary = "Adapt LLVM Functions for the Shave calling convention";

    let description = [{
        Shave has some expectations for the functions it calls via the management kernel.
        This pass adjusts the generated LLVM functions to be compatible with this calling convention.
        More exactly:
            1. Packs all the function arguments into a singular pointer type which is all-encapsulating
                Note: To accomplish this, pointer arithmetic is introduced in order to retrieve
                      the necessary data from the opaque pointer argument.
            2. Removes function return values
    }];

    let constructor = "vpux::ShaveCodeGen::createAdaptLLVMFuncsForShavePass()";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::math::MathDialect"
    ];
}

//
// EncapsulateCodeGenOps
//

def EncapsulateCodeGenOps : PassBase<"encapsulate-codegen-ops", "vpux::FunctionPass"> {
    let summary = "Encapsulate ShaveCodeGen supported ops into IE.CodeGenCapsule operations";

    let description = [{
        This pass aims at encapsulating all the ShaveCodeGen supported operations within
        IE::CodeGenCapsule ops. The triage for the encapsulation is done based on the
        ShaveCodeGenSupportedOpInterface. All ops implementing this interface will be wrapped.

        Additionally, this pass also constructs a FusionChainAnalysis which can be used in a
        succeding fusion pass. The construction of this analysis is also required for the
        well-functioning of the encapsulation logic.
    }];

    let constructor = "vpux::ShaveCodeGen::createEncapsulateCodeGenOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// EarlyCodeGenCapsuleFusion
//

def EarlyCodeGenCapsuleFusion : PassBase<"early-codegen-capsule-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse compatible CodeGenCapsules";

    let description = [{
        This pass aims at fusing compatible CodeGenCapsules such that the compute-abstracting
        operation within these capsules can be further fused.

        The fusion is done by using the FusionChainAnalysis and querying the pre-computed capsule chains.
        Because of this, EncapsulateCodeGenOps & the ahead-of-time construction of the FusionChainAnalysis
        is a hard requirement for the well-functioning of this pass.
    }];

    let constructor = "vpux::ShaveCodeGen::createEarlyCodeGenCapsuleFusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
