//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_RESOURCES
#define VPUX_COMPILER_DIALECT_IE_OPS_RESOURCES

include "vpux/compiler/dialect/IE/ops/base.td"
include "vpux/compiler/dialect/IE/ops_interfaces.td"
include "vpux/compiler/dialect/IE/attributes.td"
include "vpux/compiler/dialect/core/ops_interfaces.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def IE_ExecutorResourceOp :
        IE_Op<
            "ExecutorResource",
            [
                IE_ResourceOpInterface,
                IE_ComputeResourceOpInterface,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about executor resource";

    let description = [{
        The executor resource is defined by the following attributes:

          * sym_name - kind of the executor.
          * count - number of executor units.
          * activity_factor - power activity factor.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$count,
        OptionalAttr<F64Attr>:$activity_factor
    );

    let extraClassDeclaration = [{
        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name
    }];

    let hasVerifier = 1;
}

def IE_MemoryResourceOp :
        IE_Op<
            "MemoryResource",
            [
                IE_ResourceOpInterface,
                Symbol
            ]
        > {
    let summary = "Information about memory resource";

    let description = [{
        The memory resource is defined by the following attributes:

          * sym_name - kind of memory space.
          * byteSize - size in bytes of memory space.
          * offset - optional offset attribute if this resource is part of larger memory space
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$byteSize,
        OptionalAttr<IntAttr>:$offset
    );

    let extraClassDeclaration = [{
        vpux::Byte size() { return vpux::Byte(getByteSize()); }

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        $byteSize `bytes` `of` $sym_name (`offset` $offset^)?
        attr-dict
    }];
}

def IE_TileResourceOp :
        IE_Op<
            "TileResource",
            [
                IE_ResourceOpInterface,
                IE_ComputeResourceOpInterface,
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about tile resource";

    let description = [{
        The tile resource is defined by the following attributes:
          * sym_name - kind of the executor.
          * count - number of executor units.
          * proc_freq - processor frequency (in MHz).
          * activity_factor - power activity factor.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        IntAttr:$count,
        OptionalAttr<F64Attr>:$proc_freq,
        OptionalAttr<F64Attr>:$activity_factor
    );

    let regions = (region
        SizedRegion<1>:$subExecutors
    );

    let extraClassDeclaration = [{
        mlir::FloatAttr getProcessorFrequency() {
            return getProcFreqAttr();
        }

        void setProcessorFrequency(mlir::FloatAttr procFreqAttr) {
            setProcFreqAttr(procFreqAttr);
        }

        bool hasProcessorFrequency() {
            return getProcFreqAttr() != nullptr;
        }

        vpux::IE::ExecutorResourceOp addSubExecutor(mlir::SymbolRefAttr executor, size_t count);
        bool hasSubExecutor(mlir::SymbolRefAttr executor);
        vpux::IE::ExecutorResourceOp getSubExecutor(mlir::SymbolRefAttr executor);
        vpux::IE::MemoryResourceOp addAvailableMemory(mlir::SymbolRefAttr memSpace, Byte size);

        bool hasAvailableMemory(mlir::SymbolRefAttr memSpace);

        vpux::IE::MemoryResourceOp getAvailableMemory(mlir::SymbolRefAttr memSpace);

        template <typename Enum, typename OutT = vpux::IE::ExecutorResourceOp>
        using exec_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum, typename OutT = vpux::IE::MemoryResourceOp>
        using mem_resource_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        exec_resource_if<Enum> addSubExecutor(Enum kind, size_t count) {
            return addSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)), count);
        }

        template <typename Enum, typename OutT = bool>
        using bool_if = enable_t<OutT, std::is_enum<Enum>, details::HasStringifyEnum<Enum>>;

        template <typename Enum>
        bool_if<Enum> hasSubExecutor(Enum kind) {
            return hasSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        exec_resource_if<Enum> getSubExecutor(Enum kind) {
            return getSubExecutor(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        mem_resource_if<Enum> addAvailableMemory(Enum kind, Byte size) {
            return addAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)), size);
        }

        template <typename Enum>
        bool_if<Enum> hasAvailableMemory(Enum kind) {
            return hasAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        template <typename Enum>
        mem_resource_if<Enum> getAvailableMemory(Enum kind) {
            return getAvailableMemory(mlir::SymbolRefAttr::get(getContext(), stringifyEnum(kind)));
        }

        static mlir::StringRef getDefaultDialect() {
            return "IE";
        }
    }];

    let assemblyFormat = [{
        attr-dict
        $count `of` $sym_name (`at` $proc_freq^`MHz`)?
        custom<OptionalBlockRegion>($subExecutors)
    }];

    let hasVerifier = 1;
}

#endif
