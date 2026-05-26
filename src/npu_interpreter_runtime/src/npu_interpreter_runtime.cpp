//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/runtime/npu_vm_runtime.hpp"

#include "vpux/utils/bytecode/virtual_machine/virtual_machine.hpp"

#include <intel_npu/utils/logger/logger.hpp>

#include <vector>

/// Opaque runtime object holding the VirtualMachine instance.
struct _npu_vm_runtime_handle_t {
    vpux::bytecode::VirtualMachine vm;
};

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeGetAPIVersion(npu_vm_runtime_version_t* pVersion) {
    if (pVersion == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    *pVersion = NPU_VM_RUNTIME_VERSION_CURRENT;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreate(const npu_vm_runtime_blob_desc_t* desc,
                                                                            npu_vm_runtime_handle_t* phRuntime,
                                                                            npu_vm_runtime_properties_t* pProperties) {
    auto log = intel_npu::Logger::global();
    log.setName("npuVMRuntimeCreate");

    if (desc == nullptr || phRuntime == nullptr || pProperties == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    if (desc->pInput == nullptr || desc->inputSize == 0) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        const std::vector<uint8_t> bytecode(desc->pInput, desc->pInput + desc->inputSize);

        auto* handle = new _npu_vm_runtime_handle_t();
        if (!handle->vm.parse(bytecode)) {
            delete handle;
            return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
        }
        *phRuntime = handle;
        // TODO: populate pProperties from parsed bytecode metadata
        pProperties->numOfSubGraphs = 0;
        pProperties->numOfGraphArgs = 0;
    } catch (const std::exception& e) {
        log.error("Exception while creating runtime: %s", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeDestroy(npu_vm_runtime_handle_t hRuntime) {
    if (hRuntime == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    delete hRuntime;
    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeGetMetadata(npu_vm_runtime_handle_t hRuntime, uint32_t /*argIndex*/,
                        ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                        ze_graph_argument_metadata_t* pGraphArgumentMetadata, int64_t* upperBound) {
    if (hRuntime == nullptr || pGraphArgumentProperties == nullptr || pGraphArgumentMetadata == nullptr ||
        upperBound == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement metadata retrieval
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeExecute(npu_vm_runtime_handle_t hRuntime,
                                                                             npu_vm_runtime_execute_params_t* pParams) {
    auto log = intel_npu::Logger::global();
    log.setName("npuVMRuntimeExecute");

    if (hRuntime == nullptr || pParams == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        hRuntime->vm.run();
    } catch (const std::exception& e) {
        log.error("Exception while executing model: %s", e.what());
        return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_VM_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimePredictOutputShape(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_predict_output_shape_params_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement output shape prediction
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeCreateMemRef(int64_t /*dimsCount*/, npu_vm_runtime_mem_ref_handle_t* phMemRef) {
    if (phMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement MemRef creation
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef) {
    if (hMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement MemRef destruction
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeSetMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef, const void* /*basePtr*/, const void* /*data*/,
                      int64_t /*offset*/, int64_t* /*pSizes*/, int64_t* /*pStrides*/, int64_t /*dimsCount*/) {
    if (hMemRef == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement MemRef update
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeParseMemRef(npu_vm_runtime_mem_ref_handle_t hMemRef, const void** pBasePtr, const void** pData,
                        int64_t* pOffset, int64_t* pSizes, int64_t* pStrides, int64_t* pDimsCount) {
    if (hMemRef == nullptr || pBasePtr == nullptr || pData == nullptr || pOffset == nullptr || pSizes == nullptr ||
        pStrides == nullptr || pDimsCount == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement MemRef parsing
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL npuVMRuntimeCreateExecutionContext(
        npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execution_context_handle_t* phExecutionHandle) {
    if (hRuntime == nullptr || phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement execution context creation
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeDestroyExecutionContext(npu_vm_runtime_execution_context_handle_t phExecutionHandle) {
    if (phExecutionHandle == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement execution context destruction
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

DLLEXPORT npu_vm_runtime_result_t NPU_VM_RUNTIME_APICALL
npuVMRuntimeUpdateMutableCommandList(npu_vm_runtime_handle_t hRuntime, npu_vm_runtime_execute_params_t* pParams,
                                     uint64_t* argIndexArray, uint64_t /*argIndexArraySize*/) {
    if (hRuntime == nullptr || pParams == nullptr || argIndexArray == nullptr) {
        return NPU_VM_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // TODO: implement mutable command list update
    return NPU_VM_RUNTIME_RESULT_ERROR_UNKNOWN;
}

#ifdef __cplusplus
}
#endif
