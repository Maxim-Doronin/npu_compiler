//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#define NOMINMAX

#include "level_zero_wrapper.h"

#include <stdio.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/utils/IE/network_metadata.hpp"
#include "vpux_headers/serial_metadata.hpp"
#include "ze_graph_ext.h"

/* #define TEST */

// Workaround for win specific save funtions
#ifndef _WIN32
#include <errno.h>
#include <string.h>

// Basic strcpy_s implementation for Linux
inline error_t strcpy_s(char* dest, size_t destsz, const char* src) {
    if (dest == NULL || src == NULL || destsz == 0) {
        return EINVAL;
    }

    size_t srcsz = strlen(src);
    if (srcsz >= destsz) {
        dest[0] = '\0';  // Null terminate even in case of failure
        return ERANGE;
    }

    strcpy(dest, src);
    return 0;
}

template <size_t size>
inline error_t strcpy_s(char (&dest)[size], const char* src) {
    return strcpy_s(dest, size, src);
}

// Basic memcpy_s implementation for Linux
inline error_t memcpy_s(void* dest, size_t destsz, const void* src, size_t count) {
    if (dest == NULL || src == NULL) {
        return EINVAL;
    }

    if (destsz < count) {
        return ERANGE;
    }

    memcpy(dest, src, count);
    return 0;
}
#endif

// End of workaround for win specific save funtions

#define RETURN_SUCCESS() return static_cast<uint32_t>(ZE_RESULT_SUCCESS);

#ifdef TEST
NPU_API(void*) npu_level_zero_alloc(int64_t size, void*) {
    printf("npu_level_zero_alloc was called %lld\n", size);
#if !defined(WIN32)
    void* result = aligned_alloc(64, size);
#else
    void* result = _aligned_malloc(size, 64);
#endif
    return result;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void* commandList) {
    printf("npu_level_zero_append_memory_copy was called %ld\n", size);
    RETURN_SUCCESS();
}

NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList) {
    printf("npu_level_zero_append_barrier was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue) {
    printf("npu_level_zero_create_graph was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue) {
    printf("npu_level_zero_create_graphs was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_execute_graph(void** input, int32_t numInputs, void** output, int32_t numOutputs, void* kernel,
                             int64_t kernelSize, void* context, void* device, void* ddiTable, void* commandList) {
    printf("npu_level_zero_execute_graph was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_submit_commandlist(void* commandList, void* commandQueue, void* fence, void* event) {
    printf("npu_level_zero_submit_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    printf("npu_level_zero_get_last_error was called\n");
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void* commandList) {
    printf("npu_level_zero_reset_commandlist was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandList, int32_t numCommandLists) {
    printf("npu_level_zero_reset_commandlists was called\n");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* levelZeroMetadata, void* inputDescs,
                                    void* outputDescs) {
    printf("npu_level_zero_metadata was called\n");
    RETURN_SUCCESS();
}

#else

struct graph_info {
    ze_graph_handle_t graphHandle;
    uint32_t numArgs;
    uint32_t numInputArgs;
    graph_info(): graphHandle(nullptr), numArgs(0), numInputArgs(0) {
    }
    graph_info(ze_graph_handle_t handle, uint32_t numArgs, uint32_t numInputArgs)
            : graphHandle(handle), numArgs(numArgs), numInputArgs(numInputArgs) {
    }
};

constexpr size_t max_message_length = 256;
bool dumpKernelName = false;
bool enabledFailedArgumentDesc = false;

static char lastErrorMessage[max_message_length];
#define ERROR_HANDLE(result, msg)                                                         \
    if (result != ZE_RESULT_SUCCESS) {                                                    \
        size_t size = std::min(static_cast<size_t>(max_message_length - 1), strlen(msg)); \
        std::strncpy(lastErrorMessage, msg, size);                                        \
        lastErrorMessage[size] = '\0';                                                    \
        return static_cast<int32_t>(result);                                              \
    }

ze_graph_dditable_ext_t* ddiTableHandle = nullptr;
std::map<void*, graph_info>* graphMap = nullptr;

struct scratch_buffer {
    ze_context_handle_t contextHandle;
    void* data;
    int64_t size;
};

struct ze_memory_deleter {
    void operator()(scratch_buffer* buffer) const {
        if (buffer != nullptr) {
            if (buffer->data != nullptr) {
                zeMemFree(buffer->contextHandle, buffer->data);
            }
        }
    }
};

std::unique_ptr<scratch_buffer, ze_memory_deleter> scratchBuffer = nullptr;

// shared library initialization function for ExecutionEngine
NPU_API(void) __mlir_execution_engine_init() {
    graphMap = new std::map<void*, graph_info>();

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    auto dumpKernelNameFlag = std::getenv("ENABLE_PRINT_HOSTCOMPILE_KERNEL_NAME");
    if (dumpKernelNameFlag != nullptr && strcmp(dumpKernelNameFlag, "1")) {
        dumpKernelName = true;
    }

    auto enableFailedArgumentFlag = std::getenv("ENABLE_FAILED_ARGUMENT_DESCRIPTOR");
    if (enableFailedArgumentFlag != nullptr && strcmp(enableFailedArgumentFlag, "1")) {
        enabledFailedArgumentDesc = true;
    }
#endif
}

// shared library destroy function for ExecutionEngine
NPU_API(void) __mlir_execution_engine_destroy() {
    if (graphMap) {
        if (ddiTableHandle) {
            for (auto& g : *graphMap) {
                ddiTableHandle->pfnDestroy(g.second.graphHandle);
                g.second.graphHandle = nullptr;
            }
        }
        delete graphMap;
        graphMap = nullptr;
    }

    if (scratchBuffer) {
        scratchBuffer.reset();
    }
}

NPU_API(void*) npu_level_zero_alloc(int64_t bytes, void* context) {
    if (scratchBuffer != nullptr && scratchBuffer->size >= bytes) {
        return scratchBuffer->data;
    }

    if (scratchBuffer == nullptr || scratchBuffer->size < bytes) {
        ze_host_mem_alloc_flag_t flag = {};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                         static_cast<ze_host_mem_alloc_flags_t>(flag)};
        auto contextHandle = static_cast<ze_context_handle_t>(context);
        void* data = nullptr;
        // user is responsible for alignment, so we pass 0 for alignment
        int32_t res = zeMemAllocHost(contextHandle, &desc, bytes, /*no alignment*/ 0, &data);

        if (res == ZE_RESULT_SUCCESS) {
            scratchBuffer.reset(new scratch_buffer{contextHandle, data, bytes});

            return data;
        }
    }

    return nullptr;
}

NPU_API(int32_t) npu_level_zero_append_memory_copy(void* src, void* dst, int64_t size, void** commandList) {
    auto commandListHandle =
            (commandList == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandList);
    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid nullpointer");
    }
    auto result = zeCommandListAppendMemoryCopy(commandListHandle, dst, src, size, nullptr, 0, nullptr);
    ERROR_HANDLE(result, "Failed to append memory copy");

    RETURN_SUCCESS();
}

NPU_API(int32_t) npu_level_zero_append_barrier(void* commandList) {
    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto result = zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        ERROR_HANDLE(result, "Failed to append barrier");
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graph(void* kernel, int64_t kernelSize, void* context, void* device, void* ddiTable,
                            void* commandList, void* commandQueue) {
    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    if (::ddiTableHandle == nullptr) {
        ::ddiTableHandle = ddiTableHandle;
    }

    ze_graph_desc_t desc = {
            ZE_STRUCTURE_TYPE_GRAPH_DESC,       nullptr, ZE_GRAPH_FORMAT_NATIVE, static_cast<size_t>(kernelSize),
            reinterpret_cast<uint8_t*>(kernel), nullptr};

    auto contextHandle = static_cast<ze_context_handle_t>(context);
    auto deviceHandle = static_cast<ze_device_handle_t>(device);

    ze_pfnGraphCreate_ext_t pfnCreate = ddiTableHandle->pfnCreate;
    ze_graph_handle_t graphHandle = nullptr;
    auto result = pfnCreate(contextHandle, deviceHandle, &desc, &graphHandle);
    ERROR_HANDLE(result, "Failed to create graph");

    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
    auto numInputArguments = 0;
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;

        ze_graph_argument_property_strides_t strides{ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        arg3.pNext = reinterpret_cast<void*>(&strides);
        result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
        ERROR_HANDLE(result, "Failed to get argument properties");

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            numInputArguments++;
        }
    }

    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize = ddiTableHandle->pfnAppendGraphInitialize;

    if (commandListHandle != nullptr) {
        result = pfnAppendGraphInitialize(commandListHandle, graphHandle, /*profiling_query_handle*/ nullptr, 0,
                                          nullptr);
        ERROR_HANDLE(result, "Failed to append graph initialize");
    }

    (*graphMap)[kernel] = graph_info(graphHandle, props.numGraphArgs, numInputArguments);

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_create_graphs(void** kernels, int64_t* kernelSizes, int32_t numKernels, void* context, void* device,
                             void* ddiTable, void* commandList, void* commandQueue) {
    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);
    auto commandListHandle = static_cast<ze_command_list_handle_t>(commandList);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize = ddiTableHandle->pfnAppendGraphInitialize;

    for (int32_t kernelIndex = 0; kernelIndex < numKernels; ++kernelIndex) {
        ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES,
                                nullptr,
                                ZE_GRAPH_FORMAT_NATIVE,
                                static_cast<size_t>(kernelSizes[kernelIndex]),
                                reinterpret_cast<uint8_t*>(kernels[kernelIndex]),
                                nullptr};

        auto contextHandle = static_cast<ze_context_handle_t>(context);
        auto deviceHandle = static_cast<ze_device_handle_t>(device);

        ze_pfnGraphCreate_ext_t pfnCreate = ddiTableHandle->pfnCreate;
        ze_graph_handle_t graphHandle = nullptr;
        auto result = pfnCreate(contextHandle, deviceHandle, &desc, &graphHandle);
        ERROR_HANDLE(result, "Failed to create graph");

        ze_graph_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        result = ddiTableHandle->pfnGetProperties(graphHandle, &props);
        auto numInputArguments = 0;
        for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
            ze_graph_argument_properties_3_t arg3{};
            arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
            result = ddiTableHandle->pfnGetArgumentProperties3(graphHandle, index, &arg3);
            ERROR_HANDLE(result, "Failed to get argument properties\n");

            if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
                numInputArguments++;
            } else {
                break;
            }
        }

        result = pfnAppendGraphInitialize(commandListHandle, graphHandle, /*profiling_query_handle*/ nullptr, 0,
                                          nullptr);
        ERROR_HANDLE(result, "Failed to append graph initialize");

        (*graphMap)[kernels[kernelIndex]] = graph_info(graphHandle, props.numGraphArgs, numInputArguments);
    }

    RETURN_SUCCESS();
}

int32_t set_arguments(uint64_t index, const vpux::HostExec::MemRefDesc& desc, ze_graph_handle_t graphHandle,
                      ze_graph_dditable_ext_t* ddiTableHandle) {
    ze_result_t result = ZE_RESULT_SUCCESS;

    if (ZE_GRAPH_EXT_VERSION_CURRENT >= ZE_GRAPH_EXT_VERSION_1_15) {
        // Below is an example implementation.
        // When a new graph ext is available in master, this will be finalized.
        //
        ze_graph_argument_value_tensor_t tensor_value{
                ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_TENSOR, nullptr,
                reinterpret_cast<void*>(reinterpret_cast<uint64_t>(desc.data) + desc.elementByteSize * desc.offset)};

        // Strides information
        ze_graph_argument_value_strides_t tensor_strides = {};
        tensor_strides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_STRIDES;
        tensor_strides.pNext = nullptr;
        for (auto dim = 0; dim < desc.dimCount; dim++) {
            // store strides in reverse order
            tensor_strides.userStrides[dim] = static_cast<uint32_t>(desc.strides[(desc.dimCount - 1) - dim]);
        }
        tensor_value.pNext = reinterpret_cast<void*>(&tensor_strides);
        result = ddiTableHandle->pfnSetArgumentValue2(graphHandle, index, &tensor_value);
    } else {
        result = ddiTableHandle->pfnSetArgumentValue(graphHandle, index, desc.data);
    }

    return result;
}

NPU_API(int32_t)
npu_level_zero_execute_graph(void** inputDescs, int32_t numInputs, void** outputDescs, int32_t numOutputs,
                             void* kernelName, void* kernel, int64_t kernelSize, void* context, void* device,
                             void* ddiTable, void** commandList) {
    auto inputs = reinterpret_cast<vpux::HostExec::MemRefDesc*>(inputDescs);
    auto outputs = reinterpret_cast<vpux::HostExec::MemRefDesc*>(outputDescs);

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (dumpKernelName) {
        const char* kernelNameStr = static_cast<const char*>(kernelName);
        std::cout << "Kernel name " << kernelNameStr << std::endl;
    }
#endif

    if (inputs == nullptr || outputs == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_POINTER, "Invalid nullpointer");
    }

    if (numInputs <= 0 || numOutputs <= 0) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_SIZE, "Invalid size");
    }

    graph_info graphInfo = (*graphMap)[kernel];
    if (graphInfo.graphHandle == nullptr) {
        // this is required until graph_init function is generated
        auto result = npu_level_zero_create_graph(kernel, kernelSize, context, device, ddiTable, nullptr, nullptr);

        ERROR_HANDLE(result, "Failed to compile a graph");

        graphInfo = (*graphMap)[kernel];
        if (graphInfo.graphHandle == nullptr) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid arguments");
        }
    }
    if (graphInfo.numArgs != (numInputs + numOutputs)) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid arguments");
    }
    auto* ddiTableHandle = static_cast<ze_graph_dditable_ext_t*>(ddiTable);

    const auto graphHandle = graphInfo.graphHandle;
    for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
        // Process inputs
        if (index < graphInfo.numInputArgs) {
            ERROR_HANDLE(set_arguments(index, inputs[index], graphHandle, ddiTableHandle),
                         "Failed to set input arguments");

        } else {
            ERROR_HANDLE(set_arguments(index, outputs[index - graphInfo.numInputArgs], graphHandle, ddiTableHandle),
                         "Failed to set output arguments");
        }
    }

    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute = ddiTableHandle->pfnAppendGraphExecute;
    auto commandListHandle =
            (commandList == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandList);

    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid nullpointer");
    }

    auto result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
        result = zeCommandListReset(commandListHandle);
        ERROR_HANDLE(result, "Failed to reset command list");
        result = pfnAppendGraphExecute(commandListHandle, graphHandle, nullptr, nullptr, 0, nullptr);
    }

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if ((result != ZE_RESULT_SUCCESS) && enabledFailedArgumentDesc) {
        for (uint32_t index = 0; index < graphInfo.numArgs; ++index) {
            vpux::HostExec::MemRefDesc desc;
            if (index < graphInfo.numInputArgs) {
                desc = inputs[index];

            } else {
                desc = outputs[index - graphInfo.numInputArgs];
            }
            std::cout << "Set argument index, " << index << " with data ptr, 0x" << std::hex
                      << reinterpret_cast<uint64_t>(desc.data) << std::dec << reinterpret_cast<uint64_t>(desc.data)
                      << ", byteSize, " << desc.elementByteSize << ", offset, " << desc.offset << ", byte offset, "
                      << desc.elementByteSize * desc.offset << std::endl;
            std::cout.flush();
        }
        std::cout.flush();
    }
#endif

    ERROR_HANDLE(result, "Failed to append graph execute");

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_submit_commandlist(void** commandLists, void* commandQueue, void* fence, void* event) {
    auto commandListHandle =
            (commandLists == nullptr) ? nullptr : *reinterpret_cast<ze_command_list_handle_t*>(commandLists);
    auto commandQueueHandle = static_cast<ze_command_queue_handle_t>(commandQueue);
    auto fenceHandle = static_cast<ze_fence_handle_t>(fence);
    auto eventHandle = static_cast<ze_event_handle_t>(event);
    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "Invalid nullpointer");
    }

    // note commnad queue is null when immediate command list is used
    if (commandQueueHandle != nullptr) {
        if (eventHandle != nullptr) {
            zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
            zeCommandListAppendSignalEvent(commandListHandle, eventHandle);
            zeCommandListClose(commandListHandle);
            auto result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, nullptr);
            ERROR_HANDLE(result, "Failed to zeCommandQueueExecuteCommandList");

        } else {
            if (fence == nullptr) {
                // add a barrier at the end of command list to ensure all commands are finished
                zeCommandListAppendBarrier(commandListHandle, nullptr, 0, nullptr);
            }
            zeCommandListClose(commandListHandle);
            auto result = zeCommandQueueExecuteCommandLists(commandQueueHandle, 1, &commandListHandle, fenceHandle);
            ERROR_HANDLE(result, "Failed to zeCommandQueueExecuteCommandList");
        }
    }

    RETURN_SUCCESS();
}

NPU_API(void)
npu_level_zero_get_last_error(char** pError) {
    *pError = lastErrorMessage;
}

NPU_API(int32_t)
npu_level_zero_reset_commandlist(void** commandLists) {
    if (commandLists == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
    }

    auto commandListHandle = *reinterpret_cast<ze_command_list_handle_t*>(commandLists);

    if (commandListHandle == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
    }

    auto result = zeCommandListReset(commandListHandle);
    ERROR_HANDLE(result, "Failed to reset a commandlist");
    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_reset_commandlists(void** commandLists, int32_t numCommandLists) {
    for (int32_t index = 0; index < numCommandLists; ++index) {
        auto commandListHandle = static_cast<ze_command_list_handle_t>(commandLists[index]);

        if (commandListHandle == nullptr) {
            ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
        }

        auto result = zeCommandListReset(commandListHandle);
        ERROR_HANDLE(result, "Failed to reset a commandlist");
    }

    RETURN_SUCCESS();
}

NPU_API(int32_t)
npu_level_zero_get_network_metadata(void* metadata, uint64_t metadataSize, void* networkMetadata, void* inputDescs,
                                    void* outputDescs) {
    if (metadata == nullptr || networkMetadata == nullptr || inputDescs == nullptr || outputDescs == nullptr) {
        ERROR_HANDLE(ZE_RESULT_ERROR_INVALID_ARGUMENT, "Invalid argument");
    }

    auto blob = reinterpret_cast<uint8_t*>(metadata);
    auto deserializedMetadata = elf::MetadataSerialization::deserialize(blob, metadataSize);

    auto input_descriptors = reinterpret_cast<std::vector<ArgumentDescriptor>*>(inputDescs);
    auto output_descriptors = reinterpret_cast<std::vector<ArgumentDescriptor>*>(outputDescs);

    vpux::NetworkMetadata* network = reinterpret_cast<vpux::NetworkMetadata*>(networkMetadata);

    *network = vpux::VPUMI37XX::getNetworkMetadata(reinterpret_cast<uint8_t*>(metadata), metadataSize);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: The blow coders are from L0 to initialize as L0 driver does not support IRGraph(LLVM w/ ELFs for subgraphs)
    // This will be remove when dynamic model compilation is supported by L0 (CID)
    static std::map<elf::DType, ze_graph_argument_precision_t> precisions = {
            {elf::DType::DType_NOT_SET, ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN},
            {elf::DType::DType_FP64, ZE_GRAPH_ARGUMENT_PRECISION_FP64},
            {elf::DType::DType_FP32, ZE_GRAPH_ARGUMENT_PRECISION_FP32},
            {elf::DType::DType_FP16, ZE_GRAPH_ARGUMENT_PRECISION_FP16},
            {elf::DType::DType_U64, ZE_GRAPH_ARGUMENT_PRECISION_UINT64},
            {elf::DType::DType_U32, ZE_GRAPH_ARGUMENT_PRECISION_UINT32},
            {elf::DType::DType_U16, ZE_GRAPH_ARGUMENT_PRECISION_UINT16},
            {elf::DType::DType_U8, ZE_GRAPH_ARGUMENT_PRECISION_UINT8},
            {elf::DType::DType_U4, ZE_GRAPH_ARGUMENT_PRECISION_UINT4},
            {elf::DType::DType_I64, ZE_GRAPH_ARGUMENT_PRECISION_INT64},
            {elf::DType::DType_I32, ZE_GRAPH_ARGUMENT_PRECISION_INT32},
            {elf::DType::DType_I16, ZE_GRAPH_ARGUMENT_PRECISION_INT16},
            {elf::DType::DType_I8, ZE_GRAPH_ARGUMENT_PRECISION_INT8},
            {elf::DType::DType_I4, ZE_GRAPH_ARGUMENT_PRECISION_INT4},
            {elf::DType::DType_BIN, ZE_GRAPH_ARGUMENT_PRECISION_BIN},
            {elf::DType::DType_I4X, ZE_GRAPH_ARGUMENT_PRECISION_NF4},
            {elf::DType::DType_BFP16, ZE_GRAPH_ARGUMENT_PRECISION_BF16}};

    static std::map<size_t, ze_graph_argument_layout_t> layouts = {
            {0x1, ZE_GRAPH_ARGUMENT_LAYOUT_C},         {0x12, ZE_GRAPH_ARGUMENT_LAYOUT_NC},
            {0x21, ZE_GRAPH_ARGUMENT_LAYOUT_CN},       {0x123, ZE_GRAPH_ARGUMENT_LAYOUT_CHW},
            {0x1234, ZE_GRAPH_ARGUMENT_LAYOUT_NCHW},   {0x1342, ZE_GRAPH_ARGUMENT_LAYOUT_NHWC},
            {0x12345, ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW}, {0x13452, ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC}};

    auto set_properties = [](ze_graph_argument_properties_3_t& properties, elf::TensorRef& tensor_desc,
                             elf::TensorRef& network_desc, elf::OVNode& node, vpux::IODescriptor& io_desc) {
        strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.name, tensor_desc.name);
        strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.debug_friendly_name, node.friendly_name);

        if (node.tensor_names_count == 0) {
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(node.tensor_names[node.tensor_names_count++], tensor_desc.name);
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(node.tensor_names[node.tensor_names_count++], node.friendly_name);
        }

        for (uint32_t i = 0; i < node.tensor_names_count; i++) {
            strcpy_s<ZE_MAX_GRAPH_ARGUMENT_NAME>(properties.associated_tensor_names[i], node.tensor_names[i]);

            io_desc.outputTensorNames.insert(node.tensor_names[i]);
        }
        properties.associated_tensor_names_count = node.tensor_names_count;

        memcpy_s(properties.dims, sizeof(properties.dims), tensor_desc.dimensions,
                 tensor_desc.dimensions_size * sizeof(uint32_t));

        properties.dims_count = tensor_desc.dimensions_size;
        properties.networkPrecision = precisions[network_desc.data_type];
        properties.devicePrecision = precisions[tensor_desc.data_type];
        properties.networkLayout = layouts[network_desc.order];
        properties.deviceLayout = layouts[tensor_desc.order];
    };

    auto set_property_strides = [](ze_graph_argument_property_strides_t& strides, elf::OVNode& node) {
        strides = {ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES, nullptr, false};
        const auto dynamicDim = std::numeric_limits<uint64_t>::max();
        for (uint32_t index = 0; index < node.shape_size; index++) {
            // store strides in reverse order
            if (node.shape[index] == dynamicDim) {
                strides.supportsDynamicStrides = true;
                break;
            }
        }
    };

    input_descriptors->resize(network->inputs.size());
    for (size_t index = 0; index < input_descriptors->size(); ++index) {
        auto& descriptor = input_descriptors->at(index);
        descriptor.idx = index;
        auto& properties = descriptor.info;
        properties.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;

        auto tensor = index < deserializedMetadata->mInTensorDescriptors.size()
                              ? deserializedMetadata->mInTensorDescriptors[index]
                              : elf::TensorRef{};
        auto net = index < deserializedMetadata->mNetInputs.size() ? deserializedMetadata->mNetInputs[index]
                                                                   : elf::TensorRef{};
        auto node = index < deserializedMetadata->mOVParameters.size() ? deserializedMetadata->mOVParameters[index]
                                                                       : elf::OVNode{};

        set_properties(properties, tensor, net, node, network->inputs.at(index));
        set_property_strides(descriptor.infoStrides, node);

        properties.quantReverseScale = 1.0f;
        properties.quantZeroPoint = 0;
    }

    output_descriptors->resize(network->outputs.size());
    const auto inputCount = network->inputs.size();
    for (size_t index = 0; index < output_descriptors->size(); ++index) {
        auto& descriptor = output_descriptors->at(index);
        descriptor.idx = index + inputCount;
        auto& properties = descriptor.info;
        properties.type = ZE_GRAPH_ARGUMENT_TYPE_OUTPUT;

        auto tensor = index < deserializedMetadata->mOutTensorDescriptors.size()
                              ? deserializedMetadata->mOutTensorDescriptors[index]
                              : elf::TensorRef{};
        auto net = index < deserializedMetadata->mNetOutputs.size() ? deserializedMetadata->mNetOutputs[index]
                                                                    : elf::TensorRef{};
        auto node = index < deserializedMetadata->mOVResults.size() ? deserializedMetadata->mOVResults[index]
                                                                    : elf::OVNode{};

        set_properties(properties, tensor, net, node, network->outputs.at(index));
        set_property_strides(descriptor.infoStrides, node);

        properties.quantReverseScale = 1.0f;
        properties.quantZeroPoint = 0;
    }

    RETURN_SUCCESS();
}

#endif
