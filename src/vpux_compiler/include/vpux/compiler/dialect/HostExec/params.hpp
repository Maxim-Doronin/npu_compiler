//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include "llvm/Support/Format.h"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux::HostExec {

constexpr bool defaultEnablePipelinedCmdListRecording = true;

enum HostMainFuncArgs {
    HOST_MAIN_FUNC_ARGS_CONTEXT,
    HOST_MAIN_FUNC_ARGS_DEVICE,
    HOST_MAIN_FUNC_ARGS_DDI_TABLE,
    HOST_MAIN_FUNC_ARGS_COMMAND_LIST,
    HOST_MAIN_FUNC_ARGS_COMMAND_LIST_COUNT,
    HOST_MAIN_FUNC_ARGS_COMMAND_QUEUE,
    HOST_MAIN_FUNC_ARGS_FENCE,
    HOST_MAIN_FUNC_ARGS_EVENT,
    HOST_MAIN_FUNC_ARGS_EXECUTION_CONTEXT,
    HOST_MAIN_FUNC_ARGS_COUNT
};

#define GET_ARG_INDEX_CONTEXT(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_CONTEXT)
#define GET_ARG_INDEX_DEVICE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_DEVICE)
#define GET_ARG_INDEX_DDI_TABLE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_DDI_TABLE)
#define GET_ARG_INDEX_COMMAND_LIST(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_COMMAND_LIST)
#define GET_ARG_INDEX_COMMAND_LIST_COUNT(numFuncArgs) \
    ((numFuncArgs) -                                  \
     (vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COMMAND_LIST_COUNT))
#define GET_ARG_INDEX_COMMAND_QUEUE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_COMMAND_QUEUE)
#define GET_ARG_INDEX_COMMAND_FENCE(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_FENCE)
#define GET_ARG_INDEX_COMMAND_EVENT(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_EVENT)
#define GET_ARG_INDEX_COMMAND_EXECUTION_CONTEXT(numFuncArgs) \
    ((numFuncArgs) - vpux::HostExec::HOST_MAIN_FUNC_ARGS_COUNT + vpux::HostExec::HOST_MAIN_FUNC_ARGS_EXECUTION_CONTEXT)

inline constexpr std::string_view HOST_EXEC_NETWORK_METADATA_NAME = "HostExec.networkMetadata";
inline constexpr std::string_view HOST_EXEC_NUM_SUBGRAPH_ATTR_NAME = "HostExec.numSubgraphs";

// NOTE: keep in sync with dynamicStridesAttrName in vpux_compiler/core/dialect/core
//       when dynamicstrides is supported in core dialect, this definition should be removed
inline constexpr std::string_view HOST_EXEC_DYNAMIC_STRIDES_ATTR_NAME = "dynamicStrides";
inline constexpr std::string_view HOST_EXEC_FUNC_ARG_DYNAMIC_STRIDES_ATTR_NAME = "func.dynamicStrides";

constexpr uint64_t MaxStrideDim = 5;

/**
 * @brief LLVM struct for dynamic stride support
 * @note Update getTensorDescStructType if this struct is changed
 */
enum class MemRefDescMemberIndex {
    DATA,
    OFFSET,
    ELEMENT_BYTE_SIZE,
    DIM_COUNT,
    NETWORK_ARG_INDEX,
    SIZES,
    STRIDES,

    COUNT
};

struct MemRefDesc {
    void* data;
    uint64_t offset;
    uint64_t elementByteSize;
    uint64_t dimCount;
    uint64_t networkArgIndex;
    uint64_t sizes[MaxStrideDim];
    uint64_t strides[MaxStrideDim];
};

inline constexpr std::string_view OVERRIDE_ENABLE_PIPELINED_COMMANDLIST_RECORDING =
        "OVERRIDE_ENABLE_PIPELINED_COMMANDLIST_RECORDING";
inline constexpr std::string_view ENABLE_HOSTCOMPILE_PRINT_KERNEL_NAME = "ENABLE_PRINT_HOSTCOMPILE_KERNEL_NAME";

}  // namespace vpux::HostExec

namespace llvm {
template <>
struct format_provider<vpux::HostExec::MemRefDesc> final {
    static void format(const vpux::HostExec::MemRefDesc& desc, raw_ostream& os, StringRef) {
        os << "data:" << llvm::format_hex(reinterpret_cast<uint64_t>(desc.data), 10, true);
        os << ", offset: " << desc.offset;
        os << ", elementByteSize: " << desc.elementByteSize;
        os << ", dimCount: " << desc.dimCount;
        os << ", networkArgIndex: " << desc.networkArgIndex;
        os << ", sizes: [";
        auto dimCount = std::min(desc.dimCount, vpux::HostExec::MaxStrideDim);
        for (uint64_t i = 0; i < dimCount; ++i) {
            os << desc.sizes[i];
            if (i != dimCount - 1) {
                os << ", ";
            }
        }
        os << "]";
        os << ", strides: [";
        for (uint64_t i = 0; i < dimCount; ++i) {
            os << desc.strides[i];
            if (i != dimCount - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
};
}  // namespace llvm
