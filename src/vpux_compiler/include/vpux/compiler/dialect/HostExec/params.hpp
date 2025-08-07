//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux::HostExec {

enum HostMainFuncArgs {
    HOST_MAIN_FUNC_ARGS_CONTEXT,
    HOST_MAIN_FUNC_ARGS_DEVICE,
    HOST_MAIN_FUNC_ARGS_DDI_TABLE,
    HOST_MAIN_FUNC_ARGS_COMMAND_LIST,
    HOST_MAIN_FUNC_ARGS_COMMAND_QUEUE,
    HOST_MAIN_FUNC_ARGS_FENCE,
    HOST_MAIN_FUNC_ARGS_EVENT,
    HOST_MAIN_FUNC_ARGS_COUNT
};

#define GET_ARG_INDEX_CONTEXT(numFuncArgs) ((numFuncArgs) - 7)
#define GET_ARG_INDEX_DEVICE(numFuncArgs) ((numFuncArgs) - 6)
#define GET_ARG_INDEX_DDI_TABLE(numFuncArgs) ((numFuncArgs) - 5)
#define GET_ARG_INDEX_COMMAND_LIST(numFuncArgs) ((numFuncArgs) - 4)
#define GET_ARG_INDEX_COMMAND_QUEUE(numFuncArgs) ((numFuncArgs) - 3)
#define GET_ARG_INDEX_COMMAND_FENCE(numFuncArgs) ((numFuncArgs) - 2)
#define GET_ARG_INDEX_COMMAND_EVENT(numFuncArgs) ((numFuncArgs) - 1)

}  // namespace vpux::HostExec
