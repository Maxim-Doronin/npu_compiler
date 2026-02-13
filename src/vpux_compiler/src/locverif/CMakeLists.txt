#
# Copyright (C) 2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_locverif")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_ov_utils
        npu_llvm_utils
        MLIRIR
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_net
)
