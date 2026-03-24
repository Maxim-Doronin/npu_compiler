#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_core
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        # npu_compiler_dialect_core should not depend on other dialects (IE, VPU, VPUIP...),
        # to avoid circular dependencies.
        npu_compiler_utils
        npu_compiler_act_kernels
        npu_llvm_utils
        npu_ov_utils
        npu_elf
        MLIRTransformUtils)
