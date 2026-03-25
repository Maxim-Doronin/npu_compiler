#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_pipelines")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_core_utils
        npu_llvm_utils
        # TODO: see discussion 16934
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_hostexec_transforms
        npu_compiler_locverif
    PUBLIC
        mlir-dependencies
    SYSTEM_INCLUDES
        $<TARGET_PROPERTY:openvino::npu_al,INTERFACE_INCLUDE_DIRECTORIES>
    )
