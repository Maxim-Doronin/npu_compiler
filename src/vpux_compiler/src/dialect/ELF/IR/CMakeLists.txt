#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_elf
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_config
        npu_core_utils
        npu_llvm_utils
        npu_ov_utils
        npu_elf
        openvino::npu_al
        # E#98178: remove dependency on firmware headers
        npu_40xx_firmware_headers
        MLIRFuncDialect
    )
