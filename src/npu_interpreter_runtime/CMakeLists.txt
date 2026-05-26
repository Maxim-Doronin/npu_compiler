#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_interpreter_runtime")

set(SOURCES
    src/npu_interpreter_runtime.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_vm_runtime_api
        openvino::ze_loader
        openvino::level-zero-ext
        openvino::npu_logger_utils
        openvino::runtime
        npu_bytecode_utils
)

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PRIVATE
        $<TARGET_PROPERTY:openvino::npu_vm_runtime_api,INTERFACE_INCLUDE_DIRECTORIES>
)

install(TARGETS ${TARGET_NAME}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT})
