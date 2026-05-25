#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME npu_bytecode_utils)

list(APPEND SOURCES
    bytecode_reader.cpp
    magic_number.cpp
    print_utils.cpp
    section_header_table.cpp
    version.cpp
    virtual_machine/utils/buffer.cpp
    virtual_machine/virtual_machine.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})
# Prevents warning about unused parameters in NPU plugin logger header:
#   include/openvino/core/attribute_adapter.hpp:38:44: error: unused parameter 'x' [-Werror=unused-parameter]
if(NOT MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wno-error=unused-parameter
    )
endif()

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_logger_utils
        openvino::runtime
)
