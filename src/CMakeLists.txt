#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_NPU_MONO)
    add_definitions(-DOV_BUILD_POSTFIX=${OV_BUILD_POSTFIX_VAR})
endif()

add_subdirectory(vpux_driver_compiler)


add_subdirectory(vpux_utils)

add_subdirectory(vpux_compiler)

if(BUILD_SHARED_LIBS)
endif()

if(BUILD_SHARED_LIBS AND ENABLE_NPU_EXECUTION_ENGINE)
    add_subdirectory(mlir_runtime)
endif()
