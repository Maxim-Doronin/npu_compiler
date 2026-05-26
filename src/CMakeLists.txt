#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_NPU_MONO)
    add_definitions(-DOV_BUILD_POSTFIX=${OV_BUILD_POSTFIX_VAR})
endif()

# Enable thin archives on Linux to reduce the size of static libraries and speed up linking.
# Use directory scope variable to apply to all targets in src/
# Guard with a capability check: Ubuntu 20 ships ar (binutils 2.34) which does not
# recognise --thin as a long option; llvm-ar and newer binutils do.
function(try_enable_thin_archives)
    execute_process(
        COMMAND ${CMAKE_AR} --help
        OUTPUT_VARIABLE ar_help
        ERROR_VARIABLE ar_help
    )
    if(ar_help MATCHES "--thin")
        string(APPEND CMAKE_STATIC_LINKER_FLAGS " --thin")
        set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS}" PARENT_SCOPE)
    endif()
endfunction()

if (UNIX)
    try_enable_thin_archives()
endif()

# Setup dependency checker
if(ENABLE_DEP_CHECK)
    include(${CMAKE_SOURCE_DIR}/cmake/dep_check.cmake)
    depcheck_setup()
endif()

add_subdirectory(vpux_driver_compiler)


add_subdirectory(vpux_utils)

add_subdirectory(vpux_compiler)

if(BUILD_SHARED_LIBS)
endif()

if(BUILD_SHARED_LIBS AND ENABLE_NPU_EXECUTION_ENGINE)
    add_subdirectory(mlir_runtime)
    add_subdirectory(npu_interpreter_runtime)
endif()
