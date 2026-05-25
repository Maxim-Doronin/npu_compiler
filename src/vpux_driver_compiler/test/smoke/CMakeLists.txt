#
# Copyright (C) 2025-2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# compilerTest as smoke test
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(compilerTest compilerTest.cpp)
target_link_libraries(compilerTest
    PRIVATE
        ${COMPILER_LOADER_DLL} # VCL function
        npu_logger
        openvino::npu_logger_utils # logging utils
        openvino::runtime
        openvino::core::dev)
ov_add_api_validator_post_build_step(TARGET compilerTest)

# profilingTest to check profiling API
add_executable(profilingTest profilingTest.c)
target_link_libraries(profilingTest
    PRIVATE
        ${COMPILER_LOADER_DLL})
ov_add_api_validator_post_build_step(TARGET profilingTest)

# loaderTest to check npu_elf
add_executable(loaderTest loaderTest.cpp)
target_link_libraries(loaderTest
    PRIVATE
        npu_elf
        npu_core_utils
        LLVMSupport)
ov_add_api_validator_post_build_step(TARGET loaderTest)

# Install functional test to CiD component
if(BUILD_COMPILER_FOR_DRIVER)
    install_target_with_debug_files(compilerTest ${INSTALL_DESTINATION} ${INSTALL_COMPONENT})
    install_target_with_debug_files(profilingTest ${INSTALL_DESTINATION} ${INSTALL_COMPONENT})
    install_target_with_debug_files(loaderTest ${INSTALL_DESTINATION} ${INSTALL_COMPONENT})
endif()
