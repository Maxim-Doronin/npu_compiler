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
    PUBLIC
        ${LINK_COMPILER_LIBRARY} # VCL function
        npu_ov_utils # OV core function
        npu_llvm_utils # CL command line
        openvino::npu_logger_utils # logging utils
)
ov_add_api_validator_post_build_step(TARGET compilerTest)

# profilingTest to check profiling API
add_executable(profilingTest profilingTest.c)
target_link_libraries(profilingTest PUBLIC ${LINK_COMPILER_LIBRARY})
ov_add_api_validator_post_build_step(TARGET profilingTest)

# loaderTest to check npu_elf
add_executable(loaderTest loaderTest.cpp)
target_link_libraries(loaderTest PUBLIC npu_elf npu_llvm_utils npu_ov_utils LLVMSupport)
ov_add_api_validator_post_build_step(TARGET loaderTest)

# Install smoke tests to component
install_target_with_debug_files(compilerTest ${INSTALL_DESTINATION} ${INSTALL_COMPONENT})
install_target_with_debug_files(profilingTest ${INSTALL_DESTINATION} ${INSTALL_COMPONENT})
install_target_with_debug_files(loaderTest ${INSTALL_DESTINATION} ${INSTALL_COMPONENT})
