#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
  set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /LTCG")
endif()

if(BUILD_COMPILER_FOR_DRIVER)
    set(LINK_COMPILER_LIBRARY npu_driver_compiler)
else()
    set(LINK_COMPILER_LIBRARY openvino_intel_npu_compiler)
endif()

add_subdirectory(functional)
add_subdirectory(smoke)
