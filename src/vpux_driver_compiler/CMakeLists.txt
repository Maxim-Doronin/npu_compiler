#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(cmake/install_target_with_debug_files.cmake)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/sdl)
    if(ENABLE_PRODUCTION_BUILD)
        add_link_options("/INTEGRITYCHECK")
    endif()
endif()

set(CID_COMPONENT "CiD")
ov_cpack_add_component(${CID_COMPONENT})
set(CIP_COMPONENT "CiP")
ov_cpack_add_component(${CIP_COMPONENT})

if(BUILD_COMPILER_FOR_DRIVER)
    set(INSTALL_COMPONENT ${CID_COMPONENT})
    set(INSTALL_DESTINATION cid)
else()
    set(INSTALL_COMPONENT ${CIP_COMPONENT})
    set(INSTALL_DESTINATION cip)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)

install(
    FILES
        "${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/CHANGES.txt"
    DESTINATION ${INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT})

install(
    FILES
        "${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/README.md"
    DESTINATION ${INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT})
