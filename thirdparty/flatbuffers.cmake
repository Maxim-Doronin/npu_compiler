#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#


if(TARGET flatbuffers OR TARGET flatc)
    # we are building NPU plugin via -DOPENVINO_EXTRA_MODULES
    # and flatbuffers is already built as part of OpenVINO in case of
    # building in a single tree
    message(WARNING "Flatbuffers target present. Possible version mismatch.")
elseif(NOT OpenVINO_SOURCE_DIR)
    # Using OpenVINO prebuilt developer package
    set(OPENVINO_INSTALL_DEVELOPER_PACKAGE_DIR "${OpenVINODeveloperPackage_DIR}/..")
    message(STATUS "OpenVINO Developer Package Directory: ${OPENVINO_INSTALL_DEVELOPER_PACKAGE_DIR}")
    add_native_exec_target(flatc TOOL_DIR "${OPENVINO_INSTALL_DEVELOPER_PACKAGE_DIR}/bin")

    add_library(flatbuffers INTERFACE)
    target_link_libraries(flatbuffers INTERFACE openvino::flatbuffers)
    get_target_property(FLATBUFFERS_OV_PKG_INCLUDE_DIR openvino::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(flatbuffers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FLATBUFFERS_OV_PKG_INCLUDE_DIR}")
    set(flatbuffers_TARGET flatbuffers PARENT_SCOPE)
else()
    if (ANDROID)
        add_native_exec_target(flatc)
        set(FLATBUFFERS_BUILD_FLATC OFF)
    else()
        set(FLATBUFFERS_BUILD_FLATC ON)
    endif()

    set(FLATBUFFERS_BUILD_TESTS OFF)
    set(FLATBUFFERS_INSTALL OFF)
    # Use flatbuffers from OpenVINO third-party source code
    set(FLATBUFFERS_DIR "${OpenVINO_SOURCE_DIR}/thirdparty/flatbuffers/flatbuffers")
    add_subdirectory(${FLATBUFFERS_DIR} "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/flatbuffers" EXCLUDE_FROM_ALL)

    if(NOT MSVC AND FLATBUFFERS_BUILD_FLATC)
        target_compile_options(flatc PRIVATE -Wno-suggest-override)
    endif()
endif()

set(flatc_TARGET flatc)
set(flatc_COMMAND $<TARGET_FILE:flatc>)
set(flatc_TARGET "${flatc_TARGET}" PARENT_SCOPE)
set(flatc_COMMAND "${flatc_COMMAND}" PARENT_SCOPE)
