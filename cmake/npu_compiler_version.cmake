#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Function to configure NPU Compiler version properties
function(npu_compiler_configure_version TARGET_NAME)
    ov_commit_hash(PLUGIN_GIT_COMMIT_HASH ${CMAKE_CURRENT_SOURCE_DIR})

    string(TIMESTAMP DATE_MAJOR "%m%d")
    string(TIMESTAMP DATE_MINOR "%H%M%S")
    string(TIMESTAMP CURRENT_YEAR "%Y")

    if(BUILD_COMPILER_FOR_DRIVER)
        # The first section is WDDM version, it aligns with operation system change
        set(VERSION_WDDM 32)
        # The second section is always 0 now
        set(VERSION_UNUSED 0)
        # The last two sections use date to identify build number
        set(VERSION_BUILD_MAJOR ${DATE_MAJOR})
        set(VERSION_BUILD_MINOR ${DATE_MINOR})

        set(COPYRIGHT_STR "Copyright (C) ${CURRENT_YEAR} Intel Corporation")

        set(PRODUCTNAME_BASE "NPU Compiler")
        set(FILEDESCRIPTION_BASE "Intel NPU Compiler")

        set(OV_VS_VER_FILEVERSION_QUAD "${VERSION_WDDM},${VERSION_UNUSED},${VERSION_BUILD_MAJOR},${VERSION_BUILD_MINOR}")
        set(OV_VS_VER_PRODUCTVERSION_QUAD "${VERSION_WDDM},${VERSION_UNUSED},${VERSION_BUILD_MAJOR},${VERSION_BUILD_MINOR}")
        set(OV_VS_VER_FILEVERSION_STR "${VERSION_WDDM}.${VERSION_UNUSED}.${VERSION_BUILD_MAJOR}.${VERSION_BUILD_MINOR}")
        set(OV_VS_VER_PRODUCTVERSION_STR "${VERSION_WDDM}.${VERSION_UNUSED}.${VERSION_BUILD_MAJOR}.${VERSION_BUILD_MINOR}")
    else()
        # On master branch or HEAD, PLUGIN_GIT_BRANCH_POSTFIX is empty, otherwise it is "-branch_name"
        ov_branch_name(PLUGIN_GIT_BRANCH ${CMAKE_CURRENT_SOURCE_DIR})
        if(NOT PLUGIN_GIT_BRANCH MATCHES "^(master|HEAD)$")
            set(PLUGIN_GIT_BRANCH_POSTFIX "-${PLUGIN_GIT_BRANCH}")
        endif()

        set(COPYRIGHT_STR "Copyright (C) 2023-${CURRENT_YEAR}, Intel Corporation")
        set(PRODUCTNAME_BASE "OpenVINO toolkit")
        set(FILEDESCRIPTION_BASE "OpenVINO NPU Compiler")

        set(OV_VS_VER_FILEVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
        set(OV_VS_VER_PRODUCTVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
        set(OV_VS_VER_FILEVERSION_STR "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}.${OpenVINO_VERSION_BUILD}")
        set(OV_VS_VER_PRODUCTVERSION_STR "${CI_BUILD_NUMBER}-${PLUGIN_GIT_COMMIT_HASH}${PLUGIN_GIT_BRANCH_POSTFIX}")
    endif()

    if(NOT ENABLE_DEVELOPER_BUILD)
        set(OV_VS_VER_PRODUCTNAME_STR "${PRODUCTNAME_BASE}")
        set(OV_VS_VER_FILEDESCRIPTION_STR "${FILEDESCRIPTION_BASE}")
    else()
        set(OV_VS_VER_PRODUCTNAME_STR "${PRODUCTNAME_BASE} DEV")
        set(OV_VS_VER_FILEDESCRIPTION_STR "${FILEDESCRIPTION_BASE} DEV")
    endif()

    set(OV_VS_VER_COMPANY_NAME_STR "Intel Corporation")
    set(OV_VS_VER_COPYRIGHT_STR "${COPYRIGHT_STR}")
    set(OV_VS_VER_ORIGINALFILENAME_STR "${CMAKE_SHARED_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(OV_VS_VER_INTERNALNAME_STR ${TARGET_NAME})
    
    set(vs_version_output "${CMAKE_CURRENT_BINARY_DIR}/vs_version.rc")
    configure_file("${IEDevScripts_DIR}/vs_version/vs_version.rc.in" "${vs_version_output}" @ONLY)
    source_group("src" FILES ${vs_version_output})
    target_sources(${TARGET_NAME} PRIVATE ${vs_version_output})
endfunction()
