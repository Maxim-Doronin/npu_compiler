#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(NPU_MV_TOOLS_DIR "${VPUX_SOURCE_DIR}/bin/MoviTools")

function(get_mv_tools_url output)
    file(READ "${VPUX_SOURCE_DIR}/artifacts/vpuip_2/revisions.json" json_string)
    string(JSON json_common GET ${json_string} "common")
    string(JSON json_runtime_kernels GET ${json_common} "runtime_kernels")
    string(JSON json_movitools GET ${json_runtime_kernels} "movitools")
    string(JSON json_artifactory_url GET ${json_runtime_kernels} "artifactory_url")
    string(REPLACE "{movitools_version}" ${json_movitools} movitools_url ${json_artifactory_url})
    set(${output} ${movitools_url} PARENT_SCOPE)
endfunction()

function(get_mv_tools_path output)
    if(DEFINED ENV{IE_NPU_FORCE_MV_TOOLS_PATH})
        set(${output} $ENV{IE_NPU_FORCE_MV_TOOLS_PATH} PARENT_SCOPE)
    else()
        get_mv_tools_version(mv_tools_version)
        file(MAKE_DIRECTORY "${NPU_MV_TOOLS_DIR}")
        set(${output} "${NPU_MV_TOOLS_DIR}/${mv_tools_version}" PARENT_SCOPE)
    endif()
endfunction()

function(get_mv_tools_version output)
    # get the last folder name from url, which is also the tools version
    get_mv_tools_url(mv_tools_url)
    get_filename_component(mv_tools_directory ${mv_tools_url} DIRECTORY)
    get_filename_component(mv_tools_version ${mv_tools_directory} NAME)
    set(${output} ${mv_tools_version} PARENT_SCOPE)
endfunction()

function(exists_mv_tools_version exists)
    if(DEFINED ENV{IE_NPU_FORCE_MV_TOOLS_PATH})
        message(WARNING "You are using forced MoviTools version which is recommended for debugging only.")
        set(${exists} TRUE PARENT_SCOPE)
        return()
    endif()

    get_mv_tools_path(mv_tools_path)

    if(EXISTS "${mv_tools_path}")
        set(${exists} TRUE PARENT_SCOPE)
    else()
        set(${exists} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(remove_old_mv_tools mv_tools_dir limit)
  file(GLOB children RELATIVE ${mv_tools_dir} "${mv_tools_dir}/*")
  list(LENGTH children num_dirs)

  math(EXPR num_to_remove "${num_dirs} - ${limit}")
  if(num_to_remove LESS_EQUAL 0)
    return()
  endif()

  set(dirs_with_date)
  foreach(child ${children})
    file(TIMESTAMP ${mv_tools_dir}/${child} mtime "%Y%m%d%H%M%S")
    list(APPEND dirs_with_date "${mtime},${mv_tools_dir}/${child}")
  endforeach()

  message(${dirs_with_date})

  list(SORT dirs_with_date)

  list(SUBLIST dirs_with_date 0 ${num_to_remove} dirs_to_remove)
  foreach(dir_with_date ${dirs_to_remove})
    string(REGEX REPLACE "^[0-9]*," "" folder ${dir_with_date})
    message("Removing old tools: ${folder}")
    file(REMOVE_RECURSE "${folder}")
  endforeach()
endfunction()

function(get_mv_tools)
    get_mv_tools_url(mv_tools_url)
    get_mv_tools_path(mv_tools_path)
    get_mv_tools_version(mv_tools_version)

    get_filename_component(tools_archive_name ${mv_tools_url} NAME)
    set(temp_dir "${CMAKE_BINARY_DIR}/MoviTools-temporary")

    remove_old_mv_tools("${NPU_MV_TOOLS_DIR}" 2)

    message("Downloading MoviTools to location ${temp_dir}/${tools_archive_name}")
    file(DOWNLOAD ${mv_tools_url} "${temp_dir}/${tools_archive_name}" SHOW_PROGRESS STATUS download_status)
    list(GET download_status 0 download_status_code)
    if(download_status_code)
        list(GET download_status 1 download_status_message)
        message(SEND_ERROR "MoviTools download failed with the error: ${download_status_message}")
    endif()

    file(ARCHIVE_EXTRACT INPUT "${temp_dir}/${tools_archive_name}" DESTINATION ${NPU_MV_TOOLS_DIR})
    file(TOUCH_NOCREATE ${mv_tools_path})
    message("MoviTools extracted to ${mv_tools_path}")

    file(REMOVE_RECURSE ${temp_dir})
endfunction()
