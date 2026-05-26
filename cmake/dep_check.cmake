#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Find static/object library targets in the provided list of targets
# The remaining arguments are the targets to filter
function(get_library_targets LIB_TARGETS)
    set(only_libs "")
    foreach(target IN LISTS ARGN)
        if(NOT TARGET ${target})
            message(FATAL_ERROR "Target '${target}' does not exist")
        endif()
        get_target_property(target_type ${target} TYPE)
        if(target_type MATCHES "STATIC_LIBRARY|OBJECT_LIBRARY")
            list(APPEND only_libs ${target})
        else()
            message(WARNING "Skipping `${target}` as it is not a library target")
        endif()
    endforeach()
    set(${LIB_TARGETS} ${only_libs} PARENT_SCOPE)
endfunction()

# Setup the dependency checker and wipe CSV file
function(depcheck_setup)
    set(DEP_CHECK_FILE "${CMAKE_BINARY_DIR}/dep_check.csv")
    set_property(GLOBAL PROPERTY DEP_CHECK_FILE "${DEP_CHECK_FILE}")
    # Clear the dependency file
    file(WRITE ${DEP_CHECK_FILE} "target, prefix, target_path, src_dir, inc_dir, pub_deps, all_deps, target_type\n") # Ensure the file is clean before appending dependencies
endfunction()

# Append dependency information for the provided list of targets to a CSV file,
# in the following format
#
#   target, prefix, target_path, src_dir, inc_dir, pub_deps, all_deps
#
# The remaining arguments after the first four are the targets
function(depcheck_collect_deps BASE_SRC_DIR BASE_INC_DIR PREFIX)
    message(STATUS "Collecting dependencies for ${PREFIX}")
    get_property(DEP_CHECK_FILE GLOBAL PROPERTY DEP_CHECK_FILE)

    if(NOT IS_ABSOLUTE ${BASE_SRC_DIR})
        set(BASE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${BASE_SRC_DIR}")
    endif()
    if(NOT IS_ABSOLUTE ${BASE_INC_DIR})
        set(BASE_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${BASE_INC_DIR}")
    endif()

    # Use project relative paths
    file(RELATIVE_PATH src_dir ${CMAKE_SOURCE_DIR} "${BASE_SRC_DIR}")
    file(RELATIVE_PATH inc_dir ${CMAKE_SOURCE_DIR} "${BASE_INC_DIR}")

    foreach(target IN LISTS ARGN)
        # Unwrap $<BUILD_INTERFACE:target>
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" target ${target})
        if(NOT TARGET ${target})
            message(FATAL_ERROR "Target '${target}' does not exist.")
        endif()

        get_target_property(target_src_dir ${target} SOURCE_DIR)
        file(RELATIVE_PATH target_path ${BASE_SRC_DIR} "${target_src_dir}")
        if(target_path MATCHES "^\\.\\./" OR target_path STREQUAL "..")
            message(FATAL_ERROR "Target '${target}' is outside of the base source directory '${src_dir}'.")
        endif()

        get_target_property(target_type ${target} TYPE)
        get_target_property(all_deps ${target} LINK_LIBRARIES)
        get_target_property(pub_deps ${target} INTERFACE_LINK_LIBRARIES)
        if (NOT all_deps)
            set(all_deps "")
        endif()
        if(NOT pub_deps)
            set(pub_deps "")
        endif()
        # Strip transitive dependencies from the interface
        list(FILTER pub_deps EXCLUDE REGEX "\\$<LINK_ONLY:.*>")

        # Strip INTERFACE targets if any
        # Disabled for speed
        #get_library_targets(pub_deps ${pub_deps})
        #get_library_targets(all_deps ${all_deps})

        set(dep_line "${target}, ${PREFIX}, ${target_path}, ${src_dir}, ${inc_dir}, ${pub_deps}, ${all_deps}, ${target_type}\n")
        file(APPEND "${DEP_CHECK_FILE}" "${dep_line}")
    endforeach()
endfunction()

# Add dep_check target to run the checker during build
function(add_depcheck_target target)
    get_property(DEP_CHECK_FILE GLOBAL PROPERTY DEP_CHECK_FILE)
    find_package(Python3 QUIET)
    if (NOT Python3_FOUND)
        message(WARNING "Python3 not found, dependency check will be skipped")
        return()
    endif()
    add_custom_target(${target} ALL
        COMMAND
            ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/cmake/dep_check.py" "${CMAKE_SOURCE_DIR}" "${DEP_CHECK_FILE}"
        COMMENT "Checking dependencies..."
        VERBATIM
    )
endfunction()

function(get_lib_targets_recursive targets dir)
    get_directory_property(local_targets DIRECTORY "${dir}" BUILDSYSTEM_TARGETS)
    get_library_targets(local_targets ${local_targets})
    set(all_targets ${local_targets})

    get_directory_property(subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
        get_lib_targets_recursive(subdir_targets "${subdir}")
        get_library_targets(subdir_targets ${subdir_targets})
        list(APPEND all_targets ${subdir_targets})
    endforeach()

    set(${targets} ${all_targets} PARENT_SCOPE)
endfunction()
