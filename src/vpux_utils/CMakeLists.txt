#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(VPUX_UTILS_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(src)

if(ENABLE_DEP_CHECK)
    # Append utils dependencies
    get_lib_targets_recursive(VPUX_UTILS_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})
    depcheck_collect_deps(src include "vpux/utils" ${VPUX_UTILS_TARGETS})
endif()
