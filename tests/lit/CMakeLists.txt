#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

#

vpux_setup_lit_tool()

add_custom_target(copy_lit_tests_script ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy
            "${PROJECT_SOURCE_DIR}/scripts/run_all_lit_tests.sh"
            "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/run_all_lit_tests.sh"
        COMMENT "[LIT] Copy runner script"
)

set(data_path_npu "path('NPU/data')")

vpux_setup_lit_tests(NPU
    PATTERNS "*.mlir" "*.json"
    EXTRA_SOURCES "lit.local.cfg" "*.bin" "*.txt" "*.xml"
    VARS "data_path_npu"
    PARAMS "arch"
    PARAMS_DEFAULT_VALUES "NPU37XX"
    SUBSTITUTIONS "data_path_npu"
)

set(_lit_tests_dir_full_path "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests")

install(PROGRAMS "${_lit_tests_dir_full_path}"
        DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
