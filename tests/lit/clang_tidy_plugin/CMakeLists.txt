#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# implementation is inspired by lit_tests.cmake
# exceptions are: no install commands

# require clang-tidy to be available on the system
# as check_clang_tidy.py calls "clang-tidy" internally
find_program(CHECK_CLANG_TIDY_EXISTS clang-tidy REQUIRED)

# use TARGET_FILE_DIR property of vpux-opt instead of CMAKE_BINARY_DIR
# CMAKE_BINARY_DIR is binary dir of compiler repo, not OpenVINO (where all
# binaries are eventually placed)
# note: TESTS_DIR contains a string with not-yet evaluated generator-expression
# can't store generator-expression to a variable as its value is available
# post-configuration
# note: vpux-opt is not required to exist at this point
# it's intentionally placed to lit-tests/NPU to be along all other LIT tests
# and to be run as a part of run_all_lit_tests.sh
set(TESTS_DIR "$<TARGET_FILE_DIR:vpux-opt>/lit-tests/NPU/clang_tidy_plugin")

# even though clang-tidy plugin depends on clang-tidy pre-installed on the system
# instead of built from sources of project's LLVM submodule
# LLVM package doesn't seem to contain check_clang_tidy.py script as it's a test helper
# hence take it from the source tree of LLVM submodule
set(CHECK_CLANG_TIDY_SOURCE_PATH "${CMAKE_SOURCE_DIR}/thirdparty/llvm-project/clang-tools-extra/test/clang-tidy/check_clang_tidy.py")
set(CHECK_CLANG_TIDY_BINARY_PATH "${TESTS_DIR}/check_clang_tidy.py")

get_clang_tidy_plugin_path(CLANG_TIDY_PLUGIN_PATH)

# to be copied later to TESTS_DIR
# will extend/override LIT configuration provided for NPU LIT tests
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py"
    @ONLY
)

add_custom_target(copy_check_clang_tidy ALL
    # remove pre-existing directory to avoid clash with previous usages
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${TESTS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TESTS_DIR}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CHECK_CLANG_TIDY_SOURCE_PATH} ${CHECK_CLANG_TIDY_BINARY_PATH}
    # lit.cfg.py requires check_clang_tidy.py to be available in the same directory
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py" ${TESTS_DIR}
)

# copy_check_clang_tidy depends on copy_NPU_tests
# to create $<TARGET_FILE_DIR:vpux-opt>/lit-tests/NPU directory
# note: copy_NPU_tests is created by `vpux_setup_lit_tests` call
add_dependencies(copy_check_clang_tidy copy_NPU_tests)

add_custom_target(copy_clang_tidy_plugin_tests ALL
    # assume TESTS_DIR has been re-created by copy_check_clang_tidy
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/tests" "${TESTS_DIR}/tests"
)

add_dependencies(copy_clang_tidy_plugin_tests copy_check_clang_tidy)
