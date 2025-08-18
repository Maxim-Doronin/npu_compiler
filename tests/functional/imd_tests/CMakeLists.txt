#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(EXCLUDED_FUNC_TESTS_DIR "")
set(OPTIONAL_FUNC_TESTS_INCLUDES "")
set(OPTIONAL_FUNC_TESTS_LIBS "")

set(TARGET_NAME imdFuncTests)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    EXCLUDED_SOURCE_PATHS
    ${EXCLUDED_FUNC_TESTS_DIR}
    INCLUDES
    ${OPTIONAL_FUNC_TESTS_INCLUDES}
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/tests/functional/shared_tests_instances"
    "${PROJECT_SOURCE_DIR}/tests/functional/behavior"
    "${PROJECT_SOURCE_DIR}/tests/functional/subgraph_tests"
    "${PROJECT_SOURCE_DIR}/tests/functional/custom/single_layer_tests/classes"
    "${PROJECT_SOURCE_DIR}/artifacts/vpuip_2"
    "${PROJECT_SOURCE_DIR}/src/vpux_translate_utils/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../internal"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shared_test_classes/include"
    ${LLVM_INCLUDE_DIRS}
    LINK_LIBRARIES
    ${OPTIONAL_FUNC_TESTS_LIBS}
    openvino::func_test_utils
    openvino::funcSharedTests
    openvino::format_reader
    openvino::runtime
    npu_llvm_utils
    npu_ov_utils
    openvino::npu_al
    DEFINES
    DATA_PATH=\"${DATA_PATH}\"
    MODELS_PATH=\"${MODELS_PATH}\"
    LABELS KMB
)

set(IMD_TEST_CONSTANTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/test_constants.cpp")
get_target_property(${TARGET_NAME}_SOURCES ${TARGET_NAME} SOURCES)
list(INSERT ${TARGET_NAME}_SOURCES 0 ${IMD_TEST_CONSTANTS_FILE})

add_compile_definitions(ENABLE_IMD_PLUGIN=1)

set_target_properties(
    ${TARGET_NAME}
    PROPERTIES
        FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
        CXX_STANDARD 17
        SOURCES "${${TARGET_NAME}_SOURCES}"
)

if(ENABLE_MLIR_COMPILER)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MLIR_COMPILER)
endif()

install(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests COMPONENT ${VPUX_TESTS_COMPONENT} EXCLUDE_FROM_ALL
)

if(ENABLE_DEVELOPER_BUILD AND UNIX)
    # add imdFuncTest improved with gtest-parallel to ctest
    # we are using run_gtest_parallel_manual_threads.sh script to provide
    # PARALLEL_WORKERS env variable as -w workers command line argument
    add_test(
        NAME ${TARGET_NAME}-parallel
        COMMAND
            ${PROJECT_SOURCE_DIR}/scripts/run_gtest_parallel_manual_threads.sh
            ${PROJECT_SOURCE_DIR}/thirdparty/gtest-parallel/gtest-parallel
            $<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    )
    set_tests_properties(
        ${TARGET_NAME}-parallel
        PROPERTIES LABELS "NPU;PlatformSpecific;3720;4000"
    )
endif()
