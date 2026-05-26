#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME npuFuncTests)
set(EXCLUDED_FUNC_TESTS_DIR "")
set(OPTIONAL_FUNC_TESTS_INCLUDES "")
set(OPTIONAL_FUNC_TESTS_LIBS "")

set(SKIP_CONFIG "npu_skip_func_tests.xml")
set(SKIP_CONFIG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/${SKIP_CONFIG})

if(ENABLE_NPU_EXECUTION_ENGINE AND BUILD_SHARED_LIBS)
    list(APPEND OPTIONAL_FUNC_TESTS_LIBS
        openvino::npu_zero_utils
        npu_compiler_interface
        openvino::npu_vm_runtime_api
    )
else()
    list(APPEND EXCLUDED_FUNC_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/behavior/vm_runtime"
    )
endif()

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_FUNC_TESTS_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests/functional/shared_tests_instances
        ${PROJECT_SOURCE_DIR}/tests/functional/behavior
        ${PROJECT_SOURCE_DIR}/tests/functional/subgraph_tests
        ${PROJECT_SOURCE_DIR}/tests/functional/custom/single_layer_tests/classes
        ${PROJECT_SOURCE_DIR}/artifacts/vpuip_2
        ${PROJECT_SOURCE_DIR}/src/vpux_translate_utils/include
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_test_classes/include
        ${LLVM_INCLUDE_DIRS}
    LINK_LIBRARIES
        ${OPTIONAL_FUNC_TESTS_LIBS}
        npu_ov_utils
        npu_logger
        openvino::func_test_utils
        openvino::funcSharedTests
        openvino::format_reader
        openvino::runtime
        openvino::npu_al
        openvino::npu_ops
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

ov_build_target_faster(${TARGET_NAME} PCH PUBLIC "precomp.hpp")

if(WIN32)
    if(ENABLE_DX12)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX12)
        target_link_libraries(${TARGET_NAME} PRIVATE d3d12 dxcore)
    endif()
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                      CXX_STANDARD 17)

if(MSVC)
    # Enforce standards conformance on MSVC
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /Zc:preprocessor
    )
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

set(VM_RUNTIME_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/behavior/vm_runtime/data")

# Copies test data to installed package
install(FILES ${SKIP_CONFIG_PATH}
        DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

install(DIRECTORY "${VM_RUNTIME_TEST_DATA_DIR}/"
        DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

get_target_property(EXE_LOCATION ${TARGET_NAME} RUNTIME_OUTPUT_DIRECTORY)

# Copies test data to local build directory
if(EXE_LOCATION)
    configure_file(
        ${SKIP_CONFIG_PATH}
        ${EXE_LOCATION}/${SKIP_CONFIG}
        COPYONLY
    )

    file(COPY "${VM_RUNTIME_TEST_DATA_DIR}/"
         DESTINATION "${EXE_LOCATION}"
    )
endif()

if(ENABLE_DEVELOPER_BUILD AND UNIX)
    # add npuFuncTest improved with gtest-parallel to ctest
    # we are using run_gtest_parallel_manual_threads.sh script to provide
    # PARALLEL_WORKERS env variable as -w workers command line argument
    add_test(NAME ${TARGET_NAME}-parallel
            COMMAND
                ${PROJECT_SOURCE_DIR}/scripts/run_gtest_parallel_manual_threads.sh
                ${PROJECT_SOURCE_DIR}/thirdparty/gtest-parallel/gtest-parallel
                $<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
            set_tests_properties(${TARGET_NAME}-parallel PROPERTIES
            LABELS "NPU;PlatformSpecific;3720;4000"
    )
endif()
