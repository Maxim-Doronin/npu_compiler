#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.20)
project(clang_tidy_plugin)

find_package(Clang REQUIRED CONFIG)

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME} PRIVATE module.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${CLANG_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE clangTidyUtils)
