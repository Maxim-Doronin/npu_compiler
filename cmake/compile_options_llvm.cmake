#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(set_llvm_flags)
    set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
    set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
    if(NOT DEFINED LLVM_ENABLE_ASSERTIONS)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR ENABLE_DEVELOPER_BUILD)
            set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
        else()
            set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "" FORCE)
        endif()
    endif()
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    
    # Note: When building with UB sanitizer, certain ARM-specific symbols
    # are not found by the linker, so we need to also build ARM libraries.
    # It is not clear whether it's a bug in UBSan + LLVM, or something else,
    # but doing it this way allows us to build with UB sanitizer.
    if(ENABLE_UB_SANITIZER)
        set(LLVM_TARGETS_TO_BUILD "host;ARM" CACHE STRING "" FORCE)
    else()
        set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
    endif()
    
    set(CROSS_TOOLCHAIN_FLAGS_ "" CACHE STRING "" FORCE)
    set(CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING "" FORCE)
    set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)
    # we do not use examples and having it enabled
    # makes cmake complains about long path on Windows
    set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLVM_BUILD_UTILS ON CACHE BOOL "" FORCE)
    set(LLVM_INSTALL_UTILS ON CACHE BOOL "" FORCE)
    set(LLVM_ABI_BREAKING_CHECKS "FORCE_OFF" CACHE STRING "" FORCE)

    if(ENABLE_LTO AND NOT MSVC)
        message(STATUS "LLVM_ENABLE_LTO is ON")
        set(LLVM_ENABLE_LTO "ON" CACHE STRING "" FORCE)
    endif()

    if(ANDROID)
        set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "" FORCE)
        set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "" FORCE)
        set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "" FORCE)
        set(LLVM_USE_HOST_TOOLS ON CACHE BOOL "" FORCE)
    endif()
endmacro()
