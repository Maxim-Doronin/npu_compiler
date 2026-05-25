//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --create-elf-symbol-table %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

net.NetworkInfo entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}

func.func @oneDma() {
  ELF.Main {
    ELF.CreateSection @dsec1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    }
    ELF.CreateSection @dsec2 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    }
    ELF.CreateSection @dsec3 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
    }
    ELF.CreateLogicalSection @lsec1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
    }
    ELF.CreateLogicalSection @lsec2 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
    }
    ELF.CreateLogicalSection @lsec3 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
    }
  }
  return
}

//CHECK: ELF.Main

//CHECK: ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
//CHECK-NEXT: ELF.Symbol @elfsym.dsec1 of(@dsec1) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.dsec2 of(@dsec2) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.dsec3 of(@dsec3) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec1 of(@lsec1) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec2 of(@lsec2) type(<STT_SECTION>)
//CHECK-NEXT: ELF.Symbol @elfsym.lsec3 of(@lsec3) type(<STT_SECTION>)
