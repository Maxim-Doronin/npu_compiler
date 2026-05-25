//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=ReferenceSW revision-id=3" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU4000

// CHECK: module @test attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>, config.elf_version = #config.version<{{[^>]*}}>, config.platform = #config.platform<NPU4000>, config.revisionID = #config.revision_id<REVISION_B>}
module @test {
}
