//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="platform=%platform%" --setup-location-verifier="mode=fast" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

module @mainModule {
}
// CHECK: module @mainModule attributes
// CHECK-SAME: IE.LocationsVerificationMode  = "fast"
