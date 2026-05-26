//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Expected this test to fail, so revert error code
// RUN: not vpux-translate --platform=%platform% --import-IE ./IR/test_dynamic_shapes.xml 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Input: Validate IR without upper bounds specified, should fail on frontend level
// Case : Cannot handle IR without information about bounds

// CHECK: Upper bounds are not specified for node 'Relu_70' (type 'Relu'): input '0' bounds are '[1, 9223372036854775807, 3]'
// CHECK: Missing upper bound for one or more nodes
