//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace vpux {
class DimsOrder;

DimsOrder inferNewDimsOrder(DimsOrder origOrder, size_t numShapeDims);

}  // namespace vpux
