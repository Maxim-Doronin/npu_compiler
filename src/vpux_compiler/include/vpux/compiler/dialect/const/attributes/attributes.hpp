//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux::Const {
/// @brief This function returns true if attr can change the shape. This is currently the case for PadWithZeroAttr,
/// BroadcastAttr, ReshapeAttr, SubViewAttr, AffineReshapeAttr.
bool canChangeShape(Const::TransformAttrInterface attr);
}  // namespace vpux::Const
