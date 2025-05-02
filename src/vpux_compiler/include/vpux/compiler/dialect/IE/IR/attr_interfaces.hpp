//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Attributes.h>

namespace vpux {
namespace IE {

//
// ChannelAgnosticAttr
//

template <typename ConcreteAttr>
class ChannelAgnosticAttr : public mlir::AttributeTrait::TraitBase<ConcreteAttr, ChannelAgnosticAttr> {};

}  // namespace IE
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/IE/attr_interfaces.hpp.inc>
