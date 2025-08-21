#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(const)
add_vpux_ops(const GENERIC)
add_vpux_attr_interface(Const dialect/const/)
add_vpux_attribute(Const ENABLE_VPUX_ATTR)
add_vpux_pass(Const Const dialect/const/)
