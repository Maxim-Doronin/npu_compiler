#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(VPU)
add_vpux_ops(VPU GENERIC)
add_vpux_ops_interface(VPU dialect/VPU/)
add_vpux_attr_interface(VPU dialect/VPU/)
add_vpux_type_interface(VPU dialect/VPU/)
add_vpux_attribute(VPU ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(VPU)
add_vpux_pass(VPU VPU dialect/VPU/)
