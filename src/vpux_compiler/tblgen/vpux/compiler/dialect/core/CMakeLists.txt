#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(Core)
add_vpux_ops(Core GENERIC)
add_vpux_attr_interface(Core core/)
add_vpux_ops_interface(Core core/)
add_vpux_type_interface(Core core/)
add_vpux_type(Core core/)
add_vpux_pass(Core Core core/)

add_vpux_attribute(Core ENABLE_VPUX_ATTR)
