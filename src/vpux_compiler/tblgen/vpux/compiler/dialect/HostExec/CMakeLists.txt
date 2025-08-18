#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(HostExec)
add_vpux_ops(HostExec GENERIC)
add_vpux_pass(HostExec HostExec dialect/HostExec/)
add_vpux_attribute(HostExec ENABLE_VPUX_ATTR)
add_vpux_type(HostExec)
