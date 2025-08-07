//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/net/IR/ops.hpp"

namespace vpux::net {

/** @brief Utility that sets up basic sections of the net::NetworkInfoOp.

    The function creates net::NetworkInfoOp sections if they are not created
    already. The sections are: inputsInfo, outputsInfo, profilingOutputsInfo.
 */
void setupSections(net::NetworkInfoOp netInfo, bool enableProfiling = false);

/** @brief Utility that erases entries from the net::NetworkInfoOp section.

    The function erases the entries (namely, net::DataInfoOp operations) within
    the specified section. The removal starts from @a begin.
 */
void eraseSectionEntries(mlir::Region& section, size_t begin = 0);

}  // namespace vpux::net
