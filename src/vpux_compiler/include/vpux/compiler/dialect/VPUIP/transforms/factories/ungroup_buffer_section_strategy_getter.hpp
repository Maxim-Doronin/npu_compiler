//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

namespace vpux::VPUIP {

class UngroupBufferSectionStrategy final : public IDynamicRewriterStrategy {
public:
    explicit UngroupBufferSectionStrategy(bool enableReorderSubViewOp)
            : _enableReorderSubViewOp(enableReorderSubViewOp) {
    }

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    bool _enableReorderSubViewOp = false;
};

std::unique_ptr<IDynamicRewriterStrategy> createUngroupBufferSectionStrategy(bool enableReorderSubViewOp = false);
}  // namespace vpux::VPUIP
