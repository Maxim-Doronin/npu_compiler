//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"
#include "vpux/utils/core/array_ref.hpp"

namespace vpux::IE::arch50xx {

/*
   Class for getting WeightsDequantizeToFakeQuantizeStrategy patterns for NPU50XX
*/
class WeightsDequantizeToFakeQuantizeStrategy final : public IDynamicRewriterStrategy {
public:
    explicit WeightsDequantizeToFakeQuantizeStrategy(ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index)
            : _benefitLevels(benefitLevels), _index(index) {};

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    ArrayRef<mlir::PatternBenefit> _benefitLevels;
    size_t _index;
};

}  // namespace vpux::IE::arch50xx
