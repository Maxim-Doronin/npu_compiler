//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/counters_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/unroll_distributed_ops_strategy.hpp"

namespace vpux::VPUIP {

class StrategyFactory {
public:
    virtual ~StrategyFactory() = default;

    virtual std::unique_ptr<IIterativeWalkPassStrategy> getUnrollDepthToSpaceDMAStrategy(mlir::MLIRContext* ctx,
                                                                                         int64_t dmaPortCount) = 0;
    virtual std::unique_ptr<IIterativeWalkPassStrategy> getUnrollSpaceToDepthDMAStrategy(mlir::MLIRContext* ctx,
                                                                                         int64_t dmaPortCount) = 0;
    virtual std::unique_ptr<IIterativeWalkPassStrategy> getUnrollPermuteDMAStrategy(mlir::MLIRContext* ctx,
                                                                                    int64_t dmaPortCount) = 0;
    virtual std::unique_ptr<IGreedilyPassStrategy> getUnrollPerAxisTileDMAStrategy(int64_t dmaPortCount) = 0;
    virtual std::unique_ptr<IUnrollDistributedOpsStrategy> getUnrollDistributedOpsStrategy(
            mlir::func::FuncOp funcOp, std::optional<bool> enableSegmentedDmaFusion) = 0;
    virtual std::unique_ptr<ICountersStrategy> getCountersStrategy() = 0;
};

class StrategyFactoryCache final : public mlir::DialectInterface::Base<StrategyFactoryCache> {
    std::unique_ptr<StrategyFactory> _strategyFactory = nullptr;

public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrategyFactoryCache)

    StrategyFactoryCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const std::unique_ptr<VPUIP::StrategyFactory>& getStrategyFactory() const {
        return _strategyFactory;
    }

    void setStrategyFactory(std::unique_ptr<VPUIP::StrategyFactory> strategyFactory) {
        _strategyFactory = std::move(strategyFactory);
    }
};

void setVPUIPStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<VPUIP::StrategyFactory> factory);
const std::unique_ptr<VPUIP::StrategyFactory>& getVPUIPStrategyFactory(mlir::MLIRContext* context);

}  // namespace vpux::VPUIP
