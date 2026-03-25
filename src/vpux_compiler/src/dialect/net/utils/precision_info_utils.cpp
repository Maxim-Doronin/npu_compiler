//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/net/utils/precision_info_utils.hpp"

namespace vpux::net {

PrecisionSensitiveOps::PrecisionSensitiveOps(mlir::ModuleOp module, vpux::Logger log): _logger(log), _lookup({}) {
    auto precOps = to_small_vector(module.getOps<net::PrecisionRequirementOp>());

    if (precOps.empty()) {
        return;
    }

    auto prec = precOps.front();
    auto& infos = prec.getPrecisionInfo().front().getOperations();
    for (auto& info : infos) {
        auto asOp = mlir::cast<net::PrecisionInfoOp>(info);
        const auto key = asOp.getOpName().str();
        _lookup.emplace(key, asOp);
    }
}

bool PrecisionSensitiveOps::isPrecisionSensitiveOp(mlir::Operation* op) const {
    auto loc = mlir::dyn_cast<mlir::FusedLoc>(op->getLoc());
    if (loc == nullptr) {
        return false;
    }
    auto locParts = loc.getLocations();
    if (locParts.empty()) {
        return false;
    }
    auto keyNameLoc = mlir::dyn_cast<mlir::NameLoc>(locParts.front());
    if (keyNameLoc == nullptr) {
        return false;
    }
    const auto key = keyNameLoc.getName().strref().data();
    for (auto it = _lookup.find(key); it != _lookup.end() && it->first == key; ++it) {
        auto opPrec = it->second;
        auto opTypeSaved = opPrec.getOpType();
        auto opType = op->getName();
        if (opType.getStringRef() == opTypeSaved) {
            return true;
        }
    }
    return false;
}

}  // namespace vpux::net
