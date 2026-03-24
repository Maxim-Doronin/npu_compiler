//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/counters_category.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIP {

//
// SpecificCategoryCounter Implementation
//

bool SpecificCategoryCounter::record(mlir::Operation* op) {
    if (!utils::OpCounter::record(op)) {
        return false;
    }

    if (mlir::isa_and_nonnull<VPUIP::DMATypeOpInterface>(op)) {
        auto opType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
        _size += opType.getCompactAllocSize().count();
    }

    if (auto nceOp = mlir::dyn_cast_if_present<VPUIP::NCEClusterTaskOp>(op)) {
        auto outputShape = mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape();
        const auto outputHeight = outputShape[Dims4D::Act::H];
        const auto outputWidth = outputShape[Dims4D::Act::W];
        const auto outputChannels = outputShape[Dims4D::Act::C];
        if (auto kernelSizeOpt = nceOp.getKernelSize()) {
            const auto kernelSize = parseIntArrayAttr<int64_t>(*kernelSizeOpt);
            const auto kernelHeight = kernelSize[Dims4D::Kernel::Y.ind()];
            const auto kernelWidth = kernelSize[Dims4D::Kernel::X.ind()];

            auto inputShape = mlir::cast<vpux::NDTypeInterface>(nceOp.getInput().getType()).getShape();
            const auto inputChannels = inputShape[Dims4D::Act::C];

            const size_t isNotConvOpCount = outputHeight * outputWidth * kernelHeight * kernelWidth * outputChannels;
            _opCount += nceOp.getTaskType() == VPUIP::NCETaskType::CONV ? inputChannels * isNotConvOpCount
                                                                        : isNotConvOpCount;
        } else if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE) {
            _opCount += (outputHeight * outputWidth * outputChannels);
        }
    }
    return true;
}

void SpecificCategoryCounter::printStatistics(const vpux::Logger& log) const {
    if (_count == 0) {
        return;
    }
    if (printDMASizes(_category, _size)) {
        log.info("{0} - {1} ops : Size - {2}", _category, _count, convertBytesToReadableSize(_size));
    } else if (_category == "opCount") {
        log.info("{0} - {1} ops", _category, _opCount);
    } else {
        log.info("{0} - {1} ops", _category, _count);
    }
}

//
// Utility Functions
//

bool printDMASizes(const std::string& category, const uint64_t& size) {
    std::vector<std::string> dmaSubStrings = {"DDR", "CMX", "NNDMA"};

    bool anyDmaSubStringFound =
            std::any_of(dmaSubStrings.begin(), dmaSubStrings.end(), [&](const std::string& dmaSubString) {
                return category.find(dmaSubString) != std::string::npos;
            });

    return anyDmaSubStringFound && size;
}

std::string convertBytesToReadableSize(uint64_t bytes) {
    const uint64_t kilobyte = 1024;
    const uint64_t megabyte = kilobyte * 1024;
    const uint64_t gigabyte = megabyte * 1024;

    std::string result;
    if (bytes >= gigabyte) {
        double size = static_cast<double>(bytes) / gigabyte;
        result = std::to_string(size);
        result.resize(result.find('.') + 3);  // Truncate to two decimal digits
        result += " GB";
    } else if (bytes >= megabyte) {
        double size = static_cast<double>(bytes) / megabyte;
        result = std::to_string(size);
        result.resize(result.find('.') + 3);
        result += " MB";
    } else if (bytes >= kilobyte) {
        double size = static_cast<double>(bytes) / kilobyte;
        result = std::to_string(size);
        result.resize(result.find('.') + 3);
        result += " KB";
    } else {
        result = std::to_string(bytes) + " bytes";
    }

    return result;
}

CountersNode makeCounterNode(const std::string& category, utils::OpCounter::IsOperationSuitable predicate,
                             CountersVec&& nestedCounters, utils::OpCounter::HandleUnrecognizedCounter handler) {
    return {std::unique_ptr<utils::OpCounter>(
                    new SpecificCategoryCounter(category, std::move(predicate), std::move(handler))),
            std::move(nestedCounters)};
}

bool isOpCounterSupported(mlir::Operation* op) {
    if (auto nceOp = mlir::dyn_cast_if_present<VPUIP::NCEClusterTaskOp>(op)) {
        return nceOp.getTaskType() == VPUIP::NCETaskType::CONV || nceOp.getTaskType() == VPUIP::NCETaskType::DWCONV ||
               nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE ||
               nceOp.getTaskType() == VPUIP::NCETaskType::MAXPOOL || nceOp.getTaskType() == VPUIP::NCETaskType::AVEPOOL;
    }
    return false;
}

}  // namespace vpux::VPUIP
