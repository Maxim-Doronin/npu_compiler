//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ov::test::utils {

struct MismatchInfo {
    size_t flatIndex;
    std::vector<size_t> multiIndex;
    double expected;
    double actual;
    double absError;
    double tolerance;  // atol + rtol * |expected|
};

struct TensorComparisonResult {
    ov::Shape shape;
    ov::element::Type elementType;
    size_t totalElements = 0;
    double absThreshold = 0.0;
    double relThreshold = 0.0;

    size_t expectedNanCount = 0;
    size_t expectedInfCount = 0;
    size_t actualNanCount = 0;
    size_t actualInfCount = 0;
    size_t mismatchCount = 0;

    double expectedMin = 0.0;
    double expectedMax = 0.0;
    double actualMin = 0.0;
    double actualMax = 0.0;

    std::string errorMessage;  // non-empty when comparison could not run (shape/type mismatch, unsupported type)
    std::vector<MismatchInfo> worstMismatches;  // up to topN, sorted by absError desc

    bool passed() const {
        return errorMessage.empty() && mismatchCount == 0;
    }

    double mismatchPercentage() const {
        return totalElements > 0 ? 100.0 * static_cast<double>(mismatchCount) / static_cast<double>(totalElements)
                                 : 0.0;
    }

    bool hasExpectedAnomalies() const {
        return expectedNanCount > 0 || expectedInfCount > 0;
    }
};

// Compare a single tensor pair. Returns detailed comparison result.
TensorComparisonResult compareTensors(const ov::Tensor& expected, const ov::Tensor& actual, double absThreshold,
                                      double relThreshold, size_t topN = 10);

// Format result as human-readable string for error reporting.
std::string formatComparisonResult(const TensorComparisonResult& result, std::string_view tensorName = "");

// Returns a warning string if the expected tensor had NaN/Inf, empty string otherwise.
std::string formatExpectedAnomalyWarning(const TensorComparisonResult& result, std::string_view tensorName = "");

// GTEST helper: compare all output tensors, fail with formatted message if any mismatch.
// Can be called from compare() overrides for easy integration.
void assertTensorsClose(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual,
                        double absThreshold, double relThreshold, size_t topN = 10);

}  // namespace ov::test::utils
