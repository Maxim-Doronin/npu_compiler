//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"

using namespace vpux;

namespace {

inline void assertBound(int64_t dimValue, int64_t bound) {
    VPUX_THROW_WHEN(bound < 1, "Got non-positive shape dim bound: '{0}'", bound);
    VPUX_THROW_WHEN(dimValue != bound && dimValue > 0, "Got mismatching shape dim size: '{0}' and bound: '{1}'",
                    dimValue, bound);
}

inline void assertMask(int64_t dimValue) {
    VPUX_THROW_WHEN(dimValue < 1, "Got non-positive shape dim size or bound: '{0}'", dimValue);
}

}  // namespace

//
// BoundedDim
//

BoundedDim::BoundedDim(int64_t dimValue, int64_t bound): _dimValue(dimValue), _bound(bound) {
    assertBound(_dimValue, _bound);
}

BoundedDim::BoundedDim(int64_t dimValue): BoundedDim(dimValue, /*bound=*/dimValue) {
}

BoundedDim::BoundedDim(const MaskedDim& maskedDim): BoundedDim(maskedDim.dimValue(), maskedDim.reifiedSize()) {
}

int64_t BoundedDim::dimValue() const {
    return _dimValue;
}

int64_t BoundedDim::representation() const {
    return _bound;
}

bool BoundedDim::isDynamic() const {
    return _dimValue == mlir::ShapedType::kDynamic;
}

int64_t BoundedDim::reifiedSize() const {
    return _bound;
}

BoundedDim vpux::operator+(const BoundedDim& x, const BoundedDim& y) {
    return BoundedDim::apply(x, y, std::plus<>());
}

BoundedDim vpux::operator-(const BoundedDim& x, const BoundedDim& y) {
    return BoundedDim::apply(x, y, std::minus<>());
}

BoundedDim vpux::operator*(const BoundedDim& x, const BoundedDim& y) {
    return BoundedDim::apply(x, y, std::multiplies<>());
}

BoundedDim& BoundedDim::operator+=(const BoundedDim& other) {
    return *this = *this + other;
}

BoundedDim& BoundedDim::operator-=(const BoundedDim& other) {
    return *this = *this - other;
}

BoundedDim& BoundedDim::operator*=(const BoundedDim& other) {
    return *this = *this * other;
}

bool vpux::operator==(const BoundedDim& x, const BoundedDim& y) {
    return x.reifiedSize() == y.reifiedSize();
}

bool vpux::operator!=(const BoundedDim& x, const BoundedDim& y) {
    return !(x == y);
}

bool vpux::operator<(const BoundedDim& x, const BoundedDim& y) {
    return x.reifiedSize() < y.reifiedSize();
}

bool vpux::operator>(const BoundedDim& x, const BoundedDim& y) {
    return x.reifiedSize() > y.reifiedSize();
}

bool vpux::operator<=(const BoundedDim& x, const BoundedDim& y) {
    return x.reifiedSize() <= y.reifiedSize();
}

bool vpux::operator>=(const BoundedDim& x, const BoundedDim& y) {
    return x.reifiedSize() >= y.reifiedSize();
}

//
// MaskedDim
//

MaskedDim::MaskedDim(int64_t dimValue, int64_t isDynamic): _dimValue(dimValue), _isDynamic(isDynamic) {
    assertMask(_dimValue);
}

MaskedDim::MaskedDim(int64_t dimValue): MaskedDim(dimValue, /*isDynamic=*/false) {
}

MaskedDim::MaskedDim(const BoundedDim& boundedDim): MaskedDim(boundedDim.reifiedSize(), boundedDim.isDynamic()) {
}

int64_t MaskedDim::dimValue() const {
    return _isDynamic ? mlir::ShapedType::kDynamic : _dimValue;
}

int64_t MaskedDim::representation() const {
    return _isDynamic;
}

bool MaskedDim::isDynamic() const {
    return _isDynamic;
}

int64_t MaskedDim::reifiedSize() const {
    return _dimValue;
}

MaskedDim vpux::operator+(const MaskedDim& x, const MaskedDim& y) {
    return MaskedDim::apply(x, y, std::plus<>());
}

MaskedDim vpux::operator-(const MaskedDim& x, const MaskedDim& y) {
    return MaskedDim::apply(x, y, std::minus<>());
}

MaskedDim vpux::operator*(const MaskedDim& x, const MaskedDim& y) {
    return MaskedDim::apply(x, y, std::multiplies<>());
}

MaskedDim& MaskedDim::operator+=(const MaskedDim& other) {
    return *this = *this + other;
}

MaskedDim& MaskedDim::operator-=(const MaskedDim& other) {
    return *this = *this - other;
}

MaskedDim& MaskedDim::operator*=(const MaskedDim& other) {
    return *this = *this * other;
}

bool vpux::operator==(const MaskedDim& x, const MaskedDim& y) {
    return x.reifiedSize() == y.reifiedSize();
}

bool vpux::operator!=(const MaskedDim& x, const MaskedDim& y) {
    return !(x == y);
}

bool vpux::operator<(const MaskedDim& x, const MaskedDim& y) {
    return x.reifiedSize() < y.reifiedSize();
}

bool vpux::operator>(const MaskedDim& x, const MaskedDim& y) {
    return x.reifiedSize() > y.reifiedSize();
}

bool vpux::operator<=(const MaskedDim& x, const MaskedDim& y) {
    return x.reifiedSize() <= y.reifiedSize();
}

bool vpux::operator>=(const MaskedDim& x, const MaskedDim& y) {
    return x.reifiedSize() >= y.reifiedSize();
}
