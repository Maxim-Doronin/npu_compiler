//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>
#include <initializer_list>

template <typename T, typename Tag>
struct NamedContainerType {
    explicit NamedContainerType(mlir::ArrayRef<T> value): _value(value.begin(), value.end()) {
    }
    NamedContainerType(std::initializer_list<T> list): _value(list) {
    }

    NamedContainerType() = default;
    NamedContainerType(const NamedContainerType&) = default;
    NamedContainerType(NamedContainerType&&) noexcept = default;
    NamedContainerType& operator=(const NamedContainerType&) = default;
    NamedContainerType& operator=(NamedContainerType&&) noexcept = default;
    ~NamedContainerType() = default;

    void printFormat(llvm::raw_ostream& stream) const {
        stream << formatv("{0}", raw());
    }

    auto raw() const {
        return _value;
    }

    auto begin() const {
        return _value.begin();
    }

    auto end() const {
        return _value.end();
    }

    auto rbegin() const {
        return _value.rbegin();
    }

    auto rend() const {
        return _value.rend();
    }

    const auto& front() const {
        assert(!empty());
        return _value.front();
    }

    const auto& back() const {
        assert(!empty());
        return _value.back();
    }

    const auto& operator[](std::size_t idx) const {
        return _value[idx];
    }

    auto size() const {
        return _value.size();
    }

    auto empty() const {
        return _value.empty();
    }

private:
    mlir::SmallVector<T> _value = {};
};

struct BoundsTag {};
struct DynamicDimsMaskTag {};

// Dynamic tensor can have two representations
// Upper bounded representation - dynamic shape with static bounds attribute
// Dynamic dims mask representation - static shape equal to the upper bounds with an attribute
// where each value represents static dimension with (0) or dynamic dimension with (1)
using Bounds = NamedContainerType<int64_t, BoundsTag>;
using DynamicDimsMask = NamedContainerType<int64_t, DynamicDimsMaskTag>;
