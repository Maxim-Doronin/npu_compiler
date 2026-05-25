//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>

namespace vpux::bytecode {

// Represents a contiguous sequence of objects, with the first element of the sequence at position zero
template <typename T>
class Span {
    T* _data;
    size_t _size;

public:
    Span(T* data, size_t size) noexcept: _data{data}, _size{size} {
    }

    T& operator[](size_t i) noexcept {
        assert(i < _size && "Span index out of bounds");
        return *(begin() + i);
    }

    T const& operator[](size_t i) const noexcept {
        assert(i < _size && "Span index out of bounds");
        return *(begin() + i);
    }

    size_t size() const noexcept {
        return _size;
    }

    T* begin() noexcept {
        return _data;
    }

    const T* begin() const noexcept {
        return _data;
    }

    T* end() noexcept {
        return _data + _size;
    }

    const T* end() const noexcept {
        return _data + _size;
    }

    Span<T> subspan(size_t offset) const noexcept {
        if (offset > _size) {
            return Span<T>(nullptr, 0);
        }
        return Span<T>(_data + offset, _size - offset);
    }

    Span<T> subspan(size_t offset, size_t size) const noexcept {
        if (offset > _size || offset + size > _size) {
            return Span<T>(nullptr, 0);
        }
        return Span<T>(_data + offset, size);
    }
};

}  // namespace vpux::bytecode
