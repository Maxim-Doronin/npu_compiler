//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>

namespace vpux {
template <typename KeyT, typename ValueT>
class ThreadSafeHashMap {
public:
    ThreadSafeHashMap();
    ~ThreadSafeHashMap();

    ThreadSafeHashMap(const ThreadSafeHashMap&) = delete;
    ThreadSafeHashMap& operator=(const ThreadSafeHashMap&) = delete;
    ThreadSafeHashMap(ThreadSafeHashMap&&) noexcept;
    ThreadSafeHashMap& operator=(ThreadSafeHashMap&&) noexcept;

    std::optional<ValueT> find(const KeyT& key) const;

    void insert(const KeyT& key, const ValueT& value);

    void clear();

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

}  // namespace vpux
