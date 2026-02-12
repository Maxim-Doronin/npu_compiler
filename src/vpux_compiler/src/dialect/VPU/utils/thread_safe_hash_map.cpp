//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/thread_safe_hash_map.hpp"
#include "vpux/utils/core/dense_map.hpp"

#ifdef TBB_AVAILABLE
#include <tbb/concurrent_hash_map.h>
#endif

#include <llvm/ADT/Hashing.h>
#include <mutex>
#include <optional>

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
namespace details {

// ============================================================================
// MutexHashMapImpl Implementation
// ============================================================================

template <typename KeyT, typename ValueT>
class MutexHashMapImpl {
public:
    std::optional<ValueT> find(const KeyT& key) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void insert(const KeyT& key, const ValueT& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        _map[key] = value;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _map.clear();
    }

private:
    DenseMap<KeyT, ValueT> _map;
    mutable std::mutex _mutex;
};

// ============================================================================
// TBBHashMapImpl Implementation
// ============================================================================

#ifdef TBB_AVAILABLE
template <typename KeyT, typename ValueT>
class TBBHashMapImpl {
public:
    std::optional<ValueT> find(const KeyT& key) const {
        typename MapType::const_accessor accessor;
        if (_map.find(accessor, key)) {
            return accessor->second;
        }
        return std::nullopt;
    }

    void insert(const KeyT& key, const ValueT& value) {
        typename MapType::accessor accessor;
        _map.insert(accessor, key);
        accessor->second = value;
    }

    void clear() {
        _map.clear();
    }

private:
    tbb::concurrent_hash_map<KeyT, ValueT> _map;
};
#endif

}  // namespace details

// ============================================================================
// ThreadSafeHashMap Implementation
// ============================================================================

template <typename KeyT, typename ValueT>
class ThreadSafeHashMap<KeyT, ValueT>::Impl {
#ifdef TBB_AVAILABLE
    details::TBBHashMapImpl<KeyT, ValueT> _map;
#else
    details::MutexHashMapImpl<KeyT, ValueT> _map;
#endif

public:
    std::optional<ValueT> find(const KeyT& key) const {
        return _map.find(key);
    }

    void insert(const KeyT& key, const ValueT& value) {
        _map.insert(key, value);
    }

    void clear() {
        _map.clear();
    }
};

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>::ThreadSafeHashMap(): _impl(std::make_unique<Impl>()) {
}

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>::~ThreadSafeHashMap() = default;

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>::ThreadSafeHashMap(ThreadSafeHashMap&&) noexcept = default;

template <typename KeyT, typename ValueT>
ThreadSafeHashMap<KeyT, ValueT>& ThreadSafeHashMap<KeyT, ValueT>::operator=(ThreadSafeHashMap&&) noexcept = default;

template <typename KeyT, typename ValueT>
std::optional<ValueT> ThreadSafeHashMap<KeyT, ValueT>::find(const KeyT& key) const {
    return _impl->find(key);
}

template <typename KeyT, typename ValueT>
void ThreadSafeHashMap<KeyT, ValueT>::insert(const KeyT& key, const ValueT& value) {
    _impl->insert(key, value);
}

template <typename KeyT, typename ValueT>
void ThreadSafeHashMap<KeyT, ValueT>::clear() {
    _impl->clear();
}

// ============================================================================
// Explicit Instantiations
// ============================================================================
// Required for PIMPL pattern: the destructor and other special member functions
// must be instantiated explicitly because Impl is an incomplete type in the header.

using NTilesOnDim = Shape;
using PerClusterShapeCacheItem = std::optional<SmallVector<Shape>>;
using DimArr = SmallVector<Dim>;

template class ThreadSafeHashMap<llvm::hash_code, std::optional<NTilesOnDim>>;
template class ThreadSafeHashMap<llvm::hash_code, std::optional<::llvm::hash_code>>;
template class ThreadSafeHashMap<llvm::hash_code, SmallVector<uint32_t>>;
template class ThreadSafeHashMap<llvm::hash_code, uint32_t>;
template class ThreadSafeHashMap<llvm::hash_code, PerClusterShapeCacheItem>;
template class ThreadSafeHashMap<llvm::hash_code, SmallVector<DimArr>>;
template class ThreadSafeHashMap<llvm::hash_code, DimArr>;
template class ThreadSafeHashMap<llvm::hash_code, SmallVector<vpux::NDTypeInterface>>;
}  // namespace vpux
