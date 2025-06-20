// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <mlir/IR/TypeSupport.h>
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

using namespace mlir;

namespace vpux {
namespace detail {

/// Quantile float Type Storage and Uniquing.
struct QuantileFloatTypeStorage : public mlir::TypeStorage {
    Type storageType;
    Type quantileType;
    const double* quantilesElements;
    size_t quantilesParamsSize;

    struct KeyTy {
        KeyTy(Type storageType, Type quantileType, ArrayRef<double> quantiles)
                : storageType(storageType), quantileType(quantileType), quantiles(quantiles) {
        }

        Type storageType;
        Type quantileType;
        ArrayRef<double> quantiles;
        Type getStorageType() const {
            return storageType;
        }
        Type getQuantileType() const {
            return quantileType;
        }
        ArrayRef<double> getQuantiles() const {
            return quantiles;
        }

        template <typename T, typename U>
        static bool genericIsEqual(const T& lhs, const U& rhs) {
            return lhs.getStorageType() == rhs.getStorageType() && lhs.getQuantileType() == rhs.getQuantileType() &&
                   lhs.getQuantiles() == rhs.getQuantiles();
        }

        bool operator==(const KeyTy& other) const {
            return genericIsEqual(*this, other);
        }

        unsigned getHashValue() const {
            int64_t* quantilesCast = llvm::bit_cast<int64_t*>(quantiles.data());
            ArrayRef<int64_t> quantilesBits(quantilesCast, quantiles.size());
            return static_cast<unsigned>(llvm::hash_combine(
                    llvm::hash_combine_range(quantilesBits.begin(), quantilesBits.end()), storageType, quantileType));
        }
    };

    bool operator==(const KeyTy& key) const {
        return KeyTy::genericIsEqual(*this, key);
    }

    QuantileFloatTypeStorage(const KeyTy& key, ArrayRef<double> quantiles)
            : storageType(key.storageType),
              quantileType(key.quantileType),
              quantilesElements(quantiles.data()),
              quantilesParamsSize(quantiles.size()) {
    }

    static QuantileFloatTypeStorage* construct(TypeStorageAllocator& allocator, KeyTy key) {
        ArrayRef<double> quantiles = allocator.copyInto(key.quantiles);
        return new (allocator.allocate<QuantileFloatTypeStorage>()) QuantileFloatTypeStorage(key, quantiles);
    }

    static unsigned hashKey(const KeyTy& key) {
        return key.getHashValue();
    }

    ArrayRef<double> getQuantiles() const {
        return ArrayRef<double>(quantilesElements, quantilesParamsSize);
    }

    Type getStorageType() const {
        return storageType;
    }

    Type getQuantileType() const {
        return quantileType;
    }
};

}  // namespace detail
}  // namespace vpux
