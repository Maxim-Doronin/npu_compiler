//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/Hashing.h>

namespace vpux::Const::details {

/** @brief Implicit storage class for Const transformations.

    Encapsulates auxiliary information associated with transformation attributes
    from Const dialect.

    This storage is the underlying data storage used internally by MLIR,
    returned by mlir::Attribute::getImpl().

    @note Unlike the tablegen-generated storage type, this storage is not
    supposed to be used externally. This is purely an implementation detail.
 */
template <typename BaseStorage>
struct StableHashStorage : BaseStorage {
    template <typename... Args>
    StableHashStorage(Args&&... args): BaseStorage(std::forward<Args>(args)...), stableHash(calculateStableHash()) {
    }

    llvm::hash_code stableHash;  //!< cached stable hash of the attribute, associated with this storage

private:
    //! @brief Helper function that calculates the stable hash.
    llvm::hash_code calculateStableHash() const;
};

}  // namespace vpux::Const::details
