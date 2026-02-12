//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
class NDTypeInterface;

class DMAPattern {
public:
    DMAPattern(SmallVector<int64_t> dims, SmallVector<int64_t> strides)
            : dims(std::move(dims)), strides(std::move(strides)) {
    }
    DMAPattern() = default;

    SmallVector<int64_t> dims;
    SmallVector<int64_t> strides;
};

class DMATransaction {
public:
    DMATransaction(SmallVector<DMAPattern> inputs, SmallVector<DMAPattern> outputs)
            : inputs(std::move(inputs)), outputs(std::move(outputs)) {
    }
    DMATransaction() = default;

    SmallVector<DMAPattern> inputs;
    SmallVector<DMAPattern> outputs;
};

DMAPattern reduceDimsForDma(SmallVector<int64_t> memShape, SmallVector<vpux::MemSize<vpux::MemType::Bit>> memStrides,
                            int64_t elemSize, bool strided);
void patchDimsForNPU37XX(DMAPattern& dmaPattern);

std::tuple<SmallVector<int64_t>, SmallVector<Bit>, int64_t> getTypeInfo(NDTypeInterface type);

}  // namespace vpux
