//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

void vpux::ELF::DmaSymbolOp::serialize(elf::writer::DmaSymbol* dmaSymbol) {
    elf::DmaSymbolEntry dmaSymData{};

    dmaSymData.address = 0;
    auto dmaShapes = parseIntArrayAttr<uint32_t>(getDmaShapes());
    auto dmaStrides = parseIntArrayAttr<uint32_t>(getDmaStrides());
    auto strides = parseIntArrayAttr<uint32_t>(getTensorStrides());
    auto shapes = parseIntArrayAttr<uint32_t>(getTensorShapes());
    auto tileOffsets = parseIntArrayAttr<uint32_t>(getTileOffsets());

    VPUX_THROW_WHEN(dmaShapes.size() != elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS,
                    "dmaShapes not compatible with dma symbol");
    VPUX_THROW_WHEN(dmaStrides.size() != elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS,
                    "dmaStrides not compatible with dma symbol");
    VPUX_THROW_WHEN(strides.size() != elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, "strides not compatible with dma symbol");
    VPUX_THROW_WHEN(shapes.size() != elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, "shapes not compatible with dma symbol");
    VPUX_THROW_WHEN(tileOffsets.size() != elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS,
                    "tileOffsets not compatible with dma symbol");

    for (auto idx : vpux::irange(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS)) {
        dmaSymData.dmaShapes[idx] = dmaShapes[idx];
        dmaSymData.dmaStrides[idx] = dmaStrides[idx];
        dmaSymData.strides[idx] = strides[idx];
        dmaSymData.shapes[idx] = shapes[idx];
        dmaSymData.tileOffsets[idx] = tileOffsets[idx];
    }
    dmaSymData.dmaSize = getDmaSize();
    dmaSymData.ioIndex = getIoIndex();

    dmaSymbol->setDmaSymbol(dmaSymData);
}
