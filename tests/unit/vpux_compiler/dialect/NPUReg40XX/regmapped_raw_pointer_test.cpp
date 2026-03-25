//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"

#include <cstring>
#include <npu_40xx_nnrt.hpp>

using namespace vpux::NPUReg40XX;
using namespace npu40xx;

using NPURegDescriptorOpInterfaceRawPointer = MLIR_UnitBase;

TEST_F(NPURegDescriptorOpInterfaceRawPointer, DescriptrRawPointer) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();

    nn_public::VpuMappedInference mi = {};
    // populate some random fields
    mi.vpu_nnrt_api_ver = 0x00FF;
    mi.dma_tasks_cmx_[4].count = 55;
    mi.dma_tasks_ddr_[0].count = 55;
    mi.variants[0].reserved1 = 55;
    mi.invariants[5].count = 55;
    mi.barrier_configs.address = 55;
    mi.shv_rt_configs.stack_frames[4] = 0x55;

    Descriptors::VpuMappedInference miDescriptor;
    auto descriptorRawStorage = miDescriptor.getStorage();
    std::copy_n(reinterpret_cast<uint8_t*>(&mi), descriptorRawStorage.size(), descriptorRawStorage.begin());
    EXPECT_TRUE(!std::memcmp(&mi, descriptorRawStorage.begin(), descriptorRawStorage.size()));
}
