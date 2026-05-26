//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/icompiler.hpp"

namespace vpux {

class CompilerImpl final : public ICompiler {
public:
    CompilerImpl();
    ~CompilerImpl() override = default;

    CompilerImpl(const CompilerImpl&) = delete;
    CompilerImpl(CompilerImpl&&) = delete;
    CompilerImpl& operator=(const CompilerImpl&) = delete;
    CompilerImpl& operator=(CompilerImpl&&) = delete;

    // Mutable model variant for direct use with deserialized model in VCL
    NetworkDescription compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config) const;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                               const intel_npu::Config& config) const final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const intel_npu::Config& config) const final;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const intel_npu::Config&) const final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final;

    // CiD-specific methods

    NetworkDescriptionView compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                   BlobAllocator& allocator, bool generateCompatibilityString = false) const override;

    NetworkDescriptionView compile(const std::shared_ptr<const ov::Model>& model, const intel_npu::Config& config,
                                   BlobAllocator& allocator, bool generateCompatibilityString = false) const override;

    // WS CiP-specific methods

    /// @brief Returns Init schedules and Main in a single call. There is always exactly one Main schedule, placed at
    /// the back of the vector.
    std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                      const intel_npu::Config& config) const override;

    /// @brief Sequentially compiles Init and Main schedules. The Main schedule is always last.
    NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                          size_t callIdx) const override;

    // WS VCL-specific methods

    /// @brief Returns Init schedules and Main in a single call. The blobs are allocated using the provided allocator.
    /// There is always exactly one Main schedule, placed at the back of the vector.
    std::vector<std::shared_ptr<NetworkDescriptionView>> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                          const intel_npu::Config& config,
                                                                          BlobAllocator& allocator) const override;

    /// @brief Sequentially compiles Init and Main schedules. The blob is allocated using the provided allocator. The
    /// Main schedule is always last.
    NetworkDescriptionView compileWsIterative(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                              size_t callIdx, BlobAllocator& allocator) const override;
};

}  // namespace vpux
