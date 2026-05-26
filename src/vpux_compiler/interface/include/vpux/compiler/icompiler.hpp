//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/config/config.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "vpux/compiler/network_metadata.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux {

constexpr uint32_t SUPPORTED_OPSET = 11;

class BlobAllocator {
public:
    virtual ~BlobAllocator() = default;
    virtual uint8_t* allocate(vpux::Byte) = 0;
    virtual void deallocate(uint8_t*) = 0;
};

// Non-owning view into a memory occupied by a blob. Used by AllocatedCompiledNetwork
// to store compiled model allocated via BlobAllocator implementation.
struct BlobView final {
    // E#-140887: ptr left mutable to be compatible with initial version of VCL
    // interface; make BlobView immutable and reuse it in CompiledNetwork
    uint8_t* ptr = nullptr;
    uint64_t size = 0;

    BlobView(uint8_t*, uint64_t);
    // E#-140887: enable implicit conversion from std::vector<uint8_t>
    // currently it'll fail due to blob.data() being const uint8_t* that
    // can't be converted to uint8_t*
    // /* implicit */ BlobView(const std::vector<uint8_t>& blob);
};

// The object returned by the compiler to provide such information about a network
// as description of inputs and outputs, name and compiled network in a format
// executable by device
// The difference between NetworkDescriptionView and NetworkDescription is
// compiled network is represented via BlobView. Blob in this case is allocated by
// compiler via provided BlobAllocator implementation.
struct NetworkDescriptionView {
    NetworkDescriptionView(BlobView blob, NetworkMetadata&&);
    NetworkDescriptionView(BlobView blob, BlobView compatibilityString, NetworkMetadata&&);

    NetworkDescriptionView(const NetworkDescriptionView&) = delete;
    NetworkDescriptionView& operator=(const NetworkDescriptionView&) = delete;

    NetworkDescriptionView(NetworkDescriptionView&&) = default;
    NetworkDescriptionView& operator=(NetworkDescriptionView&&) = default;

    ~NetworkDescriptionView() = default;

    BlobView compiledNetwork;
    BlobView compatibilityString;
    NetworkMetadata metadata;
};

/**
 * @interface ICompiler
 * @brief An interface to be implemented by a concrete compiler to provide
 * methods for preparing a network for execution on a NPU device
 */
class ICompiler : public std::enable_shared_from_this<ICompiler> {
protected:
    ICompiler() = default;

public:
    virtual ~ICompiler() = default;

    ICompiler(const ICompiler&) = delete;
    ICompiler(ICompiler&&) = delete;
    ICompiler& operator=(const ICompiler&) = delete;
    ICompiler& operator=(ICompiler&&) = delete;

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device
     * @param model a shared pointer to the OpenVINO model to be compiled
     * @param config a reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                                       const intel_npu::Config& config) const = 0;

    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main one are compiled in
     * the same scope.
     * @return A "NetworkDescription" object for each init schedule, followed by another one corresponding to the main
     * part.
     */
    virtual std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(
            const std::shared_ptr<ov::Model>& /*model*/, const intel_npu::Config& /*config*/) const {
        OPENVINO_NOT_IMPLEMENTED;
    }

    /**
     * @brief Sequential compilation of Init(s) and Main
     *
     * "Stateless compiler" approach
     * We want to get multiple Inits in the case of a large number of weights.
     * This allows us to build pipeline:
     * Allocate W1 -> Init1
     *             Allocate W2 -> Init2
     *                          Allocate W3 -> Init2
     *
     * This is why there is an additional parameter callNumber:
     * Compiler should somehow understand which Init(or Main) to return
     * Plugin does not know total numbers of Init schedules
     */
    virtual NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& /*model*/,
                                                  const intel_npu::Config& /*config*/, size_t /*callNumber*/) const {
        OPENVINO_NOT_IMPLEMENTED;
    }

    /**
     * @brief Returns information about supported layers of the network passed
     * @param model The model to be queried
     * @param config A reference to NPUConfig containing plugin config options
     *        including config options related to compilation
     * @returns SupportedOpsMap structure with information about supported layers
     */
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                                      const intel_npu::Config& config) const = 0;

    /**
     * @brief Parses already compiled network to extract meta information:
     *        inputs and outputs descriptions
     * @param network compiled network represented as a vector of char
     * @param config a reference to NPUConfig containing plugin config options
     *        Note: compilation options will be ignored,
     *        since the network is already compiled
     * @return a shared pointer on an object implementing NetworkDescription interface
     */
    virtual NetworkMetadata parse(const std::vector<uint8_t>& network, const intel_npu::Config& config) const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const std::vector<uint8_t>& network,
                                                                    const intel_npu::Config& config) const = 0;

    // CiD-specific methods

    virtual NetworkDescriptionView compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                           BlobAllocator& allocator,
                                           bool generateCompatibilityString = false) const = 0;

    virtual NetworkDescriptionView compile(const std::shared_ptr<const ov::Model>& model,
                                           const intel_npu::Config& config, BlobAllocator& allocator,
                                           bool generateCompatibilityString = false) const = 0;

    // WS VCL-specific methods

    /// @brief Returns Init schedules and Main in a single call. The blobs are allocated using the provided allocator.
    /// There is always exactly one Main schedule, placed at the back of the vector.
    virtual std::vector<std::shared_ptr<NetworkDescriptionView>> compileWsOneShot(
            const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
            BlobAllocator& allocator) const = 0;

    /// @brief Sequentially compiles Init and Main schedules. The blob is allocated using the provided allocator. The
    /// Main schedule is always last.
    virtual NetworkDescriptionView compileWsIterative(const std::shared_ptr<ov::Model>& model,
                                                      const intel_npu::Config& config, size_t callIdx,
                                                      BlobAllocator& allocator) const = 0;
};

}  // namespace vpux
