//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include "intel_npu/npu_mlir_runtime.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"

#include <openvino/openvino.hpp>
#include <openvino/util/file_util.hpp>
#include <openvino/util/shared_object.hpp>
#include "vpux/compiler/network_metadata.hpp"

struct npu_mlir_runtime_fntbl {
    using npu_mlir_runtime_get_api_version_t = npu_mlir_runtime_result_t(npu_mlir_runtime_version_t* pVersion);
    using npu_mlir_runtime_create_t = npu_mlir_runtime_result_t(const npu_mlir_runtime_blob_desc_t* desc,
                                                                npu_mlir_runtime_handle_t* phRuntime,
                                                                const npu_mlir_runtime_properties_t* pProperties);
    using npu_mlir_runtime_destroy_t = npu_mlir_runtime_result_t(npu_mlir_runtime_handle_t hRuntime);
    using npu_mlir_runtime_get_metadata_t =
            npu_mlir_runtime_result_t(npu_mlir_runtime_handle_t hRuntime, uint32_t argIndex,
                                      ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                                      ze_graph_argument_metadata_t* pGraphArgumentMetadata, int64_t* upperBound);
    using npu_mlir_runtime_execute_t = npu_mlir_runtime_result_t(npu_mlir_runtime_handle_t hRuntime,
                                                                 npu_mlir_runtime_execute_params_t* pParams);
    using npu_mlir_runtime_predict_output_shape_t = npu_mlir_runtime_result_t(
            npu_mlir_runtime_handle_t hRuntime, npu_mlir_runtime_predict_output_shape_params_t* pParams);

    using npu_mlir_runtime_create_mem_ref_t = npu_mlir_runtime_result_t(uint32_t dimsCount,
                                                                        npu_mlir_runtime_mem_ref_handle_t* phMemRef);

    using npu_mlir_runtime_destroy_mem_ref_t = npu_mlir_runtime_result_t(npu_mlir_runtime_mem_ref_handle_t hMemRef);

    using npu_mlir_runtime_set_mem_ref_t = npu_mlir_runtime_result_t(npu_mlir_runtime_mem_ref_handle_t hMemRef,
                                                                     const void* basePtr, const void* data,
                                                                     int64_t offset, int64_t* pSizes, int64_t* pStrides,
                                                                     int64_t dimsCount);

    using npu_mlir_runtime_parse_mem_ref_t = npu_mlir_runtime_result_t(npu_mlir_runtime_mem_ref_handle_t hMemRef,
                                                                       const void** pBasePtr, const void** pData,
                                                                       int64_t* pOffset, int64_t* pSizes,
                                                                       int64_t* pStrides, int64_t* pDimsCount);

    std::function<npu_mlir_runtime_get_api_version_t> npuMLIRRuntimeGetAPIVersion = nullptr;
    std::function<npu_mlir_runtime_create_t> npuMLIRRuntimeCreate = nullptr;
    std::function<npu_mlir_runtime_destroy_t> npuMLIRRuntimeDestroy = nullptr;
    std::function<npu_mlir_runtime_get_metadata_t> npuMLIRRuntimeGetMetadata = nullptr;
    std::function<npu_mlir_runtime_execute_t> npuMLIRRuntimeExecute = nullptr;
    std::function<npu_mlir_runtime_predict_output_shape_t> npuMLIRRuntimePredictOutputShape = nullptr;
    std::function<npu_mlir_runtime_create_mem_ref_t> npuMLIRRuntimeCreateMemRef = nullptr;
    std::function<npu_mlir_runtime_destroy_mem_ref_t> npuMLIRRuntimeDestroyMemRef = nullptr;
    std::function<npu_mlir_runtime_set_mem_ref_t> npuMLIRRuntimeSetMemRef = nullptr;
    std::function<npu_mlir_runtime_parse_mem_ref_t> npuMLIRRuntimeParseMemRef = nullptr;

    void init(std::shared_ptr<void> lib) {
        npuMLIRRuntimeGetAPIVersion = reinterpret_cast<npu_mlir_runtime_get_api_version_t*>(
                ov::util::get_symbol(lib, "npuMLIRRuntimeGetAPIVersion"));
        npuMLIRRuntimeCreate =
                reinterpret_cast<npu_mlir_runtime_create_t*>(ov::util::get_symbol(lib, "npuMLIRRuntimeCreate"));
        npuMLIRRuntimeDestroy =
                reinterpret_cast<npu_mlir_runtime_destroy_t*>(ov::util::get_symbol(lib, "npuMLIRRuntimeDestroy"));
        npuMLIRRuntimeGetMetadata = reinterpret_cast<npu_mlir_runtime_get_metadata_t*>(
                ov::util::get_symbol(lib, "npuMLIRRuntimeGetMetadata"));
        npuMLIRRuntimeExecute =
                reinterpret_cast<npu_mlir_runtime_execute_t*>(ov::util::get_symbol(lib, "npuMLIRRuntimeExecute"));
        npuMLIRRuntimePredictOutputShape = reinterpret_cast<npu_mlir_runtime_predict_output_shape_t*>(
                ov::util::get_symbol(lib, "npuMLIRRuntimePredictOutputShape"));
        npuMLIRRuntimeCreateMemRef = reinterpret_cast<npu_mlir_runtime_create_mem_ref_t*>(
                ov::util::get_symbol(lib, "npuMLIRRuntimeCreateMemRef"));
        npuMLIRRuntimeDestroyMemRef = reinterpret_cast<npu_mlir_runtime_destroy_mem_ref_t*>(
                ov::util::get_symbol(lib, "npuMLIRRuntimeDestroyMemRef"));
        npuMLIRRuntimeSetMemRef =
                reinterpret_cast<npu_mlir_runtime_set_mem_ref_t*>(ov::util::get_symbol(lib, "npuMLIRRuntimeSetMemRef"));
        npuMLIRRuntimeParseMemRef = reinterpret_cast<npu_mlir_runtime_parse_mem_ref_t*>(
                ov::util::get_symbol(lib, "npuMLIRRuntimeParseMemRef"));
    }
};

using NPUMLIRRuntimeCAPITestParams = std::tuple<std::string,  // Device name
                                                std::string,  // Model path
                                                std::string,  // Library name
                                                ov::AnyMap    // Config
                                                >;

class NPUMLIRRuntimeCAPITest :
        public testing::WithParamInterface<NPUMLIRRuntimeCAPITestParams>,
        public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NPUMLIRRuntimeCAPITestParams>& obj) {
        std::string targetDevice;
        std::string modelPath;
        std::string libName;
        ov::AnyMap configuration;

        std::tie(targetDevice, modelPath, libName, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "model=" << modelPath << "_";
        result << "target_device=" << targetDevice << "_";
        result << "library_name=" << libName << "_";

        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

protected:
    void checkStatus(const std::string& step, const ze_result_t result) {
        if (ZE_RESULT_SUCCESS != result) {
            throw std::runtime_error("L0 " + step + " failed: " + intel_npu::ze_result_to_string(result) + ", code 0x" +
                                     std::to_string(result) + " - " + intel_npu::ze_result_to_description(result));
        }
    }

    void SetUp() override {
        try {
            // Prepare compiled blob
            std::string targetDevice;
            std::string modelPath;
            std::string libName;
            ov::AnyMap configuration;

            std::tie(targetDevice, modelPath, libName, configuration) = this->GetParam();
            if (!std::filesystem::exists(modelPath)) {
                throw std::runtime_error("Cannot open " + modelPath);
            }

            // Prepare symbol
            auto libpath = ov::util::make_plugin_library_name({}, libName);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            lib = ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
            lib = ov::util::load_shared_object(libpath.c_str());
#endif
            // Initialize function table
            fntbl.init(lib);

            ov::Core core;

            auto model = core.read_model(modelPath);
            model->reshape({{1, 16, ov::Dimension(10, 1920), 1920}});
            auto preprocessor = ov::preprocess::PrePostProcessor(model);
            const auto inputs = model->inputs();
            const auto outputs = model->outputs();

            for (size_t i = 0; i < inputs.size(); i++) {
                preprocessor.input(i).tensor().set_element_type(ov::element::f16);
            }
            for (size_t i = 0; i < outputs.size(); i++) {
                preprocessor.output(i).tensor().set_element_type(ov::element::f16);
            }
            auto newModel = preprocessor.build();

            ov::CompiledModel compiledModel;
            compiledModel = core.compile_model(newModel, targetDevice, configuration);
            compiledModel.export_model(blob);
            if (blob.str().size() == 0) {
                GTEST_SKIP() << "Can not export blob";
            }
        } catch (const std::runtime_error& error) {
            GTEST_SKIP() << error.what();
        }
    }

    std::stringstream blob;
    std::shared_ptr<void> lib;
    npu_mlir_runtime_fntbl fntbl;
};

TEST_P(NPUMLIRRuntimeCAPITest, GetAPIVersion_Success) {
    npu_mlir_runtime_version_t version;
    auto result = fntbl.npuMLIRRuntimeGetAPIVersion(&version);
    EXPECT_EQ(result, NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    EXPECT_EQ(version, NPU_MLIR_RUNTIME_VERSION_CURRENT);
}

TEST_P(NPUMLIRRuntimeCAPITest, ExecuteCompiledModel) {
    // Prepare blob descriptor
    std::string content = blob.str();
    size_t size = content.size();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(content.c_str());

    npu_mlir_runtime_blob_desc_t blobDesc;
    blobDesc.pInput = data;
    blobDesc.inputSize = size;

    npu_mlir_runtime_handle_t handle = nullptr;
    npu_mlir_runtime_properties_t props;
    EXPECT_EQ(fntbl.npuMLIRRuntimeCreate(&blobDesc, &handle, &props), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    EXPECT_NE(handle, nullptr);

    // Create ze related structures
    // Init zero structure and inputs
    auto initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();

    // Create input and output buffer
    constexpr std::size_t STANDARD_PAGE_SIZE = 4096;
    std::vector<std::pair<std::shared_ptr<intel_npu::ZeroMem>, npu_mlir_runtime_mem_ref_handle_t>> inputs;
    std::vector<std::pair<std::shared_ptr<intel_npu::ZeroMem>, npu_mlir_runtime_mem_ref_handle_t>> outputs;

    vpux::NetworkMetadata metaData;
    ze_graph_argument_properties_3_t arg;
    ze_graph_argument_metadata_t meta;
    int64_t upperBound[5];
    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        EXPECT_EQ(fntbl.npuMLIRRuntimeGetMetadata(handle, i, &arg, &meta, upperBound), NPU_MLIR_RUNTIME_RESULT_SUCCESS);

        // This is the upper bound, actual size may be smaller
        ov::Shape shapeFromCompiler;
        for (uint32_t d = 0; d < arg.dims_count; d++) {
            shapeFromCompiler.push_back(arg.dims[d]);
        }

        // This shape contains std::numeric_limits<uint64_t>::max() for dynamic dimensions
        // TODO: Use this to determine which dimensions are dynamic when more models are supported
        ov::Shape shapeFromIRModel;
        for (uint32_t d = 0; d < meta.shape_size; d++) {
            shapeFromIRModel.push_back(static_cast<size_t>(meta.shape[d]));
        }

        // Calc the size
        ov::element::Type_t precision = intel_npu::zeroUtils::toOVElementType(arg.devicePrecision);
        ov::element::Type elementSize(precision);
        size_t totalSize = std::accumulate(shapeFromCompiler.begin(), shapeFromCompiler.end(), elementSize.size(),
                                           [](int a, int b) {
                                               return a * b;
                                           });
        std::shared_ptr<intel_npu::ZeroMem> memPtr = intel_npu::ZeroMemPool::get_instance().allocate_zero_memory(
                initStruct, totalSize, STANDARD_PAGE_SIZE, arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT);
        npu_mlir_runtime_mem_ref_handle_t hMemRef;
        void* pData = memPtr->data();
        std::vector<int64_t> tensorSize(shapeFromCompiler.begin(), shapeFromCompiler.end());
        std::vector<int64_t> tensorStride;
        tensorStride.resize(tensorSize.size());
        int64_t stride = 1;
        for (int64_t i = tensorSize.size() - 1; i >= 0; i--) {
            tensorStride[i] = stride;
            stride *= tensorSize[i];
        }
        EXPECT_EQ(fntbl.npuMLIRRuntimeCreateMemRef(tensorSize.size(), &hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        EXPECT_EQ(fntbl.npuMLIRRuntimeSetMemRef(hMemRef, pData, pData, 0, tensorSize.data(), tensorStride.data(),
                                                tensorSize.size()),
                  NPU_MLIR_RUNTIME_RESULT_SUCCESS);

        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            inputs.emplace_back(memPtr, hMemRef);
        } else if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            outputs.emplace_back(memPtr, hMemRef);
        }

        {
            // Test the function to parse memref handle info
            const void* parsedBaseData = nullptr;
            const void* parsedData = nullptr;
            int64_t parsedOffset = 0;
            std::vector<int64_t> parsedSize;
            parsedSize.resize(tensorSize.size());
            std::vector<int64_t> parsedStride;
            parsedStride.resize(tensorStride.size());
            int64_t parsedDimsCount;
            EXPECT_EQ(fntbl.npuMLIRRuntimeParseMemRef(hMemRef, &parsedBaseData, &parsedData, &parsedOffset,
                                                      parsedSize.data(), parsedStride.data(), &parsedDimsCount),
                      NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        }
    }

    uint32_t groupOrdinal = intel_npu::zeroUtils::findCommandQueueGroupOrdinal(
            initStruct->getDevice(), ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    // TODO:
    // Use other MODEL_PRIORITY
    // Use ZE_NPU_COMMAND_QUEUE_OPTION_TURBO to test TURBO mode
    // Use ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC to test device sync mode
    auto commandQueue =
            std::make_shared<intel_npu::CommandQueue>(initStruct, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, groupOrdinal);

    ze_graph_dditable_ext_t* graphDdiTableExt = nullptr;
    checkStatus("zeDriverGetExtensionFunctionAddress",
                intel_npu::zeDriverGetExtensionFunctionAddress(initStruct->getDriver(), ZE_GRAPH_EXT_NAME,
                                                               reinterpret_cast<void**>(&graphDdiTableExt)));

    size_t commandListCount = props.numOfSubGraphs;
    std::vector<std::shared_ptr<intel_npu::CommandList>> commandLists;
    commandLists.reserve(commandListCount);
    for (size_t i = 0; i < commandListCount; ++i) {
        commandLists.emplace_back(std::make_shared<intel_npu::CommandList>(initStruct, groupOrdinal));
    }

    std::shared_ptr<intel_npu::Fence> fence = std::make_shared<intel_npu::Fence>(commandQueue);

    // Prepare input and output buffers
    std::vector<npu_mlir_runtime_mem_ref_handle_t> inputArgs;
    inputArgs.reserve(inputs.size());
    std::vector<npu_mlir_runtime_mem_ref_handle_t> outputArgs;
    outputArgs.reserve(outputs.size());

    for (const auto& memRef : inputs) {
        inputArgs.push_back(const_cast<npu_mlir_runtime_mem_ref_handle_t>(memRef.second));
    }
    for (const auto& memRef : outputs) {
        outputArgs.push_back(const_cast<npu_mlir_runtime_mem_ref_handle_t>(memRef.second));
    }

    // Prepare pointer of commandlist
    std::vector<ze_command_list_handle_t> commandListHandles;
    commandListHandles.reserve(commandLists.size());
    for (const auto& commandList : commandLists) {
        commandListHandles.push_back(commandList->handle());
    }

    // Execute
    npu_mlir_runtime_execute_params_t execParams;
    execParams.numOfInputs = static_cast<uint32_t>(inputArgs.size());
    execParams.pInputs = inputArgs.data();
    execParams.numOfOutputs = static_cast<uint32_t>(outputArgs.size());
    execParams.pOutputs = outputArgs.data();
    execParams.ctx = initStruct->getContext();
    execParams.device = initStruct->getDevice();
    execParams.graphDdiTableExt = graphDdiTableExt;
    execParams.commandLists = commandListHandles.data();
    execParams.numCommandLists = static_cast<uint64_t>(commandListHandles.size());
    execParams.commandQueue = commandQueue->handle();
    execParams.inferenceFence = fence->handle();
    execParams.event = nullptr;

    EXPECT_EQ(fntbl.npuMLIRRuntimeExecute(handle, &execParams), NPU_MLIR_RUNTIME_RESULT_SUCCESS);

    fence->hostSynchronize();

    // Destroy
    EXPECT_EQ(fntbl.npuMLIRRuntimeDestroy(handle), NPU_MLIR_RUNTIME_RESULT_SUCCESS);

    // TODO dump output to check accuracy once more models are supported

    // Destroy handles
    for (const auto& hMemRef : inputArgs) {
        EXPECT_EQ(fntbl.npuMLIRRuntimeDestroyMemRef(hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    }
    for (const auto& hMemRef : outputArgs) {
        EXPECT_EQ(fntbl.npuMLIRRuntimeDestroyMemRef(hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    }
}

TEST_P(NPUMLIRRuntimeCAPITest, PredictShape) {
    // Prepare blob descriptor
    std::string content = blob.str();
    size_t size = content.size();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(content.c_str());

    npu_mlir_runtime_blob_desc_t blobDesc;
    blobDesc.pInput = data;
    blobDesc.inputSize = size;

    npu_mlir_runtime_handle_t handle = nullptr;
    npu_mlir_runtime_properties_t props;
    EXPECT_EQ(fntbl.npuMLIRRuntimeCreate(&blobDesc, &handle, &props), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    EXPECT_NE(handle, nullptr);
    struct LocalMemRef {
        const void* basePtr;
        const void* data;
        int64_t offset;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t dimsCount;
        std::vector<int64_t> dynamicRanks;
    };

    // Get input and output info
    std::vector<LocalMemRef> inputs;
    std::vector<LocalMemRef> outputs;

    vpux::NetworkMetadata metaData;
    ze_graph_argument_properties_3_t arg;
    ze_graph_argument_metadata_t meta;
    int64_t upperBound[5];
    uint64_t dynamicRankValue = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < props.numOfGraphArgs; i++) {
        EXPECT_EQ(fntbl.npuMLIRRuntimeGetMetadata(handle, i, &arg, &meta, upperBound), NPU_MLIR_RUNTIME_RESULT_SUCCESS);

        LocalMemRef localMemRef;
        // prepare initial memref for the tensor
        localMemRef.basePtr = nullptr;
        localMemRef.data = nullptr;
        localMemRef.offset = 0;
        localMemRef.dimsCount = arg.dims_count;
        // This is the upper bound, actual size may be smaller
        for (uint32_t d = 0; d < arg.dims_count; d++) {
            localMemRef.sizes.push_back(arg.dims[d]);
        }
        // This shape contains std::numeric_limits<uint64_t>::max() for dynamic dimensions
        for (uint32_t d = 0; d < meta.shape_size; d++) {
            if (meta.shape[d] == dynamicRankValue) {
                localMemRef.dynamicRanks.push_back(d);
            }
        }
        // Update strides (NCHW kind)
        uint64_t stride = 1;
        localMemRef.strides.resize(localMemRef.dimsCount);
        for (int32_t d = localMemRef.dimsCount - 1; d >= 0; d--) {
            // NCHW layout
            localMemRef.strides[d] = stride;
            stride *= localMemRef.sizes[d];
        }
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            inputs.push_back(localMemRef);
        } else if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            outputs.push_back(localMemRef);
        }
    }

    // Predict shape
    std::vector<npu_mlir_runtime_mem_ref_handle_t> inputArgs;
    std::vector<npu_mlir_runtime_mem_ref_handle_t> outputArgs;
    for (auto& input : inputs) {
        npu_mlir_runtime_mem_ref_handle_t hMemRef;
        EXPECT_EQ(fntbl.npuMLIRRuntimeCreateMemRef(input.dimsCount, &hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        EXPECT_EQ(fntbl.npuMLIRRuntimeSetMemRef(hMemRef, input.basePtr, input.data, input.offset, input.sizes.data(),
                                                input.strides.data(), input.dimsCount),
                  NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        inputArgs.push_back(hMemRef);
    }
    for (auto& output : outputs) {
        npu_mlir_runtime_mem_ref_handle_t hMemRef;
        EXPECT_EQ(fntbl.npuMLIRRuntimeCreateMemRef(output.dimsCount, &hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        EXPECT_EQ(fntbl.npuMLIRRuntimeSetMemRef(hMemRef, output.basePtr, output.data, output.offset,
                                                output.sizes.data(), output.strides.data(), output.dimsCount),
                  NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        outputArgs.push_back(hMemRef);
    }
    npu_mlir_runtime_predict_output_shape_params_t predictParam;
    predictParam.pInputs = inputArgs.data();
    predictParam.numOfInputs = inputArgs.size();
    predictParam.pOutputs = outputArgs.data();
    predictParam.numOfOutputs = outputArgs.size();
    EXPECT_EQ(fntbl.npuMLIRRuntimePredictOutputShape(handle, &predictParam), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    // Verify the output shape
    for (size_t i = 0; i < outputs.size(); i++) {
        LocalMemRef parsedMemRef;
        parsedMemRef.sizes.resize(outputs[i].dimsCount);
        parsedMemRef.strides.resize(outputs[i].dimsCount);
        EXPECT_EQ(fntbl.npuMLIRRuntimeParseMemRef(outputArgs[i], &parsedMemRef.basePtr, &parsedMemRef.data,
                                                  &parsedMemRef.offset, parsedMemRef.sizes.data(),
                                                  parsedMemRef.strides.data(), &parsedMemRef.dimsCount),
                  NPU_MLIR_RUNTIME_RESULT_SUCCESS);
        EXPECT_EQ(outputs[i].dimsCount, parsedMemRef.dimsCount);

        for (int64_t d = 0; d < outputs[i].dimsCount; d++) {
            EXPECT_TRUE(outputs[i].sizes[d] >= parsedMemRef.sizes[d]);
            EXPECT_TRUE(outputs[i].strides[d] >= parsedMemRef.strides[d]);
        }
    }
    EXPECT_EQ(fntbl.npuMLIRRuntimeDestroy(handle), NPU_MLIR_RUNTIME_RESULT_SUCCESS);

    // Destroy handles
    for (const auto hMemRef : inputArgs) {
        EXPECT_EQ(fntbl.npuMLIRRuntimeDestroyMemRef(hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    }
    for (const auto hMemRef : outputArgs) {
        EXPECT_EQ(fntbl.npuMLIRRuntimeDestroyMemRef(hMemRef), NPU_MLIR_RUNTIME_RESULT_SUCCESS);
    }
}

const std::vector<std::string> devices = {"NPU.4000", "NPU.5010"};

const std::vector<std::string> models = {"CustomNet_canonical_strides_1x1_no_fork.xml"};

const std::vector<std::string> libNames = {"npu_mlir_runtime"};

const std::vector<ov::AnyMap> configs = {{{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_COMPILATION_MODE", "HostCompile"}}};

INSTANTIATE_TEST_SUITE_P(smoke_npuMLIRRuntime, NPUMLIRRuntimeCAPITest,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(models),
                                            ::testing::ValuesIn(libNames), ::testing::ValuesIn(configs)),
                         NPUMLIRRuntimeCAPITest::getTestCaseName);
