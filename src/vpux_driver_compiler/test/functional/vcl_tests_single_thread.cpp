//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <functional>
#include <iostream>
#include <type_traits>

class VCLSingleThreadTest : public VCLTestsUtils::VCLTestsCommon {
public:
    /**
     * @brief Call L0 compiler to compile model to blob
     *
     * @param options Build flags of a model
     */
    vcl_result_t run(const std::string& options);
};

vcl_result_t VCLSingleThreadTest::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    /// Default device is 4000, can be updated by test config
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        std::cout << "\n############################################\n\n";
        std::cout << "Current compiler info:\n";
        std::cout << "ID: " << compilerProp.id << std::endl;
        std::cout << "Version: " << compilerProp.version.major << "." << compilerProp.version.minor << std::endl;
        std::cout << "\tSupported opsets: " << compilerProp.supportedOpsets << std::endl;
        std::cout << "\n############################################\n\n";
    }

    vcl_executable_handle_t executable = nullptr;
    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};

    ret = vclExecutableCreate(compiler, exeDesc, &executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, nullptr, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        printErrorInfo("Failed to get blob size! Result: 0x", ret);
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (!blob) {
            std::cerr << "Failed to alloc memory for blob!\n";
            vclExecutableDestroy(executable);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
        if (ret == VCL_RESULT_SUCCESS) {
#ifdef BLOB_DUMP
            const std::string blobName = std::string("output.net");
            std::ofstream bfos(blobName, std::ios::binary);
            if (!bfos.is_open()) {
                std::cerr << "Can not open " << blobName << ", skip dump!\n";
            } else {
                bfos.write(reinterpret_cast<char*>(blob), blobSize);
                if (bfos.fail()) {
                    std::cerr << "Short write to " << blobName << ", the file is invalid!\n";
                }
            }
            bfos.close();
#endif  // BLOB_DUMP
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy executable! Result: 0x", ret);
        ret = vclCompilerDestroy(compiler);
        return ret;
    }
    executable = nullptr;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

TEST_P(VCLSingleThreadTest, compileModel) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

/// The path of config files for tests
const auto cidTool = VCLSingleThreadTest::getCidToolPath();
/// Models and configs for smoke test
const auto smokeIRInfos = VCLSingleThreadTest::readJson2Vec(cidTool + VCLTestsUtils::SMOKE_TEST_CONFIG);
/// Models and configs for normal test
const auto irInfos = VCLSingleThreadTest::readJson2Vec(cidTool + VCLTestsUtils::TEST_CONFIG);
/// Parameters for smoke tests
const auto smokeParams = testing::Combine(testing::ValuesIn(smokeIRInfos));
/// Parameters for normal tests
const auto params = testing::Combine(testing::ValuesIn(irInfos));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLSingleThreadTest, smokeParams,
                         VCLSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLSingleThreadTest, params, VCLSingleThreadTest::getTestCaseName);

template <typename VclAllocT>
class VCLAllocatorSingleThreadTestBase : public VCLTestsUtils::VCLTestsCommon {
public:
    /**
     * @brief Call L0 compiler to compile model to blob
     *
     * @param options Build flags of a model
     */
    vcl_result_t run(const std::string& options);
};

template <typename VclAllocT>
vcl_result_t VCLAllocatorSingleThreadTestBase<VclAllocT>::run(const std::string& options) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    /// Default device is 4000, can be updated by test config
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_INFO;
    vcl_device_desc_t deviceDesc = {sizeof(vcl_device_desc_t), 0x643e, 3, 5};
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    std::cout << "\n############################################\n\n";
    std::cout << " Current compiler info:\n"
              << " ID: " << compilerProp.id << "\n"
              << " Version: " << compilerProp.version.major << "." << compilerProp.version.minor << "\n"
              << "\tSupported opsets: " << compilerProp.supportedOpsets << "\n";
    std::cout << "\n############################################\n\n";

    uint8_t* blob = nullptr;
    uint64_t size = 0;

    vcl_executable_desc_t exeDesc = {getModelIR().data(), getModelIRSize(), options.c_str(), options.size() + 1};
    VclAllocT allocator;
    std::function<void()> deallocate;

    if constexpr (std::is_same_v<VclAllocT, vcl_allocator2_t>) {
        allocator.allocate = VCLTestsUtils::allocateBlob2;
        allocator.deallocate = VCLTestsUtils::deallocateBlob2;
        deallocate = [&] {
            allocator.deallocate(&allocator, blob);
        };
        ret = vclAllocatedExecutableCreate2(compiler, exeDesc, &allocator, &blob, &size);
    } else {
        static_assert(std::is_same_v<VclAllocT, vcl_allocator_t>);
        allocator.allocate = VCLTestsUtils::allocateBlob;
        allocator.deallocate = VCLTestsUtils::deallocateBlob;
        deallocate = [&] {
            allocator.deallocate(blob);
        };
        ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);
    }

    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || size == 0) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }

#ifdef BLOB_DUMP
    auto ir = GetParam();
    auto netInfo = std::get<0>(ir);
    const std::string blobName = "ct0_" + netInfo.at("network") + ".net.allocator";
    std::ofstream bfos(blobName, std::ios::binary);
    if (!bfos.is_open()) {
        std::cerr << "Cannot open " << blobName << ", skip dump!" << std::endl;
    } else {
        bfos.write(reinterpret_cast<char*>(blob), size);
        if (bfos.fail()) {
            std::cerr << "Short write to " << blobName << ", the file is invalid!" << std::endl;
        }
    }
    bfos.close();
#endif  // BLOB_DUMP

    deallocate();

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return ret;
    }
    return ret;
}

struct VCLAllocatorSingleThreadTest : public VCLAllocatorSingleThreadTestBase<vcl_allocator_t> {};
struct VCLAllocator2SingleThreadTest : public VCLAllocatorSingleThreadTestBase<vcl_allocator2_t> {};

TEST_P(VCLAllocatorSingleThreadTest, compileModel) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

TEST_P(VCLAllocator2SingleThreadTest, compileModel) {
    EXPECT_EQ(run(getNetOptions()), VCL_RESULT_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocatorSingleThreadTest, smokeParams,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SingleThreadCompilation, VCLAllocator2SingleThreadTest, smokeParams,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocatorSingleThreadTest, params,
                         VCLAllocatorSingleThreadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SingleThreadCompilation, VCLAllocator2SingleThreadTest, params,
                         VCLAllocatorSingleThreadTest::getTestCaseName);
