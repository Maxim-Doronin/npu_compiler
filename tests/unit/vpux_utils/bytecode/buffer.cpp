//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/virtual_machine/utils/buffer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

using namespace vpux;

// ============================================================
// Buffer Tests
// ============================================================

class BufferTest : public ::testing::Test {};

TEST_F(BufferTest, ConstructNewAllocation) {
    {
        // 64 bytes + Read-Write
        bytecode::Buffer buf(/*size=*/64, bytecode::Permission::ReadWrite);
        EXPECT_NE(buf.getData(), nullptr);
        EXPECT_EQ(buf.getSize(), 64u);
        EXPECT_EQ(buf.getPermission(), bytecode::Permission::ReadWrite);
        EXPECT_EQ(buf.getOwnership(), bytecode::Ownership::Owned);
    }
    {
        // 128 bytes + Read-Only
        bytecode::Buffer buf(/*size=*/128, bytecode::Permission::Read);
        EXPECT_NE(buf.getData(), nullptr);
        EXPECT_EQ(buf.getSize(), 128u);
        EXPECT_EQ(buf.getPermission(), bytecode::Permission::Read);
        EXPECT_EQ(buf.getOwnership(), bytecode::Ownership::Owned);
    }
}

TEST_F(BufferTest, ConstructFromExternalMemory) {
    std::array<uint8_t, 8> externalData = {1, 2, 3, 4, 5, 6, 7, 8};
    bytecode::Buffer buf(externalData.data(), externalData.size(), bytecode::Permission::Read);
    EXPECT_EQ(buf.getData(), externalData.data());
    EXPECT_EQ(buf.getSize(), externalData.size());
    EXPECT_EQ(buf.getPermission(), bytecode::Permission::Read);
    EXPECT_EQ(buf.getOwnership(), bytecode::Ownership::Unowned);
}

TEST_F(BufferTest, DataWrite) {
    std::array<uint8_t, 4> data = {0xDE, 0xAD, 0xBE, 0xEF};
    {
        // Read-Write buffer
        bytecode::Buffer buf(4, bytecode::Permission::ReadWrite);
        buf.writeData(/*offset=*/0, data.data(), data.size());
        EXPECT_EQ(buf.getData()[0], 0xDE);
        EXPECT_EQ(buf.getData()[1], 0xAD);
        EXPECT_EQ(buf.getData()[2], 0xBE);
        EXPECT_EQ(buf.getData()[3], 0xEF);
    }
    {
        // Read-only buffer
        bytecode::Buffer buf(4, bytecode::Permission::Read);
        EXPECT_THROW(buf.writeData(/*offset=*/0, data.data(), data.size()), std::runtime_error);
    }
    {
        // Out-of-bounds write
        bytecode::Buffer buf(4, bytecode::Permission::ReadWrite);
        EXPECT_THROW(buf.writeData(/*offset=*/2, data.data(), data.size()), std::out_of_range);
    }
}

// ============================================================
// BufferManager Tests
// ============================================================

class BufferManagerTest : public ::testing::Test {
protected:
    static constexpr size_t DEFAULT_MAX_MEMORY = 1024;
    bytecode::BufferManager manager{DEFAULT_MAX_MEMORY};
};

TEST_F(BufferManagerTest, Create) {
    auto handle = manager.create(/*size=*/64);
    ASSERT_TRUE(manager.exists(handle));
    ASSERT_NO_THROW(manager.getBuffer(handle));
    auto& buf = manager.getBuffer(handle);
    EXPECT_NE(buf.getData(), nullptr);
    EXPECT_EQ(buf.getSize(), 64u);
    EXPECT_EQ(buf.getPermission(), bytecode::Permission::ReadWrite);
    EXPECT_EQ(buf.getOwnership(), bytecode::Ownership::Owned);
}

TEST_F(BufferManagerTest, CreateMultipleBuffers) {
    auto handle1 = manager.create(/*size=*/64);
    auto handle2 = manager.create(/*size=*/128);
    ASSERT_TRUE(manager.exists(handle1));
    ASSERT_TRUE(manager.exists(handle2));
    EXPECT_NE(handle1, handle2);
}

TEST_F(BufferManagerTest, CreateZeroSizeThrows) {
    EXPECT_THROW(manager.create(0), std::invalid_argument);
}

TEST_F(BufferManagerTest, CreateExactlyMaximumSizeSucceeds) {
    EXPECT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
}

TEST_F(BufferManagerTest, CreateExceedingMaximumSizeThrows) {
    EXPECT_THROW(manager.create(DEFAULT_MAX_MEMORY + 1), std::bad_alloc);
}

TEST_F(BufferManagerTest, CreateCumulativeExceedingMaximumSizeThrows) {
    manager.create(DEFAULT_MAX_MEMORY / 2);
    EXPECT_THROW(manager.create(DEFAULT_MAX_MEMORY / 2 + 1), std::bad_alloc);
}

TEST_F(BufferManagerTest, CreateAfterDeleteFitsWithinMaximumSize) {
    auto handle = manager.create(DEFAULT_MAX_MEMORY);
    manager.deleteBuffer(handle);
    EXPECT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
}

TEST_F(BufferManagerTest, CreateFromMemory) {
    std::array<uint8_t, 4> data = {0xDE, 0xAD, 0xBE, 0xEF};
    const auto handle = manager.createFromMemory(data.data(), data.size(), bytecode::Permission::Read);
    EXPECT_TRUE(manager.exists(handle));
    ASSERT_NO_THROW(manager.getBuffer(handle));
    const auto& buf = manager.getBuffer(handle);
    EXPECT_EQ(buf.getData(), data.data());
    EXPECT_EQ(buf.getSize(), data.size());
    EXPECT_EQ(buf.getPermission(), bytecode::Permission::Read);
    EXPECT_EQ(buf.getOwnership(), bytecode::Ownership::Unowned);
}

TEST_F(BufferManagerTest, CreateFromMemoryNullDataThrows) {
    EXPECT_THROW(manager.createFromMemory(nullptr, /*size=*/4, bytecode::Permission::Read), std::invalid_argument);
}

TEST_F(BufferManagerTest, CreateFromMemoryZeroSizeThrows) {
    std::array<uint8_t, 4> data = {};
    EXPECT_THROW(manager.createFromMemory(data.data(), 0, bytecode::Permission::Read), std::invalid_argument);
}

TEST_F(BufferManagerTest, CreateFromMemoryDoesNotContributeToMemoryBudget) {
    // Fill budget with external memory, then still create an owned buffer
    std::array<uint8_t, DEFAULT_MAX_MEMORY * 2> externalData = {};
    manager.createFromMemory(externalData.data(), externalData.size(), bytecode::Permission::Read);
    EXPECT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
}

TEST_F(BufferManagerTest, CreateFromBuffer) {
    auto handle = manager.create(/*size=*/64, bytecode::Permission::ReadWrite);
    ASSERT_TRUE(manager.exists(handle));
    const auto& buf = manager.getBuffer(handle);
    EXPECT_NE(buf.getData(), nullptr);
    EXPECT_EQ(buf.getSize(), 64u);
    EXPECT_EQ(buf.getPermission(), bytecode::Permission::ReadWrite);
    EXPECT_EQ(buf.getOwnership(), bytecode::Ownership::Owned);

    auto newHandle = manager.createFromBuffer(handle, /*offset=*/32, /*size=*/32);
    ASSERT_TRUE(manager.exists(newHandle));
    const auto& newBuf = manager.getBuffer(newHandle);
    EXPECT_EQ(newBuf.getData(), buf.getData() + 32);
    EXPECT_EQ(newBuf.getSize(), 32u);
    EXPECT_EQ(newBuf.getPermission(), bytecode::Permission::ReadWrite);
    EXPECT_EQ(newBuf.getOwnership(), bytecode::Ownership::Unowned);
}

TEST_F(BufferManagerTest, CreateFromBufferInvalidHandleThrows) {
    EXPECT_THROW(manager.createFromBuffer(9999, /*offset=*/0, /*size=*/1), std::invalid_argument);
}

TEST_F(BufferManagerTest, CreateFromBufferOutOfBoundsThrows) {
    auto handle = manager.create(/*size=*/64);
    ASSERT_TRUE(manager.exists(handle));

    EXPECT_THROW(manager.createFromBuffer(handle, /*offset=*/32, /*size=*/33), std::out_of_range);
    EXPECT_THROW(manager.createFromBuffer(handle, /*offset=*/64, /*size=*/1), std::out_of_range);
}

TEST_F(BufferManagerTest, MultipleExternalBuffersHaveUniqueHandles) {
    std::array<uint8_t, 4> data = {};
    const auto h1 = manager.createFromMemory(data.data(), data.size(), bytecode::Permission::Read);
    const auto h2 = manager.createFromMemory(data.data() + 2, data.size(), bytecode::Permission::Read);
    EXPECT_NE(h1, h2);
}

TEST_F(BufferManagerTest, DeleteBuffer) {
    std::array<uint8_t, 4> data = {};
    const auto handle = manager.createFromMemory(data.data(), data.size(), bytecode::Permission::Read);
    manager.deleteBuffer(handle);
    EXPECT_FALSE(manager.exists(handle));
}

TEST_F(BufferManagerTest, DeleteBufferInvalidHandleThrows) {
    EXPECT_THROW(manager.deleteBuffer(9999), std::invalid_argument);
}

TEST_F(BufferManagerTest, DeleteSameHandleTwiceThrows) {
    const auto handle = manager.create(/*size=*/4);
    manager.deleteBuffer(handle);
    EXPECT_THROW(manager.deleteBuffer(handle), std::invalid_argument);
}

TEST_F(BufferManagerTest, DeleteBufferDeletesDerivedBuffer) {
    const auto parentHandle = manager.create(/*size=*/64);
    const auto derivedHandle = manager.createFromBuffer(parentHandle, /*offset=*/0, /*size=*/32);
    ASSERT_TRUE(manager.exists(derivedHandle));

    manager.deleteBuffer(parentHandle);

    EXPECT_FALSE(manager.exists(parentHandle));
    EXPECT_FALSE(manager.exists(derivedHandle));
}

TEST_F(BufferManagerTest, DeleteDerivedBufferDoesNotDeleteParent) {
    const auto parentHandle = manager.create(/*size=*/64);
    const auto derivedHandle = manager.createFromBuffer(parentHandle, /*offset=*/0, /*size=*/32);

    manager.deleteBuffer(derivedHandle);

    EXPECT_FALSE(manager.exists(derivedHandle));
    EXPECT_TRUE(manager.exists(parentHandle));
    EXPECT_NO_THROW(manager.deleteBuffer(parentHandle));
}

TEST_F(BufferManagerTest, DeleteBufferCascadesAcrossChain) {
    const auto handleA = manager.create(/*size=*/64);
    const auto handleB = manager.createFromBuffer(handleA, /*offset=*/0, /*size=*/64);
    const auto handleC = manager.createFromBuffer(handleB, /*offset=*/0, /*size=*/32);

    manager.deleteBuffer(handleA);

    EXPECT_FALSE(manager.exists(handleA));
    EXPECT_FALSE(manager.exists(handleB));
    EXPECT_FALSE(manager.exists(handleC));
}

TEST_F(BufferManagerTest, DeleteDerivedBufferThenDeleteParentDoesNotThrow) {
    const auto parentHandle = manager.create(/*size=*/64);
    const auto derivedHandle = manager.createFromBuffer(parentHandle, /*offset=*/0, /*size=*/32);

    manager.deleteBuffer(derivedHandle);
    EXPECT_NO_THROW(manager.deleteBuffer(parentHandle));

    EXPECT_FALSE(manager.exists(parentHandle));
    EXPECT_FALSE(manager.exists(derivedHandle));
}

// ============================================================
// Stress Tests
// ============================================================

TEST(BufferManagerStressTest, CreateAndDeleteManyBuffers) {
    constexpr size_t count = 10'000;
    constexpr size_t bufferSize = 1;
    bytecode::BufferManager manager(count * bufferSize);

    for (size_t i = 0; i < count; ++i) {
        const auto handle = manager.create(bufferSize);
        ASSERT_TRUE(manager.exists(handle));
        manager.deleteBuffer(handle);
        ASSERT_FALSE(manager.exists(handle));
    }
}

TEST(BufferManagerStressTest, CreateAndDeleteManyExternalBuffers) {
    constexpr size_t iterations = 10'000;
    bytecode::BufferManager manager(0);  // No owned memory limit needed
    std::array<uint8_t, 1> dummy = {};

    for (size_t i = 0; i < iterations; ++i) {
        const auto handle = manager.createFromMemory(dummy.data(), dummy.size(), bytecode::Permission::Read);
        ASSERT_TRUE(manager.exists(handle));
        manager.deleteBuffer(handle);
        ASSERT_FALSE(manager.exists(handle));
    }
}

TEST(BufferManagerStressTest, CreateManyExternalBuffersThenDeleteAll) {
    constexpr size_t count = 10'000;
    bytecode::BufferManager manager(0);
    std::array<uint8_t, 1> dummy = {};

    std::vector<bytecode::BufferHandle> handles;
    handles.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        handles.push_back(manager.createFromMemory(dummy.data(), dummy.size(), bytecode::Permission::Read));
    }
    for (const auto& handle : handles) {
        ASSERT_TRUE(manager.exists(handle));
        manager.deleteBuffer(handle);
        ASSERT_FALSE(manager.exists(handle));
    }
}
