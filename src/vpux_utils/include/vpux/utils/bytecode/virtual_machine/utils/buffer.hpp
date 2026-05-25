//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vpux::bytecode {

using BufferHandle = uint64_t;

enum class Permission : uint8_t {
    Read = 0,
    ReadWrite = 1,
};

enum class Ownership : uint8_t {
    Owned = 0,
    Unowned = 1,
};

class Buffer {
    uint8_t* _data;
    size_t _size;
    Permission _permission;
    Ownership _ownership;

    void destroy();
    void reset();

public:
    // Create a buffer of the given size, with the given permission. The buffer owns the allocated memory
    // Throws std::bad_alloc if it fails to create the buffer
    Buffer(size_t size, Permission permission);

    // Create a buffer from external memory. The buffer does not own the memory
    Buffer(uint8_t* data, size_t size, Permission permission);

    // Disable copy operations to prevent unintended shallow copies
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Enable move operations for efficient resource transfer
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    // Delete the buffer and deallocate memory if owned
    ~Buffer();

    const uint8_t* getData() const;

    size_t getSize() const;

    Permission getPermission() const;

    Ownership getOwnership() const;

    void writeData(size_t offset, const uint8_t* data, size_t size);
};

class BufferManager {
    class BufferHandleManager {
        BufferHandle _lastHandle = 0;

    public:
        BufferHandle getNextHandle();
    };

    size_t _maximumMemorySize{};
    size_t _currentMemorySize{};
    BufferHandleManager _handleManager{};
    std::unordered_map<BufferHandle, Buffer> _buffers;
    std::unordered_map<BufferHandle, std::vector<BufferHandle>> _derivedBuffers;

    BufferHandle generateBufferHandle();

public:
    // Create a buffer manager, which manages the creation, deletion and access for buffers
    // The buffer manager keeps track of the total allocated memory size, and does not allow creation of new buffers if
    // the total size exceeds the given maximum memory size
    BufferManager(size_t maximumMemorySize);

    // Allocates a buffer of the given size. Returns a handle to the buffer
    // Throws std::invalid_argument if the requested size is zero
    // Throws std::bad_alloc if the requested size exceeds the maximum memory size
    BufferHandle create(size_t size, Permission permission = Permission::ReadWrite);

    // Create a buffer from external memory. Returns a handle to the created buffer
    // This buffer does not contribute to the memory size tracked by the buffer manager, as the memory
    // is external
    // Throws std::invalid_argument if the external data pointer is null or if the size is zero
    BufferHandle createFromMemory(uint8_t* externalData, size_t size, Permission permission);

    // Create a buffer from another buffer that references a subset of the original buffer's memory. Returns a handle to
    // the new buffer. The new buffer does not own the memory, and its permission is the same as the original buffer
    // Throws std::invalid_argument if the original handle is invalid
    // Throws std::out_of_range if the offset and size exceed the bounds of the original buffer
    BufferHandle createFromBuffer(BufferHandle handle, size_t offset, size_t size);

    // Deletes the buffer associated with the given handle. Does not deallocate memory for unowned buffers.
    // If the buffer has derived buffers created via createFromBuffer, those are recursively deleted first.
    // Throws std::invalid_argument if the handle is invalid
    void deleteBuffer(BufferHandle handle);

    // Returns true if a buffer with the given handle exists, false otherwise
    bool exists(BufferHandle handle) const;

    // Returns a reference to the buffer associated with the given handle
    // Throws std::invalid_argument if the handle is invalid
    Buffer& getBuffer(BufferHandle handle);
};

}  // namespace vpux::bytecode
