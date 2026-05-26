//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/bytecode/virtual_machine/utils/buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>

using namespace vpux;

void bytecode::Buffer::reset() {
    _data = nullptr;
    _size = 0;
    _ownership = Ownership::Unowned;
}

void bytecode::Buffer::destroy() {
    if (_data != nullptr && _ownership == Ownership::Owned) {
        delete[] _data;
        reset();
    }
}

// Note: The memory allocation will be changed to use the Level Zero allocator
bytecode::Buffer::Buffer(size_t size, Permission permission)
        : _data(new uint8_t[size]), _size(size), _permission(permission), _ownership(Ownership::Owned) {
}

bytecode::Buffer::Buffer(uint8_t* data, size_t size, Permission permission)
        : _data(data), _size(size), _permission(permission), _ownership(Ownership::Unowned) {
}

bytecode::Buffer::Buffer(Buffer&& other) noexcept
        : _data(other._data), _size(other._size), _permission(other._permission), _ownership(other._ownership) {
    other.reset();
}

bytecode::Buffer& bytecode::Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        _data = other._data;
        _size = other._size;
        _permission = other._permission;
        _ownership = other._ownership;

        other.reset();
    }
    return *this;
}

bytecode::Buffer::~Buffer() {
    destroy();
}

const uint8_t* bytecode::Buffer::getData() const {
    return _data;
}

size_t bytecode::Buffer::getSize() const {
    return _size;
}

bytecode::Permission bytecode::Buffer::getPermission() const {
    return _permission;
}

bytecode::Ownership bytecode::Buffer::getOwnership() const {
    return _ownership;
}

void bytecode::Buffer::writeData(size_t offset, const uint8_t* data, size_t size) {
    if (_permission != Permission::ReadWrite) {
        throw std::runtime_error("Buffer does not have write permission");
    }
    if (offset > _size || size > _size - offset) {
        throw std::out_of_range("Write exceeds buffer bounds");
    }
    std::memcpy(_data + offset, data, size);
}

bytecode::BufferHandle bytecode::BufferManager::BufferHandleManager::getNextHandle() {
    return ++_lastHandle;
}

bytecode::BufferHandle bytecode::BufferManager::generateBufferHandle() {
    return _handleManager.getNextHandle();
}

bytecode::BufferManager::BufferManager(size_t maximumMemorySize): _maximumMemorySize(maximumMemorySize) {
}

bytecode::BufferHandle bytecode::BufferManager::create(size_t size, Permission permission) {
    if (size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }
    if (size > _maximumMemorySize || _currentMemorySize > _maximumMemorySize - size) {
        throw std::bad_alloc();
    }

    const auto bufferHandle = generateBufferHandle();
    _buffers.insert(std::make_pair(bufferHandle, Buffer(size, permission)));
    _currentMemorySize += size;
    return bufferHandle;
}

bytecode::BufferHandle bytecode::BufferManager::createFromMemory(uint8_t* externalData, size_t size,
                                                                 Permission permission) {
    if (externalData == nullptr) {
        throw std::invalid_argument("External data pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }
    const auto handle = generateBufferHandle();
    _buffers.insert(std::make_pair(handle, Buffer(externalData, size, permission)));
    return handle;
}

bytecode::BufferHandle bytecode::BufferManager::createFromBuffer(bytecode::BufferHandle handle, size_t offset,
                                                                 size_t size) {
    auto& buffer = this->getBuffer(handle);
    const auto bufferSize = buffer.getSize();
    if (offset > bufferSize || size > bufferSize - offset) {
        throw std::out_of_range("Requested offset and size exceeds buffer bounds");
    }
    const auto newData = const_cast<uint8_t*>(buffer.getData()) + offset;
    const auto newHandle = generateBufferHandle();
    _buffers.insert(std::make_pair(newHandle, Buffer(newData, size, buffer.getPermission())));
    _derivedBuffers[handle].push_back(newHandle);
    return newHandle;
}

void bytecode::BufferManager::deleteBuffer(bytecode::BufferHandle handle) {
    // Recursively delete all buffers derived from this one via createFromBuffer.
    // Children that were already explicitly deleted are skipped.
    auto childrenIt = _derivedBuffers.find(handle);
    if (childrenIt != _derivedBuffers.end()) {
        // Copy the children list before iterating, as each recursive call modifies _derivedBuffers
        const auto children = childrenIt->second;
        for (const auto child : children) {
            if (exists(child)) {
                deleteBuffer(child);
            }
        }
        _derivedBuffers.erase(handle);
    }

    auto& buffer = this->getBuffer(handle);
    if (buffer.getOwnership() == Ownership::Owned) {
        _currentMemorySize -= buffer.getSize();
    }
    _buffers.erase(handle);
}

bool bytecode::BufferManager::exists(bytecode::BufferHandle handle) const {
    return _buffers.find(handle) != _buffers.end();
}

bytecode::Buffer& bytecode::BufferManager::getBuffer(bytecode::BufferHandle handle) {
    auto it = _buffers.find(handle);
    if (it == _buffers.end()) {
        throw std::invalid_argument("Invalid buffer handle");
    }
    return it->second;
}
