#pragma once

#ifdef DALOTIA_WITH_CUDA

#include <cstddef>
#include <cuda_runtime.h>
#include <memory_resource>

#include "dalotia_cuda_memory_resource.hpp"

namespace dalotia {

// Returns true if `ptr` is a CUDA device pointer (cudaMalloc'd or managed).
// Returns false for host pointers (including cudaMallocHost pinned memory).
bool is_device_pointer(const void* ptr) noexcept;

// Move-only owning byte buffer backed by a std::pmr::memory_resource.
//
// The default resource is `cuda_device_resource()` (sync cudaMalloc/cudaFree).
class CudaBuffer {
   public:
    CudaBuffer() noexcept = default;

    explicit CudaBuffer(size_t nbytes,
                        std::pmr::memory_resource* mr = cuda_device_resource())
        : mr_(mr), size_(nbytes) {
        if (nbytes > 0) {
            ptr_ = std::pmr::polymorphic_allocator<std::byte>(mr_).allocate(
                nbytes);
        }
    }

    ~CudaBuffer() { reset(); }

    CudaBuffer(CudaBuffer&& other) noexcept
        : mr_(other.mr_), ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            mr_ = other.mr_;
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void* data() noexcept { return ptr_; }
    const void* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

    template <typename T>
    T* as() noexcept {
        return static_cast<T*>(static_cast<void*>(ptr_));
    }

    template <typename T>
    const T* as() const noexcept {
        return static_cast<const T*>(static_cast<const void*>(ptr_));
    }

   private:
    void reset() noexcept {
        if (ptr_) {
            std::pmr::polymorphic_allocator<std::byte>(mr_).deallocate(ptr_,
                                                                       size_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    std::pmr::memory_resource* mr_ = cuda_device_resource();
    std::byte* ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUDA
