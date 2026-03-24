#pragma once

#ifdef DALOTIA_WITH_CUDA

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace dalotia {

// Returns true if `ptr` is a CUDA device pointer (cudaMalloc'd or managed).
// Returns false for host pointers (including cudaMallocHost pinned memory).
bool is_device_pointer(const void* ptr) noexcept;

// RAII wrapper for a cudaMalloc'd device buffer. Move-only.
// Type-erased (stores void*); use as<T>() for typed access.
class CudaBuffer {
   public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t nbytes) : size_(nbytes) {
        if (nbytes > 0) {
            cudaError_t err = cudaMalloc(&ptr_, nbytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CudaBuffer: cudaMalloc failed: ") +
                    cudaGetErrorString(err));
            }
        }
    }

    CudaBuffer(size_t nbytes, cudaStream_t stream) : size_(nbytes) {
        if (nbytes > 0) {
            cudaError_t err = cudaMallocAsync(&ptr_, nbytes, stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CudaBuffer: cudaMallocAsync failed: ") +
                    cudaGetErrorString(err));
            }
            stream_ = stream;
            async_ = true;
        }
    }

    ~CudaBuffer() {
        if (ptr_) {
            if (async_) {
                cudaFreeAsync(ptr_, stream_);
            } else {
                cudaFree(ptr_);
            }
        }
    }

    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
        , stream_(other.stream_)
        , async_(other.async_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                if (async_)
                    cudaFreeAsync(ptr_, stream_);
                else
                    cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            stream_ = other.stream_;
            async_ = other.async_;
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
        return static_cast<T*>(ptr_);
    }

    template <typename T>
    const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

   private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    cudaStream_t stream_ = 0;
    bool async_ = false;
};

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUDA
