#pragma once

#ifdef DALOTIA_WITH_CUDA
#ifdef DALOTIA_WITH_CPP_PMR

#include <cuda_runtime.h>
#include <memory_resource>
#include <stdexcept>
#include <string>

namespace dalotia {

namespace detail {

[[noreturn]] inline void throw_cuda(const char* context, cudaError_t err) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
}

}  // namespace detail

//TODO if there is a good library to take these from, we should consider it

class cuda_device_memory_resource : public std::pmr::memory_resource {
   protected:
    void* do_allocate(size_t bytes, size_t /*alignment*/) override {
        void* p = nullptr;
        if (bytes > 0) {
            cudaError_t err = cudaMalloc(&p, bytes);
            if (err != cudaSuccess)
                detail::throw_cuda("cuda_device_memory_resource::allocate",
                                   err);
        }
        return p;
    }

    void do_deallocate(void* p, size_t /*bytes*/,
                       size_t /*alignment*/) override {
        if (p)
            cudaFree(p);
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return dynamic_cast<const cuda_device_memory_resource*>(&other) !=
               nullptr;
    }
};

class cuda_pinned_memory_resource : public std::pmr::memory_resource {
   protected:
    void* do_allocate(size_t bytes, size_t /*alignment*/) override {
        void* p = nullptr;
        if (bytes > 0) {
            cudaError_t err = cudaMallocHost(&p, bytes);
            if (err != cudaSuccess)
                detail::throw_cuda("cuda_pinned_memory_resource::allocate",
                                   err);
        }
        return p;
    }

    void do_deallocate(void* p, size_t /*bytes*/,
                       size_t /*alignment*/) override {
        if (p)
            cudaFreeHost(p);
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return dynamic_cast<const cuda_pinned_memory_resource*>(&other) !=
               nullptr;
    }
};

class cuda_managed_memory_resource : public std::pmr::memory_resource {
   protected:
    void* do_allocate(size_t bytes, size_t /*alignment*/) override {
        void* p = nullptr;
        if (bytes > 0) {
            cudaError_t err = cudaMallocManaged(&p, bytes);
            if (err != cudaSuccess)
                detail::throw_cuda("cuda_managed_memory_resource::allocate",
                                   err);
        }
        return p;
    }

    void do_deallocate(void* p, size_t /*bytes*/,
                       size_t /*alignment*/) override {
        if (p)
            cudaFree(p);
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return dynamic_cast<const cuda_managed_memory_resource*>(&other) !=
               nullptr;
    }
};

class cuda_async_memory_resource : public std::pmr::memory_resource {
// The stream is fixed at construction; all allocations and deallocations are
// ordered on that stream.  The resource must outlive any container using it,
// and the stream must remain valid for that lifetime.
   public:
    explicit cuda_async_memory_resource(cudaStream_t stream)
        : stream_(stream) {}

    cudaStream_t stream() const noexcept { return stream_; }

   protected:
    void* do_allocate(size_t bytes, size_t /*alignment*/) override {
        void* p = nullptr;
        if (bytes > 0) {
            cudaError_t err = cudaMallocAsync(&p, bytes, stream_);
            if (err != cudaSuccess)
                detail::throw_cuda("cuda_async_memory_resource::allocate", err);
        }
        return p;
    }

    void do_deallocate(void* p, size_t /*bytes*/,
                       size_t /*alignment*/) override {
        if (p)
            cudaFreeAsync(p, stream_);
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        auto* o = dynamic_cast<const cuda_async_memory_resource*>(&other);
        return o != nullptr && o->stream_ == stream_;
    }

   private:
    cudaStream_t stream_;
};

// Thread-safe (C++11 static-local guarantee).  These resources are never
// destroyed, matching the contract of std::pmr::new_delete_resource().
inline cuda_device_memory_resource* cuda_device_resource() noexcept {
    static cuda_device_memory_resource r;
    return &r;
}

inline cuda_pinned_memory_resource* cuda_pinned_resource() noexcept {
    static cuda_pinned_memory_resource r;
    return &r;
}

inline cuda_managed_memory_resource* cuda_managed_resource() noexcept {
    static cuda_managed_memory_resource r;
    return &r;
}

}  // namespace dalotia

#endif  // DALOTIA_WITH_CPP_PMR
#endif  // DALOTIA_WITH_CUDA
