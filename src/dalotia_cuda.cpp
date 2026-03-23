#ifdef DALOTIA_WITH_CUDA

#include "dalotia_cuda.hpp"

namespace dalotia {

bool is_device_pointer(const void* ptr) noexcept {
    if (!ptr)
        return false;
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attrs.type == cudaMemoryTypeDevice ||
           attrs.type == cudaMemoryTypeManaged;
}

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUDA
