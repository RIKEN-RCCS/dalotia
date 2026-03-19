#ifdef DALOTIA_WITH_CUFILE

#include "dalotia_cufile.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cufile.h>

namespace dalotia {
CuFileDriver::CuFileDriver() {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cuFileDriverOpen failed with error " +
                                 std::to_string(status.err));
    }
}

CuFileDriver::~CuFileDriver() {
    cuFileDriverClose();
}

bool CuFileDriver::is_open() noexcept {
    return cuFileUseCount() > 0;
}

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

#endif  // DALOTIA_WITH_CUFILE
