#include "dalotia_tensor_file.hpp"

#ifdef DALOTIA_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef DALOTIA_WITH_CUFILE
#include "dalotia_cufile.hpp"
#endif

namespace dalotia {

#ifdef DALOTIA_WITH_CUDA
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
#endif  // DALOTIA_WITH_CUDA

void TensorFile::load_tensor_dense(const std::string& tensor_name,
                                   dalotia_WeightFormat weightFormat,
                                   dalotia_Ordering ordering,
                                   dalotia_byte* __restrict__ tensor,
                                   const std::vector<int>& permutation) {
#ifdef DALOTIA_WITH_CUDA
    if (is_device_pointer(tensor)) {
        if (!permutation.empty()) {
            throw std::runtime_error(
                "load_tensor_dense: permutation to device memory is not "
                "supported; transpose on-device after loading instead.");
        }

        auto extents = this->get_tensor_extents(tensor_name);
        auto total_elements =
            std::accumulate(extents.begin(), extents.end(), size_t{1},
                            std::multiplies<size_t>());
        size_t nbytes = total_elements * sizeof_weight_format(weightFormat);

#ifdef DALOTIA_WITH_CUFILE
        if (gpu_data_source_ && data_source_ && data_source_->host_data(0)) {
            // GDS path: read directly from file to device memory.
            auto ptrs = this->get_mmap_tensor_pointers(tensor_name);
            if (!ptrs.empty()) {
                size_t offset = reinterpret_cast<const uint8_t*>(ptrs[0]) -
                                data_source_->host_data(0);
                gpu_data_source_->read_into(offset, nbytes, tensor);
                return;
            }
        }
#endif  // DALOTIA_WITH_CUFILE

        // Fallback: load to a temporary host buffer, then cudaMemcpy.
        std::vector<dalotia_byte> host_buf(nbytes);
        load_tensor_dense_impl(tensor_name, weightFormat, ordering,
                               host_buf.data(), {});
        cudaError_t err =
            cudaMemcpy(tensor, host_buf.data(), nbytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("load_tensor_dense: cudaMemcpy failed: ") +
                cudaGetErrorString(err));
        }
        return;
    }
#endif  // DALOTIA_WITH_CUDA

    // Host path — delegate to the format-specific implementation.
    load_tensor_dense_impl(tensor_name, weightFormat, ordering, tensor,
                           permutation);
}

}  // namespace dalotia
