#include "dalotia_tensor_file.hpp"

#ifdef DALOTIA_WITH_CUDA
#include <cuda_runtime.h>
#include "dalotia_permute_gpu.cuh"
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
                                   const std::vector<int>& permutation
#ifdef DALOTIA_WITH_CUDA
                                   ,
                                   cudaStream_t stream
#endif
) {
    auto final_perm = final_c_permutation_from_permutation_and_order(
        permutation, ordering, this->get_num_dimensions(tensor_name));

#ifdef DALOTIA_WITH_CUDA
    if (is_device_pointer(tensor)) {
        auto info = this->get_tensor_info(tensor_name);
        if (info.format != weightFormat) {
            throw std::runtime_error(
                "load_tensor_dense: format conversion to device memory is "
                "not yet supported (file format " +
                std::to_string(info.format) + " != requested " +
                std::to_string(weightFormat) +
                "); convert on-device after loading instead.");
        }

        bool needs_permute = !final_perm.empty();

        auto input_extents = this->get_tensor_extents(tensor_name);
        auto total_elements =
            std::accumulate(input_extents.begin(), input_extents.end(),
                            size_t{1}, std::multiplies<size_t>());
        size_t element_bytes = sizeof_weight_format(weightFormat);
        size_t nbytes = total_elements * element_bytes;

        dalotia_byte* d_raw = tensor;
        dalotia_byte* d_tmp = nullptr;
        if (needs_permute) {
            cudaError_t alloc_err = cudaMallocAsync(&d_tmp, nbytes, stream);
            if (alloc_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("load_tensor_dense: cudaMallocAsync for "
                                "permutation temp buffer failed: ") +
                    cudaGetErrorString(alloc_err));
            }
            d_raw = d_tmp;
        }

        bool loaded = false;
#ifdef DALOTIA_WITH_CUFILE
        if (gpu_data_source_ && data_source_ && data_source_->host_data(0)) {
            auto ptrs = this->get_mmap_tensor_pointers(tensor_name);
            if (!ptrs.empty()) {
                size_t offset = reinterpret_cast<const uint8_t*>(ptrs[0]) -
                                data_source_->host_data(0);
                gpu_data_source_->read_into(offset, nbytes, d_raw);
                loaded = true;
            }
        }
#endif  // DALOTIA_WITH_CUFILE
        if (!loaded) {
            // Fallback: load to host buffer, then cudaMemcpyAsync.
            std::vector<dalotia_byte> host_buf(nbytes);
            load_tensor_dense_impl(tensor_name, weightFormat,
                                   dalotia_C_ordering, host_buf.data(), {});
            cudaError_t err = cudaMemcpyAsync(d_raw, host_buf.data(), nbytes,
                                              cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                if (d_tmp)
                    cudaFreeAsync(d_tmp, stream);
                throw std::runtime_error(
                    std::string("load_tensor_dense: cudaMemcpy failed: ") +
                    cudaGetErrorString(err));
            }
            // Must synchronize before host_buf goes out of scope,
            // since cudaMemcpyAsync reads from it asynchronously.
            cudaStreamSynchronize(stream);
        }

        if (needs_permute) {
            permute_on_gpu(d_tmp, tensor, total_elements, element_bytes,
                           static_cast<int>(input_extents.size()),
                           input_extents, final_perm, stream);
            // Synchronize before freeing the temp buffer.
            cudaStreamSynchronize(stream);
            cudaFreeAsync(d_tmp, stream);
        }
        return;
    }
#endif  // DALOTIA_WITH_CUDA

    // Host path
    load_tensor_dense_impl(tensor_name, weightFormat, dalotia_C_ordering,
                           tensor, final_perm);
}

void TensorFile::load_tensor_dense_impl(const std::string& tensor_name,
                                        dalotia_WeightFormat weightFormat,
                                        dalotia_Ordering /*ordering*/,
                                        dalotia_byte* __restrict__ tensor,
                                        const std::vector<int>& permutation) {
    auto info = get_tensor_info(tensor_name);
    if (!permutation.empty()) {
        assign_permuted(static_cast<uint8_t>(info.shape.size()), tensor,
                        weightFormat, info.shape.data(), info.data, info.format,
                        permutation.data());
    } else {
        assign_linearly(tensor, weightFormat, info.num_elements, info.data,
                        info.format);
    }
}

}  // namespace dalotia
