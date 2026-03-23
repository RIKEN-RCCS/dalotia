#ifdef DALOTIA_WITH_CUDA

#include "dalotia_permute_gpu.cuh"
#include "dalotia_assignment.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace dalotia {

// Fixed-size stride arrays passed by value as kernel arguments.
// At most 8 dimensions × 8 bytes = 64 bytes per array — well within
// the 4 KB kernel argument limit, and avoids device memory allocation.
struct PermuteStrides {
    size_t input[kMaxPermuteDims];
    size_t permuted[kMaxPermuteDims];
};

__global__ void permute_kernel(const char *__restrict__ src,
                               char *__restrict__ dest, size_t total_elements,
                               size_t element_bytes, int ndims,
                               PermuteStrides strides) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= total_elements) return;

    size_t dest_idx = 0;
    size_t remaining = idx;
    for (int d = 0; d < ndims; ++d) {
        size_t coord = remaining / strides.input[d];
        remaining -= coord * strides.input[d];
        dest_idx += coord * strides.permuted[d];
    }

    const char *src_ptr = src + idx * element_bytes;
    char *dest_ptr = dest + dest_idx * element_bytes;
    for (size_t b = 0; b < element_bytes; ++b) {
        dest_ptr[b] = src_ptr[b];
    }
}

void permute_on_gpu(const void *d_src, void *d_dest, size_t total_elements,
                    size_t element_bytes, int ndims,
                    const std::vector<int> &input_shape,
                    const std::vector<int> &permutation,
                    cudaStream_t stream) {
    if (ndims <= 0 || ndims > kMaxPermuteDims) {
        throw std::runtime_error(
            "permute_on_gpu: unsupported number of dimensions: " +
            std::to_string(ndims));
    }
    if (static_cast<int>(input_shape.size()) != ndims ||
        static_cast<int>(permutation.size()) != ndims) {
        throw std::runtime_error(
            "permute_on_gpu: input_shape/permutation size mismatch");
    }

    auto [input_strides_vec, permuted_strides_vec, total_size] =
        compute_permute_strides(ndims, input_shape.data(), permutation.data());

    // Copy into fixed-size struct for pass-by-value kernel argument.
    PermuteStrides strides{};
    for (int d = 0; d < ndims; ++d) {
        strides.input[d] = input_strides_vec[d];
        strides.permuted[d] = permuted_strides_vec[d];
    }

    // TODO: for the common case of 2D float32 transpose (permutation [1,0]),
    // cublasSgeam with CUBLAS_OP_T is significantly faster than this generic
    // kernel. Would require adding cuBLAS as a dependency of dalotia_cpp.
    constexpr int block_size = 256;
    int grid_size =
        static_cast<int>((total_elements + block_size - 1) / block_size);
    permute_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const char *>(d_src), static_cast<char *>(d_dest),
        total_elements, element_bytes, ndims, strides);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("permute_on_gpu: kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUDA
