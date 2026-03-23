#pragma once

#ifdef DALOTIA_WITH_CUDA

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace dalotia {

static constexpr int kMaxPermuteDims = 8;

// Permute tensor data on device. `d_src` and `d_dest` are device pointers.
// `d_src` contains elements in the original (C-order) layout.
// `d_dest` will contain elements in the permuted layout.
// `input_shape` and `permutation` are host vectors with `ndims` entries.
// `element_bytes` is the size of one element (e.g. 4 for float32).
//
// The kernel is launched on `stream` (default: 0). No synchronization is
// performed — the caller is responsible for synchronizing the stream if
// needed before reading from `d_dest`.
void permute_on_gpu(const void *d_src, void *d_dest, size_t total_elements,
                    size_t element_bytes, int ndims,
                    const std::vector<int> &input_shape,
                    const std::vector<int> &permutation,
                    cudaStream_t stream = 0);

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUDA
