// Combined test: load a permuted tensor into buffers backed by different
// dalotia CUDA memory resources (device + pinned host) and verify that the
// device-side bytes (after copy-back) match the pinned host-side bytes.

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "dalotia.hpp"
#include "dalotia_cuda.hpp"
#include "dalotia_cuda_memory_resource.hpp"
#include "dalotia_safetensors_file.hpp"

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl;  \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

static const char* TEST_FILE = "../data/model.safetensors";
// Shape [4,3,5]; permutation [1,0,2] yields shape [3,4,5] with values 0..59.
static const char* PERM_TENSOR = "embedding_firstchanged";
static constexpr int NUM_ELEMENTS = 3 * 4 * 5;
static constexpr dalotia_WeightFormat FORMAT = dalotia_float_64;

int main() {
    std::cout << "test_permuted_load_device_vs_pinned... " << std::flush;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP (no CUDA device)" << std::endl;
        return 0;
    }

    const std::vector<int> perm = {1, 0, 2};
    const size_t nbytes = NUM_ELEMENTS * sizeof(double);

    // Non-default stream: the GPU permute kernel and its temp-buffer
    // alloc/free should be ordered on this stream, not the default one.
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    dalotia::SafetensorsFile file(TEST_FILE);

    // 1) Permuted load into a device-resident buffer (GPU permute kernel path).
    //    Pass `stream` so load + permute kernel run on it.
    dalotia::CudaBuffer d_buf(nbytes, dalotia::cuda_device_resource());
    file.load_tensor_dense(PERM_TENSOR, FORMAT, dalotia_C_ordering,
                           d_buf.as<dalotia_byte>(), perm, stream);

    // 2) Permuted load into a pinned host buffer (host permute path —
    //    pinned memory is host-accessible and not detected as a device ptr).
    //    The stream argument is ignored on the host path.
    dalotia::CudaBuffer h_pinned(nbytes, dalotia::cuda_pinned_resource());
    file.load_tensor_dense(PERM_TENSOR, FORMAT, dalotia_C_ordering,
                           h_pinned.as<dalotia_byte>(), perm, stream);

    // Sanity: pinned pointer must be host-addressable, not a device pointer.
    assert(!dalotia::is_device_pointer(h_pinned.data()));
    assert(dalotia::is_device_pointer(d_buf.data()));

    // Wait for all stream work before reading the device buffer back.
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy device buffer back to host (also on the same stream) and compare.
    std::vector<dalotia_byte> d_copy(nbytes);
    CHECK_CUDA(cudaMemcpyAsync(d_copy.data(), d_buf.data(), nbytes,
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    assert(std::memcmp(d_copy.data(), h_pinned.data(), nbytes) == 0);

    // Also verify the values are the expected 0..59 sequence.
    const double* pinned_vals = h_pinned.as<double>();
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        assert(pinned_vals[i] == static_cast<double>(i));
    }

    CHECK_CUDA(cudaStreamDestroy(stream));

    std::cout << "OK" << std::endl;
    return 0;
}
