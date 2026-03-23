// Test that load_tensor_dense auto-detects host vs device pointers
// and routes through the correct data source.

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#ifdef DALOTIA_WITH_CUFILE
#include <cufile.h>
#include "dalotia_cufile.hpp"
#endif

#include "dalotia.hpp"
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

// The test model at ../data/model.safetensors contains a tensor named
// "embedding" with shape [3, 4, 5] of float64, values 0..59.
static const char* TEST_FILE = "../data/model.safetensors";
static const char* TENSOR_NAME = "embedding";
static constexpr int NUM_ELEMENTS = 3 * 4 * 5;  // 60
static constexpr dalotia_WeightFormat FORMAT = dalotia_float_64;

#ifdef DALOTIA_WITH_CUFILE
// Try to create a CuFileDriver. Returns nullptr if GDS is unavailable.
static std::unique_ptr<dalotia::CuFileDriver> try_open_driver() {
    try {
        return std::make_unique<dalotia::CuFileDriver>();
    } catch (const std::exception& e) {
        std::cout << "(GDS driver unavailable: " << e.what() << ") ";
        return nullptr;
    }
}
#endif  // DALOTIA_WITH_CUFILE

void test_is_device_pointer() {
    std::cout << "test_is_device_pointer... " << std::flush;

    double host_val = 0.0;
    assert(!dalotia::is_device_pointer(&host_val));
    assert(!dalotia::is_device_pointer(nullptr));

    double* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, sizeof(double)));
    assert(dalotia::is_device_pointer(d_ptr));
    CHECK_CUDA(cudaFree(d_ptr));

    double* m_ptr = nullptr;
    CHECK_CUDA(cudaMallocManaged(&m_ptr, sizeof(double)));
    assert(dalotia::is_device_pointer(m_ptr));
    CHECK_CUDA(cudaFree(m_ptr));

    std::cout << "OK" << std::endl;
}

#ifdef DALOTIA_WITH_CUFILE
void test_external_driver_open() {
    std::cout << "test_external_driver_open... " << std::flush;

    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::cout << "SKIP (cuFileDriverOpen failed)" << std::endl;
        return;
    }

    assert(dalotia::CuFileDriver::is_open());

    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    auto [extents, tensor] = file->load_tensor_dense<double>(
        TENSOR_NAME, FORMAT, dalotia_C_ordering);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(tensor[i] == static_cast<double>(i));
    }

    assert(dalotia::CuFileDriver::is_open());
    cuFileDriverClose();
    assert(!dalotia::CuFileDriver::is_open());

    std::cout << "OK" << std::endl;
}
#endif  // DALOTIA_WITH_CUFILE

void test_host_pointer() {
    std::cout << "test_host_pointer... " << std::flush;
    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    auto [extents, tensor] = file->load_tensor_dense<double>(
        TENSOR_NAME, FORMAT, dalotia_C_ordering);
    assert(extents.size() == 3);
    assert(extents[0] == 3);
    assert(extents[1] == 4);
    assert(extents[2] == 5);
    assert(tensor.size() == NUM_ELEMENTS);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(tensor[i] == static_cast<double>(i));
    }
    std::cout << "OK" << std::endl;
}

void test_load_to_gpu() {
    std::cout << "test_load_to_gpu... " << std::flush;

    // Load reference on host
    auto [extents_ref, tensor_ref] = dalotia::load_tensor_dense<double>(
        TEST_FILE, TENSOR_NAME, FORMAT, dalotia_C_ordering);
    assert(tensor_ref.size() == NUM_ELEMENTS);

    // Open file (no GDS driver — tests the cudaMemcpy fallback)
    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    const size_t nbytes = NUM_ELEMENTS * sizeof(double);
    double* d_tensor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tensor, nbytes));

    // load_tensor_dense should detect the device pointer and use the
    // cudaMemcpy fallback (since no CuFileDriver is active)
    file->load_tensor_dense(TENSOR_NAME, FORMAT, dalotia_C_ordering,
                            reinterpret_cast<dalotia_byte*>(d_tensor));

    // Copy back and verify
    std::vector<double> h_result(NUM_ELEMENTS);
    CHECK_CUDA(
        cudaMemcpy(h_result.data(), d_tensor, nbytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(h_result[i] == tensor_ref[i]);
    }

    CHECK_CUDA(cudaFree(d_tensor));
    std::cout << "OK (via fallback)" << std::endl;
}
#ifdef DALOTIA_WITH_CUFILE
void test_load_to_gpu_with_driver() {
    std::cout << "test_load_to_gpu_with_driver... " << std::flush;

    auto [extents_ref, tensor_ref] = dalotia::load_tensor_dense<double>(
        TEST_FILE, TENSOR_NAME, FORMAT, dalotia_C_ordering);

    auto driver = try_open_driver();
    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    const size_t nbytes = NUM_ELEMENTS * sizeof(double);
    double* d_tensor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tensor, nbytes));

    file->load_tensor_dense(TENSOR_NAME, FORMAT, dalotia_C_ordering,
                            reinterpret_cast<dalotia_byte*>(d_tensor));

    std::vector<double> h_result(NUM_ELEMENTS);
    CHECK_CUDA(
        cudaMemcpy(h_result.data(), d_tensor, nbytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(h_result[i] == tensor_ref[i]);
    }

    CHECK_CUDA(cudaFree(d_tensor));
    std::cout << "OK" << (driver ? " (GDS attempted)" : " (fallback)")
              << std::endl;
}
#endif  // DALOTIA_WITH_CUFILE

void test_same_file_host_and_gpu() {
    // Load the same tensor from a single SafetensorsFile instance to both
    // a host pointer and a device pointer, and verify both match.
    std::cout << "test_same_file_host_and_gpu... " << std::flush;

    auto driver = try_open_driver();
    dalotia::SafetensorsFile file(TEST_FILE);

    // Load to host
    auto [extents, h_tensor] =
        file.load_tensor_dense<double>(TENSOR_NAME, FORMAT, dalotia_C_ordering);
    assert(h_tensor.size() == NUM_ELEMENTS);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(h_tensor[i] == static_cast<double>(i));
    }

    // Load the same tensor to a device pointer (same file object)
    const size_t nbytes = NUM_ELEMENTS * sizeof(double);
    double* d_tensor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tensor, nbytes));

    file.load_tensor_dense(TENSOR_NAME, FORMAT, dalotia_C_ordering,
                           reinterpret_cast<dalotia_byte*>(d_tensor));

    // Copy back and verify both results match
    std::vector<double> h_result(NUM_ELEMENTS);
    CHECK_CUDA(
        cudaMemcpy(h_result.data(), d_tensor, nbytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(h_result[i] == h_tensor[i]);
    }

    CHECK_CUDA(cudaFree(d_tensor));
    std::cout << "OK" << std::endl;
}

void test_permuted_load_to_gpu() {
    std::cout << "test_permuted_load_to_gpu... " << std::flush;

    auto driver = try_open_driver();
    // The test model has "embedding_firstchanged" with shape [4,3,5].
    // Permutation [1,0,2] gives shape [3,4,5] with values 0..59.
    const char* perm_tensor = "embedding_firstchanged";
    std::vector<int> perm = {1, 0, 2};

    // Load with permutation on CPU as reference
    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));
    auto [extents_ref, h_ref] = file->load_tensor_dense<double>(
        perm_tensor, FORMAT, dalotia_C_ordering, perm);
    assert(extents_ref == std::vector<int>({3, 4, 5}));
    assert(h_ref.size() == NUM_ELEMENTS);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(h_ref[i] == static_cast<double>(i));
    }

    // Now load with permutation directly to GPU
    const size_t nbytes = NUM_ELEMENTS * sizeof(double);
    double* d_tensor = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tensor, nbytes));

    file->load_tensor_dense(perm_tensor, FORMAT, dalotia_C_ordering,
                            reinterpret_cast<dalotia_byte*>(d_tensor), perm);

    // Copy back and verify
    std::vector<double> h_result(NUM_ELEMENTS);
    CHECK_CUDA(
        cudaMemcpy(h_result.data(), d_tensor, nbytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(h_result[i] == h_ref[i]);
    }

    CHECK_CUDA(cudaFree(d_tensor));
    std::cout << "OK" << std::endl;
}

int main() {
    test_is_device_pointer();
#ifdef DALOTIA_WITH_CUFILE
    test_external_driver_open();
#endif
    test_host_pointer();
    test_load_to_gpu();
#ifdef DALOTIA_WITH_CUFILE
    test_load_to_gpu_with_driver();
#endif
    test_same_file_host_and_gpu();
    test_permuted_load_to_gpu();
    std::cout << "test_cufile succeeded" << std::endl;
    return 0;
}
