// Tests for dalotia::cuda_{device,pinned,managed,async}_memory_resource.

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory_resource>
#include <vector>

#include <cuda_runtime.h>

#include "dalotia.hpp"
#include "dalotia_cuda_memory_resource.hpp"

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl;  \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

// Query the exact CUDA memory type for a pointer.
static cudaMemoryType get_memory_type(const void* ptr) {
    cudaPointerAttributes attrs{};
    CHECK_CUDA(cudaPointerGetAttributes(&attrs, ptr));
    return attrs.type;
}

static const char* TEST_FILE = "../data/model.safetensors";
static const char* TENSOR_NAME = "embedding";
static constexpr int NUM_ELEMENTS = 3 * 4 * 5;  // 60
static constexpr dalotia_WeightFormat FORMAT = dalotia_float_64;

void test_device_resource_basic() {
    std::cout << "test_device_resource_basic... " << std::flush;

    auto* mr = dalotia::cuda_device_resource();
    void* p = mr->allocate(1024);
    assert(p != nullptr);
    assert(dalotia::is_device_pointer(p));
    mr->deallocate(p, 1024);

    // zero-byte allocation returns nullptr and doesn't crash on dealloc
    void* z = mr->allocate(0);
    assert(z == nullptr);
    mr->deallocate(z, 0);

    std::cout << "OK" << std::endl;
}

void test_pinned_resource_basic() {
    std::cout << "test_pinned_resource_basic... " << std::flush;

    auto* mr = dalotia::cuda_pinned_resource();
    void* p = mr->allocate(1024);
    assert(p != nullptr);
    // pinned memory is only host-accessible — should not be detected as device
    assert(!dalotia::is_device_pointer(p));

    // verify it's actually writable from the host
    std::memset(p, 0xAB, 1024);

    mr->deallocate(p, 1024);

    void* z = mr->allocate(0);
    assert(z == nullptr);
    mr->deallocate(z, 0);

    std::cout << "OK" << std::endl;
}

void test_managed_resource_basic() {
    std::cout << "test_managed_resource_basic... " << std::flush;

    auto* mr = dalotia::cuda_managed_resource();
    void* p = mr->allocate(1024);
    assert(p != nullptr);
    // managed memory is detected as device pointer
    assert(dalotia::is_device_pointer(p));

    // managed memory is also host-accessible
    std::memset(p, 0xCD, 1024);
    CHECK_CUDA(cudaDeviceSynchronize());

    mr->deallocate(p, 1024);

    std::cout << "OK" << std::endl;
}

void test_async_resource_basic() {
    std::cout << "test_async_resource_basic... " << std::flush;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    dalotia::cuda_async_memory_resource mr(stream);
    assert(mr.stream() == stream);

    void* p = mr.allocate(1024);
    assert(p != nullptr);
    assert(dalotia::is_device_pointer(p));
    mr.deallocate(p, 1024);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));

    std::cout << "OK" << std::endl;
}

void test_is_equal() {
    std::cout << "test_is_equal... " << std::flush;

    auto* dev = dalotia::cuda_device_resource();
    auto* pin = dalotia::cuda_pinned_resource();
    auto* mgd = dalotia::cuda_managed_resource();

    // same-type singletons are equal
    dalotia::cuda_device_memory_resource dev2;
    assert(dev->is_equal(dev2));
    assert(dev2.is_equal(*dev));

    // different types are not equal
    assert(!dev->is_equal(*pin));
    assert(!dev->is_equal(*mgd));
    assert(!pin->is_equal(*mgd));

    // not equal to the default resource
    assert(!dev->is_equal(*std::pmr::get_default_resource()));

    // async resources: equal iff same stream
    cudaStream_t s1, s2;
    CHECK_CUDA(cudaStreamCreate(&s1));
    CHECK_CUDA(cudaStreamCreate(&s2));

    dalotia::cuda_async_memory_resource a1(s1);
    dalotia::cuda_async_memory_resource a1_copy(s1);
    dalotia::cuda_async_memory_resource a2(s2);

    assert(a1.is_equal(a1_copy));
    assert(!a1.is_equal(a2));
    assert(!a1.is_equal(*dev));

    CHECK_CUDA(cudaStreamDestroy(s1));
    CHECK_CUDA(cudaStreamDestroy(s2));

    // calling the accessor twice returns the same pointer
    assert(dalotia::cuda_device_resource() == dalotia::cuda_device_resource());
    assert(dalotia::cuda_pinned_resource() == dalotia::cuda_pinned_resource());
    assert(dalotia::cuda_managed_resource() ==
           dalotia::cuda_managed_resource());

    std::cout << "OK" << std::endl;
}

void test_pmr_vector_pinned() {
    std::cout << "test_pmr_vector_pinned... " << std::flush;

    auto* mr = dalotia::cuda_pinned_resource();
    std::pmr::polymorphic_allocator<double> alloc(mr);
    std::pmr::vector<double> v(alloc);

    v.resize(100);
    for (int i = 0; i < 100; i++)
        v[i] = static_cast<double>(i);

    // verify contents (pinned memory is host-accessible)
    for (int i = 0; i < 100; i++)
        assert(v[i] == static_cast<double>(i));

    // the underlying pointer should be pinned (DMA-capable), not device
    assert(!dalotia::is_device_pointer(v.data()));

    std::cout << "OK" << std::endl;
}

void test_pmr_vector_managed() {
    std::cout << "test_pmr_vector_managed... " << std::flush;

    auto* mr = dalotia::cuda_managed_resource();
    std::pmr::polymorphic_allocator<double> alloc(mr);
    std::pmr::vector<double> v(alloc);

    v.resize(100);
    for (int i = 0; i < 100; i++)
        v[i] = static_cast<double>(i);

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < 100; i++)
        assert(v[i] == static_cast<double>(i));

    assert(dalotia::is_device_pointer(v.data()));

    std::cout << "OK" << std::endl;
}

void test_load_tensor_with_pinned_resource() {
    std::cout << "test_load_tensor_with_pinned_resource... " << std::flush;

    auto* mr = dalotia::cuda_pinned_resource();
    std::pmr::polymorphic_allocator<dalotia_byte> alloc(mr);

    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    auto [extents, tensor] = file->load_tensor_dense<double>(
        TENSOR_NAME, FORMAT, dalotia_C_ordering, {}, alloc);

    assert(extents.size() == 3);
    assert(extents[0] == 3 && extents[1] == 4 && extents[2] == 5);
    assert(tensor.size() == NUM_ELEMENTS);

    // pinned memory is host-readable
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(tensor[i] == static_cast<double>(i));
    }

    std::cout << "OK" << std::endl;
}

void test_load_tensor_with_managed_resource() {
    std::cout << "test_load_tensor_with_managed_resource... " << std::flush;

    auto* mr = dalotia::cuda_managed_resource();
    std::pmr::polymorphic_allocator<dalotia_byte> alloc(mr);

    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    auto [extents, tensor] = file->load_tensor_dense<double>(
        TENSOR_NAME, FORMAT, dalotia_C_ordering, {}, alloc);

    assert(extents.size() == 3);
    assert(tensor.size() == NUM_ELEMENTS);

    CHECK_CUDA(cudaDeviceSynchronize());

    // managed memory is host-readable after sync
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        assert(tensor[i] == static_cast<double>(i));
    }

    // and it's device-accessible (allocator preserved through move)
    assert(dalotia::is_device_pointer(tensor.data()));

    std::cout << "OK" << std::endl;
}

void test_load_tensor_memory_types() {
    std::cout << "test_load_tensor_memory_types... " << std::flush;

    auto file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(TEST_FILE));

    // default allocator → cudaMemoryTypeUnregistered (plain heap)
    {
        auto [ext, tensor] = file->load_tensor_dense<double>(
            TENSOR_NAME, FORMAT, dalotia_C_ordering);
        assert(get_memory_type(tensor.data()) == cudaMemoryTypeUnregistered);
    }

    // pinned resource → cudaMemoryTypeHost
    {
        std::pmr::polymorphic_allocator<dalotia_byte> alloc(
            dalotia::cuda_pinned_resource());
        auto [ext, tensor] = file->load_tensor_dense<double>(
            TENSOR_NAME, FORMAT, dalotia_C_ordering, {}, alloc);
        assert(get_memory_type(tensor.data()) == cudaMemoryTypeHost);
        for (int i = 0; i < NUM_ELEMENTS; i++)
            assert(tensor[i] == static_cast<double>(i));
    }

    // managed resource → cudaMemoryTypeManaged
    {
        std::pmr::polymorphic_allocator<dalotia_byte> alloc(
            dalotia::cuda_managed_resource());
        auto [ext, tensor] = file->load_tensor_dense<double>(
            TENSOR_NAME, FORMAT, dalotia_C_ordering, {}, alloc);
        assert(get_memory_type(tensor.data()) == cudaMemoryTypeManaged);
        CHECK_CUDA(cudaDeviceSynchronize());
        for (int i = 0; i < NUM_ELEMENTS; i++)
            assert(tensor[i] == static_cast<double>(i));
    }

    std::cout << "OK" << std::endl;
}

int main() {
    test_device_resource_basic();
    test_pinned_resource_basic();
    test_managed_resource_basic();
    test_async_resource_basic();
    test_is_equal();
    test_singletons();
    test_pmr_vector_pinned();
    test_pmr_vector_managed();
    test_load_tensor_with_pinned_resource();
    test_load_tensor_with_managed_resource();
    test_load_tensor_memory_types();

    std::cout << "test_cuda_memory_resource succeeded" << std::endl;
    return 0;
}
