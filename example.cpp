#include <array>
#include <cassert>
#include <iostream>  //TODO remove
#include <memory_resource>
#include <string>
#include <vector>

#include "dalotia.h"
#include "dalotia.hpp"

// application code

int main(int argc, char *argv[]) {
    char filename[] = "data/model.safetensors";
    char tensor_name[] = "embedding_firstchanged";
    DalotiaTensorFile *file = dalotia_open_file(filename);
    bool tensor_is_sparse =
        dalotia_is_sparse(file, tensor_name);  //...repeat later
    char *tensor;
    int permutation[3] = {0, 1, 2};
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_64;
    dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;

    std::cout << "tensor is sparse: " << tensor_is_sparse << std::endl;

    if (!tensor_is_sparse) {
        // get the tensor extents
        int extents[10];  // ? or call get_num_dimensions before?
                          // file formats: gguf, safetensors, onnx? channel
                          // orders? gguf with quantized ops?
                          // -> look at dnnl, darknet, safetensors-cpp, tinyml?
                          // torch: named dimensions tensor name data format
                          // data shape offsets
        int num_dimensions =
            dalotia_get_tensor_extents(file, tensor_name, extents);
        std::cout << "num_dim: " << num_dimensions << std::endl;

        // calculate the total number of elements
        int total_size = 1;
        for (int i = 0; i < 10; i++) {
            if (extents[i] == -1) {
                assert(i > 0);
                break;
            }
            total_size *= extents[i];
        }

        assert(total_size ==
               dalotia_get_num_tensor_elements(file, tensor_name));
        std::cout << "total size: " << total_size << std::endl;

        // I want to store the tensor as a very long array
        // allocate memory for the tensor
        tensor = (char *)malloc(dalotia::sizeof_weight_format<weightFormat>() *
                                total_size);

        // load the tensor

        dalotia_load_tensor_dense_with_permutation(
            file, tensor_name, tensor, weightFormat, ordering, permutation);
        // load_tensor_dense(file, tensor_name, tensor, weightFormat, ordering);
    } else {
        dalotia_SparseFormat format = dalotia_SparseFormat::dalotia_CSR;
        // get the tensor extents
        int extents[10];
        int num_dimensions =
            dalotia_get_tensor_extents(file, tensor_name, extents);

        for (int i = 0; i < 10; i++) {
            if (extents[i] == -1) {
                assert(i > 0);
                break;
            }
            std::cout << extents[i] << " ";
        }
        int sparse_extents[10];
        dalotia_get_sparse_tensor_extents(file, tensor_name, sparse_extents,
                                          dalotia_CSR);

        for (int i = 0; i < 10; i++) {
            if (sparse_extents[i] == -1) {
                assert(i > 0);
                break;
            }
            std::cout << sparse_extents[i] << " ";
        }

        // I want to store the tensor as compressed sparse row
        char *values = reinterpret_cast<char *>(
            new float[sparse_extents[0]]);  // blah blah malloc...
        int *first_indices = new int[sparse_extents[1]];
        int *second_indices = new int[sparse_extents[2]];
        dalotia_load_tensor_sparse(file, tensor_name, values, first_indices,
                                   second_indices, format, weightFormat,
                                   ordering);
    }
    dalotia_close_file(file);

    // print
    if (!tensor_is_sparse) {
        double *tensor_double = reinterpret_cast<double *>(tensor);
        for (int i = 0; i < 256; i++) {
            std::cout << tensor_double[i] << " ";
        }
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // alternative: the C++17 version
    auto [extents, tensor_cpp] =
        dalotia::load_tensor_dense(filename, tensor_name, weightFormat);

    // small tensors can even live on the stack!
    auto vector_permutation = std::pmr::vector<int>{1, 2, 0};
    std::array<double, 300> storage_array;
    std::pmr::monotonic_buffer_resource storage_resource(
        storage_array.data(), storage_array.size() * sizeof(double));
    std::pmr::polymorphic_allocator<dalotia_byte> storage_allocator(
        &storage_resource);
    auto [extents2, tensor_cpp2] = dalotia::load_tensor_dense<double>(
        filename, tensor_name, weightFormat, dalotia_C_ordering,
        storage_allocator, vector_permutation);

    for (int i = 0; i < storage_array.size(); ++i) {
        std::cout << storage_array[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < extents2.size(); ++i) {
        std::cout << extents2[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < tensor_cpp2.size(); ++i) {
        std::cout << tensor_cpp2[i] << " ";
    }
    std::cout << std::endl;

    //... // do something with the tensor
    // everything alex calls a runtime = possibly jit compiled

    // example: nicam ai, genesis, ...reconstruct?
    // run here with cuDNN, or BLIS, or...

    return 0;
}
