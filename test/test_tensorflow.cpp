#include <cassert>
#include <iostream>

#include "dalotia.h"
#include "dalotia.hpp"

void test_names() {
    std::string filename = "../data/tensorflow_model";
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_64;
    dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;


    // test the TensorflowSavedModel class
    std::unique_ptr<dalotia::TensorFile> dalotia_file(
        dalotia::make_tensor_file(filename));
    if(dalotia_file == nullptr) {
        throw std::runtime_error("Failed to open TensorFlow model file: " + filename);
    }
    auto tensor_names = dalotia_file->get_tensor_names();
    assert(!tensor_names.empty());
    std::cout << "Tensor names in the file: " << std::endl;
    for (const auto &name : tensor_names) {
        std::cout << " - " << name << std::endl;    
    }

    for (const auto &name : tensor_names) {
        // for all tensor names, check if they are sparse and get their number of dimensions
        bool is_sparse = dalotia_file->is_sparse(name);
        size_t num_dimensions = dalotia_file->get_num_dimensions(name);
        if (num_dimensions < 0) {
            throw std::runtime_error("Tensor " + name + " has " + std::to_string(num_dimensions) + " dimensions, which is unexpected.");
        }
        if (num_dimensions > 1) {
            // test get_tensor_extents
            auto extents = dalotia_file->get_tensor_extents(name);
            std::cout << "Tensor: " << name << ", extents: " << dalotia::to_string(extents) << std::endl;
            // test load_tensor_dense
            std::unique_ptr<dalotia_byte[]> tensor(
                new dalotia_byte[dalotia::sizeof_weight_format<weightFormat>() * dalotia_file->get_num_tensor_elements(name)]);
            dalotia_file->load_tensor_dense(name, weightFormat, ordering, tensor.get());
            std::cout << "Loaded tensor: " << name << ", size: "
                      << dalotia::sizeof_weight_format<weightFormat>() * dalotia_file->get_num_tensor_elements(name) << " bytes, contents: " << tensor.get() << std::endl;
        }
    }
#ifdef DALOTIA_WITH_CPP_PMR
    {
        std::string tensor_name = "dense/kernel/Read/ReadVariableOp";
        auto [extents, tensor_cpp] = dalotia::load_tensor_dense<double>(
            filename, tensor_name, weightFormat, ordering);
    }
#endif // DALOTIA_WITH_CPP_PMR
}

int main(int, char **) {
    test_names();
    std::cout << "test_tensorflow succeded" << std::endl;
    throw std::runtime_error("test_tensorflow not implemented yet");
    return 0;
}