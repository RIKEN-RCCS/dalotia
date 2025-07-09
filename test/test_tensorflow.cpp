#include <cassert>
#include <iostream>

#include "dalotia.h"
#include "dalotia.hpp"
#include "test_helper.h"

void test_names() {
    std::string filename = "../data/tensorflow_model";
    constexpr dalotia_WeightFormat weightFormat = dalotia_WeightFormat::dalotia_float_64;
    dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;

    // test the TensorflowSavedModel class
    std::unique_ptr<dalotia::TensorFile> dalotia_file(
        dalotia::make_tensor_file(filename));
    if (dalotia_file == nullptr) {
        throw std::runtime_error("Failed to open TensorFlow model file: " + filename);
    }
    auto tensor_names = dalotia_file->get_tensor_names();
    assert(!tensor_names.empty());
    std::cout << "Tensor names in the file: " << std::endl;
    for (const auto &name : tensor_names) {
        std::cout << " - " << name << std::endl;
    }

    for (const auto &name : tensor_names) {
        // for all tensor names, check if they are sparse and get their number of
        // dimensions
        bool is_sparse = dalotia_file->is_sparse(name);
        assert(!is_sparse);
        size_t num_dimensions = dalotia_file->get_num_dimensions(name);
        if (num_dimensions < 0) {
            throw std::runtime_error("Tensor " + name + " has " +
                                     std::to_string(num_dimensions) +
                                     " dimensions, which is unexpected.");
        }
        if (num_dimensions > 0) {
            // test get_tensor_extents
            auto extents = dalotia_file->get_tensor_extents(name);
            // test load_tensor_dense
            if (*std::min_element(extents.begin(), extents.end()) > 0) {
                std::unique_ptr<dalotia_byte[]> tensor(
                    new dalotia_byte[dalotia::sizeof_weight_format<weightFormat>() *
                                     dalotia_file->get_num_tensor_elements(name)]);
                dalotia_file->load_tensor_dense(name, weightFormat, ordering,
                                                tensor.get());
            }
        }
    }
#ifdef DALOTIA_WITH_CPP_PMR
    {
        std::string tensor_name = "dense/kernel/Read/ReadVariableOp";
        auto [extents, tensor_cpp_double] = dalotia::load_tensor_dense<double>(
            filename, tensor_name, weightFormat, ordering);
        assert(!extents.empty());
        assert(!tensor_cpp_double.empty());
        auto [extents_float, tensor_cpp_float] = dalotia::load_tensor_dense<float>(
            filename, tensor_name, dalotia_WeightFormat::dalotia_float_32, ordering);
        for (size_t i = 0; i < extents.size(); ++i) {
            assert(extents[i] == extents_float[i]);
        }
        for (size_t i = 0; i < tensor_cpp_double.size(); ++i) {
            assert(tensor_cpp_float[i] == static_cast<float>(tensor_cpp_double[i]));
        }
        std::vector<double> true_values_begin = {
            -0.25138268, -0.25613192, 0.16491315, -0.13381714, 0.35687172, -0.35824186,
            0.3529436,   -0.55490106, 0.27651784, 0.30784482,  -0.2846631};
        for (size_t i = 0; i < true_values_begin.size(); ++i) {
            assert_close(tensor_cpp_double[i], true_values_begin[i]);
        }
        assert_close(tensor_cpp_double.back(), -0.21346514);
    }
#endif  // DALOTIA_WITH_CPP_PMR
}

int main(int, char **) {
    test_names();
    std::cout << "test_tensorflow succeded" << std::endl;
    return 0;
}