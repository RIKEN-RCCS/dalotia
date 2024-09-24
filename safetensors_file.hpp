#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <string>

#include "dalotia_formats.hpp"
#include "dalotia_permutation.hpp"
#include "safetensors.hh"
#include "tensor_file.hpp"

namespace dalotia {

// TODO move to own header once fully tested

template <uint8_t num_dimensions>
void assign_permuted(std::byte *__restrict__dest,
                     const safetensors::tensor_t &src,
                     const std::byte *__restrict__tensor_start,
                     const int *permutation) {
    throw std::runtime_error("assign_permuted not yet implemented for " +
                             std::to_string(num_dimensions) + " dimensions");
}

// specialization for 2d
template <>
void assign_permuted<2>(std::byte *__restrict__ dest,
                        const safetensors::tensor_t &src,
                        const std::byte *__restrict__ tensor_start,
                        const int *permutation) {
    assert(src.shape.size() == 2);
    auto desired_shape = src.shape;
    for (size_t i = 0; i < src.shape.size(); ++i) {
        desired_shape[permutation[i]] = src.shape[i];
    }
    assert(permutation[0] == 1);
    assert(permutation[1] == 0);
    const size_t file_item_bytes = safetensors::get_dtype_bytes(src.dtype);
    // const size_t load_item_bytes = sizeof(float); //TODO casting
    size_t load_index = 0;
    for (size_t i = 0; i < src.shape[1]; ++i) {
        for (size_t j = 0; j < src.shape[0]; ++j) {
            auto store_index = j * src.shape[1] + i;
            for (size_t k = 0; k < file_item_bytes; ++k) {
                auto element_pointer =
                    tensor_start + load_index * file_item_bytes;
                dest[store_index * file_item_bytes + j] =
                    static_cast<std::byte>(element_pointer[k]);
            }
            ++load_index;
        }
    }

    assert(load_index == safetensors::get_shape_size(src));
}

// specialization for 3d
template <>
void assign_permuted<3>(std::byte *__restrict__ dest,
                        const safetensors::tensor_t &src,
                        const std::byte *__restrict__ tensor_start,
                        const int *permutation) {
    assert(src.shape.size() == 3);
    auto desired_shape = src.shape;
    for (size_t i = 0; i < 3; ++i) {
        desired_shape[permutation[i]] = src.shape[i];
    }
    auto new_strides = std::array<size_t, 3>();
    // C order -> last dimension is the most contiguous -> rbegin
    std::exclusive_scan(desired_shape.rbegin(), desired_shape.rend(),
                        new_strides.rbegin(), 1, std::multiplies<>{});
    auto new_strides_permuted = new_strides;
    for (size_t i = 0; i < 3; ++i) {
        new_strides_permuted[i] = new_strides[permutation[i]];
    }

    const size_t file_item_bytes = safetensors::get_dtype_bytes(src.dtype);
    size_t load_index = 0;
    std::array<size_t, 3> load_index_array{0, 0, 0};
    auto &[i, j, k] = load_index_array;
    // this is sequential load / permuted store
    // untested whether it would be faster the other way around
    for (i = 0; i < src.shape[0]; ++i) {
        for (j = 0; j < src.shape[1]; ++j) {
            for (k = 0; k < src.shape[2]; ++k) {
                auto store_index = std::inner_product(
                    new_strides_permuted.begin(), new_strides_permuted.end(),
                    load_index_array.begin(), 0);
                assert(store_index < safetensors::get_shape_size(src));

                for (size_t l = 0; l < file_item_bytes; ++l) {
                    auto element_pointer =
                        tensor_start + load_index * file_item_bytes;
                    dest[store_index * file_item_bytes + l] =
                        static_cast<std::byte>(element_pointer[l]);
                }
                ++load_index;
            }
        }
    }

    assert(load_index == safetensors::get_shape_size(src));
}

// TODO use BOOST_PP_* to generate something like Julia @nloops macro?
// for all dimensions
// ->
// https://github.com/JuliaLang/julia/blob/master/base/multidimensional.jl#L1685
// https://stackoverflow.com/questions/77130743/boost-preprocessor-boost-pp-local-iterate-nested-loops

class SafetensorsFile : public TensorFile {
   public:
    SafetensorsFile(std::string filename) : TensorFile(filename) {
        // as far as I can tell, safetensors are saved in C order
        std::string warn, err;
        bool ret = safetensors::mmap_from_file(filename, &st, &warn, &err);
        if (warn.size() > 0) {
            std::cout << "safetensors-cpp WARN: " << warn << "\n";
        }
        if (ret == false) {
            std::cerr << "Failed to load: " << filename << "\n";
            std::cerr << "  ERR: " << err << "\n";
            throw std::runtime_error("Could not open file " + filename);
        }
        // Check if data_offsets are valid //TODO maybe only in debug mode
        if (!safetensors::validate_data_offsets(st, err)) {
            std::cerr << "Invalid data_offsets\n";
            std::cerr << err << "\n";
            throw std::runtime_error("Invalid safetensors file " + filename);
        }
    }

    ~SafetensorsFile() {
        if (st.st_file != nullptr) {
            //?free  // TODO not sure where ownership is
        }
    }

    safetensors::tensor_t get_only_tensor() {
        safetensors::tensor_t tensor;
        assert(st.tensors.size() == 1);
        st.tensors.at(0, &tensor);
        return tensor;
    }

    safetensors::tensor_t get_tensor_from_name(std::string tensor_name) {
        if (tensor_name == "") {
            return get_only_tensor();
        }
        for (size_t i = 0; i < st.tensors.size(); i++) {
            std::string key = st.tensors.keys()[i];
            if (key == tensor_name) {
                safetensors::tensor_t tensor;
                st.tensors.at(i, &tensor);
                return tensor;
            }
        }
        throw std::runtime_error("Tensor " + tensor_name + " not found");
    }

    bool is_sparse(std::string tensor_name) override {
        return false;  // TODO figure out how sparsity works / could work
    }

    size_t get_num_dimensions(std::string tensor_name) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        return safetensor.shape.size();
    }

    size_t get_num_tensor_elements(std::string tensor_name) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        return safetensors::get_shape_size(safetensor);
    }

    std::array<int, 10> get_tensor_extents(
        std::string tensor_name = "",
        const int *permutation = nullptr) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        std::array<int, 10> extents;
        for (size_t i = 0; i < safetensor.shape.size(); i++) {
            extents[i] = safetensor.shape[i];
        }
        for (size_t i = safetensor.shape.size(); i < 10; i++) {
            extents[i] = -1;
        }
        if (permutation != nullptr) {
            auto final_permutation_in_c_order =
                final_c_permutation_from_permutation_and_order(
                    permutation, dalotia_Ordering::dalotia_C_ordering,
                    safetensor.shape.size());
            if (!final_permutation_in_c_order.empty()) {
                for (size_t i = 0; i < safetensor.shape.size(); i++) {
                    extents[i] =
                        safetensor.shape[final_permutation_in_c_order[i]];
                }
            }
        }
        return extents;
    }

    void load_tensor_dense(std::string tensor_name,
                           dalotia_WeightFormat weightFormat,
                           dalotia_Ordering ordering,
                           std::byte *__restrict__ tensor,
                           const int *permutation = nullptr) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        const auto num_dimensions = safetensor.shape.size();

        auto final_permutation_in_c_order =
            final_c_permutation_from_permutation_and_order(
                permutation, ordering, num_dimensions);

        const uint8_t *databuffer = st.databuffer_addr;
        const size_t nitems = safetensors::get_shape_size(safetensor);
        const size_t file_item_bytes =
            safetensors::get_dtype_bytes(safetensor.dtype);
        const size_t load_item_bytes =
            dalotia::sizeof_weight_format(weightFormat);
        auto *tensor_start =
            reinterpret_cast<const std::byte *__restrict__>(databuffer) +
            safetensor.data_offsets[0];
        if (!final_permutation_in_c_order.empty()) {
            if (num_dimensions == 2) {
                assign_permuted<2>(tensor, safetensor, tensor_start,
                                   final_permutation_in_c_order.data());
            } else if (num_dimensions == 3) {
                assign_permuted<3>(tensor, safetensor, tensor_start,
                                   final_permutation_in_c_order.data());
            } else {
                throw std::runtime_error(
                    "assign_permuted not yet implemented for " +
                    std::to_string(num_dimensions) + " dimensions");
            }
        } else {
            for (size_t i = 0; i < nitems; i++) {
                auto element_pointer = tensor_start + i * file_item_bytes;
                // TODO cast from safetensor.dtype to weightFormat -- how ???
                // use gmpxx? use quantization things?
                assert(load_item_bytes == file_item_bytes);
                for (size_t j = 0; j < load_item_bytes; ++j) {
                    tensor[i * load_item_bytes + j] =
                        static_cast<std::byte>(element_pointer[j]);
                }
            }
        }
    }

    void load_tensor_sparse(std::string tensor_name,
                            dalotia_SparseFormat sparseFormat,
                            dalotia_WeightFormat weightFormat,
                            dalotia_Ordering ordering, std::byte *values,
                            int *first_indices, int *second_indices) override {
        throw std::runtime_error(
            "Sparse tensors for safetensors not implemented");
    }
    safetensors::safetensors_t st;
};

}  // namespace dalotia