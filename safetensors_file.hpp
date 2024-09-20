#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <string>

#include "dalotia.hpp"
#include "safetensors.hh"

namespace dalotia {

// TODO move to own header once fully tested

std::pmr::vector<int> final_c_permutation_from_permutation_and_order(
    const int *permutation, dalotia_Ordering ordering, size_t num_dimensions) {
    std::pmr::vector<int> final_permutation_in_c_order;
    if (permutation == nullptr) {
        if (ordering == dalotia_Ordering::dalotia_F_ordering) {
            final_permutation_in_c_order.resize(num_dimensions);
            // assign reverse iota
            std::iota(final_permutation_in_c_order.rbegin(),
                      final_permutation_in_c_order.rend(), 0);
        }  // else leave empty
    } else {
        // find out if the permutation ranges from 0 to d-1 or 1 to d
        const auto [min, max] =
            std::minmax_element(permutation, permutation + num_dimensions);
        if (*min == 0 && *max == num_dimensions - 1) {
            final_permutation_in_c_order.assign(permutation,
                                                permutation + num_dimensions);
        } else if (*min == 1 && *max == num_dimensions) {
            final_permutation_in_c_order.resize(num_dimensions);
            std::transform(permutation, permutation + num_dimensions,
                           final_permutation_in_c_order.begin(),
                           [](int x) { return x - 1; });
        } else {
            throw std::runtime_error("Invalid permutation");
        }

        if (ordering == dalotia_Ordering::dalotia_F_ordering) {
            std::reverse(final_permutation_in_c_order.begin(),
                         final_permutation_in_c_order.end());
        } else {  // assume that 1-indexed permutations are only requested w/
                  // Fortran
                  // (remove if wrong assumption)
            assert(final_permutation_in_c_order[0] == permutation[0]);
        }
        {
            auto sorted_permutation = final_permutation_in_c_order;
            std::sort(sorted_permutation.begin(), sorted_permutation.end());
            const auto duplicate = std::adjacent_find(
                sorted_permutation.begin(), sorted_permutation.end());

            if (duplicate != sorted_permutation.end()) {
                throw std::runtime_error("dalotia: Invalid permutation");
            }
        }

        // if it is the same as iota, we can leave it empty
        if (std::is_sorted(final_permutation_in_c_order.begin(),
                           final_permutation_in_c_order.end())) {
            final_permutation_in_c_order.clear();
        }
    }
    return final_permutation_in_c_order;
}

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
        std::string tensor_name = "") override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        std::array<int, 10> extents;
        for (size_t i = 0; i < safetensor.shape.size(); i++) {
            extents[i] = safetensor.shape[i];
        }
        for (size_t i = safetensor.shape.size(); i < 10; i++) {
            extents[i] = -1;
        }
        return extents;
    }

    void load_tensor_dense(std::string tensor_name,
                           dalotia_WeightFormat weightFormat,
                           dalotia_Ordering ordering, std::byte *tensor,
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
        auto *tensor_start = reinterpret_cast<const std::byte *>(databuffer) +
                             safetensor.data_offsets[0];
        if (!final_permutation_in_c_order.empty()) {
                throw std::runtime_error(
                    "assign_permuted not yet implemented for " +
                    std::to_string(num_dimensions) + " dimensions");
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