#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <string>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"
#include "safetensors.hh"
#include "tensor_file.hpp"

namespace dalotia {

const std::map<safetensors::dtype, dalotia_WeightFormat> safetensors_type_map{
    {safetensors::dtype::kFLOAT64, dalotia_WeightFormat::dalotia_float_64},
    {safetensors::dtype::kFLOAT32, dalotia_WeightFormat::dalotia_float_32},
    {safetensors::dtype::kFLOAT16, dalotia_WeightFormat::dalotia_float_16},
    {safetensors::dtype::kBFLOAT16, dalotia_WeightFormat::dalotia_bfloat_16},
    // {kBOOL, dalotia_bool},
    // {kUINT8, dalotia_uint_8},
    // {kINT8, dalotia_int_8},
    // {kUINT16, dalotia_uint_16},
    // {kINT32, dalotia_int_32},
    // {kUINT32, dalotia_uint_32},
    // {kINT64, dalotia_int_64},
    // {kUINT64, dalotia_uint_64},
    // {dalotia_float_8},
    // {dalotia_int_2},
};

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
        const size_t file_item_bytes =
            safetensors::get_dtype_bytes(safetensor.dtype);
        const dalotia_WeightFormat input_weight_format =
            safetensors_type_map.at(safetensor.dtype);
        auto *tensor_start =
            reinterpret_cast<const std::byte *__restrict__>(databuffer) +
            safetensor.data_offsets[0];
        if (!final_permutation_in_c_order.empty()) {
            assign_permuted(num_dimensions, tensor, weightFormat,
                            safetensor.shape.data(), tensor_start,
                            input_weight_format,
                            final_permutation_in_c_order.data());
        } else {
            const size_t nitems = safetensors::get_shape_size(safetensor);
            assign_linearly(tensor, weightFormat, nitems, tensor_start,
                            input_weight_format);
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