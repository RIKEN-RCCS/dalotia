#include "dalotia_safetensors_file.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "dalotia_assignment.hpp"
// #include "dalotia_formats.hpp"
#include "safetensors.hh"

namespace dalotia {

safetensors::tensor_t get_only_tensor(const safetensors::safetensors_t &st) {
    safetensors::tensor_t tensor;
    assert(st.tensors.size() == 1);
    st.tensors.at(0, &tensor);
    return tensor;
}

safetensors::tensor_t get_tensor_from_name(
    std::string tensor_name, const safetensors::safetensors_t &st) {
    if (tensor_name.empty()) {
        return get_only_tensor(st);
    }
    for (size_t i = 0; i < st.tensors.size(); i++) {
        std::string key = st.tensors.keys()[i];
        if (key == tensor_name) {
            safetensors::tensor_t tensor;
            st.tensors.at(i, &tensor);
            return tensor;
        }
    }

    const std::string joined_keys =
            std::accumulate(st.tensors.keys().begin(),
                            st.tensors.keys().end(),
                            std::string(),
                            [](const std::string& a, const std::string& b)
                                    -> std::string {
                                    return a + (a.length() > 0 ? "," : "") + b;
                                }
                            );

    throw std::runtime_error("Tensor " + tensor_name + " not found; available: " +
                             joined_keys);
}

const std::vector<std::string> &SafetensorsFile::get_tensor_names() const {
    return st_.tensors.keys();
}

SafetensorsFile::SafetensorsFile(std::string filename) : TensorFile(filename) {
    // as far as I can tell, safetensors are saved in C order
    std::string warn, err;
    bool ret = safetensors::mmap_from_file(filename, &st_, &warn, &err);
    if (warn.size() > 0) {
        std::cout << "safetensors-cpp WARN: " << warn << "\n";
    }
    if (ret == false) {
        std::cerr << "Failed to load: " << filename << "\n";
        std::cerr << "  ERR: " << err << "\n";
        throw std::runtime_error("Could not open file " + filename);
    }
    // Check if data_offsets are valid //TODO maybe only in debug mode
    if (!safetensors::validate_data_offsets(st_, err)) {
        std::cerr << "Invalid data_offsets\n";
        std::cerr << err << "\n";
        throw std::runtime_error("Invalid safetensors file " + filename);
    }
}

SafetensorsFile::~SafetensorsFile() {
    if (st_.st_file != nullptr) {
        // delete st_.st_file;
    }
}

bool SafetensorsFile::is_sparse(std::string /*tensor_name*/) const {
    return false;  // TODO figure out how sparsity works / could work
}

size_t SafetensorsFile::get_num_dimensions(std::string tensor_name) const {
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
    return safetensor.shape.size();
}

size_t SafetensorsFile::get_num_tensor_elements(std::string tensor_name) const {
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
    return safetensors::get_shape_size(safetensor);
}

std::array<int, 10> SafetensorsFile::get_tensor_extents(
    std::string tensor_name, const int *permutation) const {
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
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
                extents[i] = safetensor.shape[final_permutation_in_c_order[i]];
            }
        }
    }
    return extents;
}

void SafetensorsFile::load_tensor_dense(std::string tensor_name,
                                        dalotia_WeightFormat weightFormat,
                                        dalotia_Ordering ordering,
                                        dalotia_byte *__restrict__ tensor,
                                        const int *permutation) {
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
    const auto num_dimensions = safetensor.shape.size();

    auto final_permutation_in_c_order =
        final_c_permutation_from_permutation_and_order(permutation, ordering,
                                                       num_dimensions);

    const uint8_t *databuffer = st_.databuffer_addr;
    const dalotia_WeightFormat input_weight_format =
        safetensors_type_map.at(safetensor.dtype);
    auto *tensor_start =
        reinterpret_cast<const dalotia_byte *__restrict__>(databuffer) +
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

void SafetensorsFile::load_tensor_sparse(
    std::string /* tensor_name */, dalotia_SparseFormat /* sparseFormat */,
    dalotia_WeightFormat /* weightFormat */, dalotia_Ordering /* ordering */,
    dalotia_byte * /* values */, int * /* first_indices */,
    int * /*second_indices */) {
    throw std::runtime_error("Sparse tensors for safetensors not implemented");
}

}  // namespace dalotia