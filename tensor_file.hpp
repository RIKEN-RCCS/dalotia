#pragma once

#include <stdio.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>  //TODO remove
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string>
#include <vector>

#include "dalotia_formats.hpp"

namespace dalotia {
class TensorFile {
   public:
    TensorFile(std::string filename) {
        bool opened = (this->file_ = fopen(filename.c_str(), "rb"));
        if (!opened) {
            throw std::runtime_error("Could not open file " + filename);
        }
    }

    TensorFile(const TensorFile &) = delete;
    TensorFile &operator=(const TensorFile &) = delete;
    TensorFile(TensorFile &&) = delete;
    TensorFile &operator=(TensorFile &&) = delete;

    virtual ~TensorFile() {
        assert(this->file_ != nullptr);
        fclose(this->file_);
    }

    virtual bool is_sparse(std::string tensor_name) {
        // This function will (lazily) read the file and return true if the
        // tensor is sparse
        return true;
    }

    virtual size_t get_num_dimensions(std::string tensor_name) {
        // This function will (lazily) read the file and return the number of
        // dimensions
        return 3;
    }

    virtual std::array<int, 10> get_tensor_extents(
        std::string tensor_name =
            "")  //? have the maximum number of dimensions = 10?
    {
        // This function will (lazily) read the file and return the tensor
        // extents, passing -1 for "unused" dimensions
        return {
            5, 4, 3, -1, -1, -1, -1, -1, -1, -1,
        };
    }

    virtual size_t get_num_tensor_elements(std::string tensor_name) {
        // ?

        auto long_extents = this->get_tensor_extents(tensor_name);
        auto num_nonzero =
            long_extents.size() -
            std::count(long_extents.begin(), long_extents.end(), -1);
        return std::accumulate(long_extents.begin(),
                               long_extents.begin() + num_nonzero, 1,
                               std::multiplies<size_t>());
    }

    virtual size_t get_nnz(std::string tensor_name) {
        // This function will read the file and return the number of non-zero
        // elements ? may take a while for dense tensors, only allow for sparse?
        return 12;
    }

    virtual std::array<int, 10> get_sparse_tensor_extents(
        std::string tensor_name, dalotia_SparseFormat format) {
        // This function will (lazily) read the file and return the tensor
        // extents
        return {
            12, 12, 13, -1, -1, -1, -1, -1, -1, -1,
        };
    }

    virtual void load_tensor_dense(std::string tensor_name,
                                   dalotia_WeightFormat weightFormat,
                                   dalotia_Ordering ordering, std::byte *tensor,
                                   const int *permutation = nullptr) {
        // This function will read the whole file and load the tensor,
        // optionally transposing it according to the permutation
        const auto num_elements = this->get_num_tensor_elements(tensor_name);
        assert(sizeof_weight_format(weightFormat) ==
               4);  // assume float for now
        for (size_t i = 0; i < num_elements; i++) {
            float f = static_cast<float>(i);
            auto b = reinterpret_cast<std::byte *>(&f);
            for (size_t j = 0; j != sizeof(float); ++j) {
                tensor[i * sizeof(float) + j] = b[j];
            }
        }
    }

    virtual void load_tensor_sparse(std::string tensor_name,
                            dalotia_SparseFormat sparseFormat,
                            dalotia_WeightFormat weightFormat,
                            dalotia_Ordering ordering, std::byte *values,
                            int *first_indices, int *second_indices) {
        // This function will read the whole file and load the tensor into the
        // three arrays
        const auto num_nonzero = this->get_nnz(tensor_name);
        for (int i = 0; i < num_nonzero; ++i) {
            values[i] = static_cast<std::byte>(i);  // chunk from weightFormat;
            first_indices[i] = i;
            second_indices[i] = i;  // TODO implement
        }
    }

    // no private section to allow visibility from C
    FILE *file_ = nullptr;
};

// C++17 version -> will not compile on Fugaku...
// -- pmr vector types can accept different allocators
//? more memory interface than that? detect if CUDA device pointer through
// unified access... how about other devices?
template <typename value_type = std::byte>  //? or have no defaults?
[[nodiscard]] std::pair<std::pmr::vector<int>, std::pmr::vector<value_type>>
load_tensor_dense(std::string filename, std::string tensor_name,
                  dalotia_WeightFormat weight_format,
                  dalotia_Ordering ordering = dalotia_C_ordering,
                  const std::pmr::polymorphic_allocator<std::byte> &allocator =
                      std::pmr::polymorphic_allocator<std::byte>(),
                  const std::pmr::vector<int> &permutation = {}) {
    auto dalotia_file = TensorFile(filename);
    auto long_extents = dalotia_file.get_tensor_extents(tensor_name);
    // shorten extents to nonzeros
    auto num_nonzero = long_extents.size() -
                       std::count(long_extents.begin(), long_extents.end(), -1);
    std::pmr::vector<int> true_extents(
        long_extents.begin(), long_extents.begin() + num_nonzero, allocator);
    auto total_size = std::accumulate(true_extents.begin(), true_extents.end(),
                                      1, std::multiplies<size_t>());

    std::pmr::vector<value_type> tensor(allocator);
    if constexpr (std::is_same_v<value_type, std::byte>) {
        tensor.resize(total_size * sizeof_weight_format(weight_format));
    } else {
        tensor.resize(total_size);
    }
    dalotia_file.load_tensor_dense(tensor_name, weight_format, ordering,
                                   reinterpret_cast<std::byte *>(tensor.data()),
                                   permutation.data());
    return std::make_pair(true_extents, tensor);
}

// TODO same for sparse

// TODO allow md-range sub-tensor requests
}  // namespace dalotia
