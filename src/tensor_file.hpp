#pragma once

#include <stdio.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "dalotia_formats.hpp"

namespace dalotia {
class TensorFile {
   public:
    TensorFile(std::string filename) {
        // bool opened = (this->file_ = fopen(filename.c_str(), "rb"));
        // if (!opened) {
        //     throw std::runtime_error("Could not open file " + filename);
        // }
    }

    TensorFile(const TensorFile &) = delete;
    TensorFile &operator=(const TensorFile &) = delete;
    TensorFile(TensorFile &&) = delete;
    TensorFile &operator=(TensorFile &&) = delete;

    virtual ~TensorFile() {
        // assert(this->file_ != nullptr);
        // fclose(this->file_);
    }

    virtual const std::vector<std::string> &get_tensor_names() const {
        throw std::runtime_error(
            "get_tensor_names not implemented for this tensor type");
    }

    virtual bool is_sparse(std::string tensor_name) const {
        throw std::runtime_error(
            "is_sparse not implemented for this tensor type");
        return false;
    }

    virtual size_t get_num_dimensions(std::string tensor_name) const {
        auto extents = this->get_tensor_extents(tensor_name);
        auto num_not_dimensions =
            std::count(extents.begin(), extents.end(), -1);
        return extents.size() - num_not_dimensions;
    }

    virtual std::array<int, 10> get_tensor_extents(
        std::string tensor_name = "",
        const int *permutation =
            nullptr) const  //? have the maximum number of dimensions = 10?
    {
        throw std::runtime_error(
            "get_tensor_extents not implemented for this tensor type");
        return std::array<int, 10>();
    }

    virtual size_t get_num_tensor_elements(std::string tensor_name) const {
        // ?

        auto long_extents = this->get_tensor_extents(tensor_name);
        auto num_zero =
            long_extents.size() -
            std::count(long_extents.begin(), long_extents.end(), -1);
        return std::accumulate(
            long_extents.begin(),
            long_extents.begin() + (long_extents.size() - num_zero), 1,
            std::multiplies<size_t>());
    }

    virtual size_t get_nnz(std::string tensor_name) const {
        // This function will read the file and return the number of non-zero
        // elements ? may take a while for dense tensors, only allow for sparse?
        throw std::runtime_error(
            "get_nnz not implemented for this tensor type");
        return 0;
    }

    virtual std::array<int, 10> get_sparse_tensor_extents(
        std::string tensor_name, dalotia_SparseFormat format) const {
        // This function will (lazily) read the file and return the tensor
        // extents
        throw std::runtime_error(
            "get_sparse_tensor_extents not implemented for this tensor type");
        return std::array<int, 10>();
    }

    virtual void load_tensor_dense(std::string tensor_name,
                                   dalotia_WeightFormat weightFormat,
                                   dalotia_Ordering ordering,
                                   std::byte *__restrict__ tensor,
                                   const int *permutation = nullptr) {
        // This function will read the whole file and load the tensor,
        // optionally transposing it according to the permutation
        throw std::runtime_error(
            "load_tensor_dense not implemented for this tensor type");
    }

    virtual void load_tensor_sparse(std::string tensor_name,
                                    dalotia_SparseFormat sparseFormat,
                                    dalotia_WeightFormat weightFormat,
                                    dalotia_Ordering ordering,
                                    std::byte *values, int *first_indices,
                                    int *second_indices) {
        // This function will read the whole file and load the tensor into the
        // three arrays
        throw std::runtime_error(
            "load_tensor_sparse not implemented for this tensor type");
    }

    // no private section to allow visibility from C
    // FILE *file_ = nullptr;
};

}  // namespace dalotia
