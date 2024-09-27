#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string>
#include <vector>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"
#include "tensor_file.hpp"

#ifdef DALOTIA_WITH_SAFETENSORS_CPP
#include "safetensors_file.hpp"
#endif

namespace dalotia {
// factory function for the file, selected by file extension and
// available implementations
TensorFile *make_tensor_file(std::string filename) {
    // make sure the file exists
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("File " + filename + " does not exist");
    }

    // check file extension
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   ::tolower);

    // select the file implementation
    if (extension == "safetensors") {
#ifdef DALOTIA_WITH_SAFETENSORS_CPP
        return new SafetensorsFile(filename);
#else   // DALOTIA_WITH_SAFETENSORS_CPP
        throw std::runtime_error("Safetensors support not enabled");
#endif  // DALOTIA_WITH_SAFETENSORS_CPP
    } else {
        throw std::runtime_error("Unsupported file extension: ." + extension);
    }
    return nullptr;
}

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
    auto dalotia_file = std::unique_ptr<TensorFile>(make_tensor_file(filename));
    const int *permutation_ptr = nullptr;
    if (!permutation.empty()) {
        permutation_ptr = permutation.data();
    }
    auto long_extents =
        dalotia_file->get_tensor_extents(tensor_name, permutation_ptr);
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
    dalotia_file->load_tensor_dense(
        tensor_name, weight_format, ordering,
        reinterpret_cast<std::byte *>(tensor.data()), permutation_ptr);
    return std::make_pair(true_extents, tensor);
}

// TODO same for sparse

// TODO allow md-range sub-tensor requests

}  // namespace dalotia

// C / Fortran interface

// file class made visible by a c struct, cf.
// https://isocpp.org/wiki/faq/mixing-c-and-cpp
typedef struct DalotiaTensorFile DalotiaTensorFile;

extern "C" DalotiaTensorFile *open_file(const char *filename) {
    return reinterpret_cast<DalotiaTensorFile *>(
        dalotia::make_tensor_file(std::string(filename)));
}

extern "C" void close_file(DalotiaTensorFile *file) {
    delete reinterpret_cast<dalotia::TensorFile *>(file);
}

extern "C" bool is_sparse(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->is_sparse(
        tensor_name);
}

extern "C" int get_num_dimensions(DalotiaTensorFile *file,
                                  const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->get_num_dimensions(
        tensor_name);
}

extern "C" int get_num_tensor_elements(DalotiaTensorFile *file,
                                       const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)
        ->get_num_tensor_elements(tensor_name);
}

extern "C" int get_nnz(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->get_nnz(tensor_name);
}

extern "C" int get_tensor_extents(DalotiaTensorFile *file,
                                  const char *tensor_name, int *extents) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    int num_dimensions = dalotia_file->get_num_dimensions(tensor_name);

    std::array<int, 10> extents_array =
        dalotia_file->get_tensor_extents(tensor_name);

    std::copy(extents_array.begin(), extents_array.end(), extents);
    return num_dimensions;
}

extern "C" int get_sparse_tensor_extents(DalotiaTensorFile *file,
                                         const char *tensor_name, int *extents,
                                         dalotia_SparseFormat format) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    int num_dimensions = dalotia_file->get_num_dimensions(tensor_name);
    if (format == dalotia_SparseFormat::dalotia_CSR) {
        std::array<int, 10> extents_array =
            dalotia_file->get_sparse_tensor_extents(
                tensor_name, dalotia_SparseFormat::dalotia_CSR);
        assert(extents_array[0] == dalotia_file->get_nnz(tensor_name));
        std::copy(extents_array.begin(), extents_array.end(), extents);
    } else {
        assert(false);
        return -1;
    }
    return num_dimensions;
}

extern "C" int load_tensor_dense(DalotiaTensorFile *file,
                                 const char *tensor_name, char *tensor,
                                 dalotia_WeightFormat format,
                                 dalotia_Ordering ordering) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    auto byte_tensor = reinterpret_cast<std::byte *>(tensor);
    dalotia_file->load_tensor_dense(tensor_name, format, ordering, byte_tensor);
    return 0;
}

extern "C" void load_tensor_dense_with_permutation(
    DalotiaTensorFile *file, const char *tensor_name, char *tensor,
    dalotia_WeightFormat format, dalotia_Ordering ordering,
    const int
        *permutation) { /* ... same as above, but with added argument... */ }

// TODO with named tensors?

extern "C" void load_tensor_sparse(DalotiaTensorFile *file,
                                   const char *tensor_name, char *values,
                                   int *first_indices, int *second_indices,
                                   dalotia_SparseFormat format,
                                   dalotia_WeightFormat weightFormat,
                                   dalotia_Ordering ordering) {
    auto byte_tensor = reinterpret_cast<std::byte *>(values);
    if (format == dalotia_SparseFormat::dalotia_CSR &&
        weightFormat == dalotia_WeightFormat::dalotia_float_32 &&
        ordering == dalotia_Ordering::dalotia_C_ordering) {
        return reinterpret_cast<dalotia::TensorFile *>(file)
            ->load_tensor_sparse(tensor_name, dalotia_SparseFormat::dalotia_CSR,
                                 dalotia_WeightFormat::dalotia_float_32,
                                 dalotia_Ordering::dalotia_C_ordering,
                                 byte_tensor, first_indices, second_indices);
    } else {
        assert(false);
    }
}
// TODO ...also with permutation and named tensors...
