#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>  //TODO remove
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string>
#include <vector>

#include "dalotia_formats.hpp"
#include "tensor_file.hpp"

#ifdef DALOTIA_WITH_SAFETENSORS_CPP
#include "safetensors_file.hpp"
#endif

namespace dalotia {
// factory function for the file, selected by file extension and
// available implementations
TensorFile *make_tensor_file(std::string filename) {
    std::cout << "make_tensor_file " << filename << std::endl;
    // make sure the file exists
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("File " + filename + " does not exist");
    }

    // check file extension
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   ::tolower);

    // select the file implementation
    if (extension == "txt") {
        // TODO remove and throw errors in implementation, only for testing
        return new TensorFile(filename);
    } else if (extension == "safetensors") {
#ifdef DALOTIA_WITH_SAFETENSORS_CPP
        return new Safetensors(filename);
#else   // DALOTIA_WITH_SAFETENSORS_CPP
        throw std::runtime_error("Safetensors support not enabled");
#endif  // DALOTIA_WITH_SAFETENSORS_CPP
    } else {
        throw std::runtime_error("Unsupported file extension: ." + extension);
    }
}
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
    dalotia_file->load_tensor_dense(tensor_name, dalotia_int_2,
                                    dalotia_C_ordering, byte_tensor);
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
