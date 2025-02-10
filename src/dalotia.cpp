#include "dalotia.hpp"
#if __cpp_lib_filesystem
#include <filesystem>
namespace dalotia {
using file_exists = std::filesystem::exists;
}
#else   // __cpp_lib_filesystem
namespace dalotia {
bool file_exists(const std::string &name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}
}  // namespace dalotia
#endif  // __cpp_lib_filesystem
#include <iostream>

#include "dalotia.h"

namespace dalotia {

// factory function for the file, selected by file extension and
// available implementations
TensorFile *make_tensor_file(const std::string &filename) {
    // make sure the file exists
    if (!dalotia::file_exists(filename)) {
        throw std::runtime_error("dalotia make_tensor_file: File " + filename +
                                 " does not exist");
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

}  // namespace dalotia

DalotiaTensorFile *dalotia_open_file(const char *filename) {
    return reinterpret_cast<DalotiaTensorFile *>(
        dalotia::make_tensor_file(std::string(filename)));
}

void dalotia_close_file(DalotiaTensorFile *file) {
    delete reinterpret_cast<dalotia::TensorFile *>(file);
}

int dalotia_sizeof_weight_format(dalotia_WeightFormat format) {
    return dalotia::sizeof_weight_format(format);
}

bool dalotia_is_sparse(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->is_sparse(
        tensor_name);
}

int dalotia_get_num_tensors(DalotiaTensorFile *file) {
    return static_cast<int>(reinterpret_cast<dalotia::TensorFile *>(file)
                                ->get_tensor_names()
                                .size());
}

int dalotia_get_tensor_name(DalotiaTensorFile *file, int index, char *name) {
    auto tensor_names =
        reinterpret_cast<dalotia::TensorFile *>(file)->get_tensor_names();
    const std::string &tensor_name = tensor_names.at(index);
    std::copy(tensor_name.begin(), tensor_name.end(), name);
    name[tensor_name.size()] = '\0';  // zero-terminate
    // return the length of the string
    // TODO find out if safetensors specifies a maximum length??
    // for now, assume 255
    assert(tensor_name.size() < 256);
    return static_cast<int>(tensor_name.size());
}

int dalotia_get_num_dimensions(DalotiaTensorFile *file,
                               const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->get_num_dimensions(
        tensor_name);
}

int dalotia_get_num_tensor_elements(DalotiaTensorFile *file,
                                    const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)
        ->get_num_tensor_elements(tensor_name);
}

int dalotia_get_nnz(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->get_nnz(tensor_name);
}

int dalotia_get_tensor_extents(DalotiaTensorFile *file, const char *tensor_name,
                               int *extents) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    int num_dimensions = dalotia_file->get_num_dimensions(tensor_name);
    {
        auto extents_array = dalotia_file->get_tensor_extents(
            tensor_name);  // TODO make this a vector?

        std::copy(extents_array.begin(), extents_array.begin() + num_dimensions,
                  extents);
    }
    return num_dimensions;
}

int dalotia_get_sparse_tensor_extents(DalotiaTensorFile *file,
                                      const char *tensor_name, int *extents,
                                      dalotia_SparseFormat format) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    int num_dimensions = dalotia_file->get_num_dimensions(tensor_name);
    if (format == dalotia_SparseFormat::dalotia_CSR) {
        std::array<int, 10> extents_array =
            dalotia_file->get_sparse_tensor_extents(
                tensor_name, dalotia_SparseFormat::dalotia_CSR);
        assert(static_cast<size_t>(extents_array[0]) ==
               dalotia_file->get_nnz(tensor_name));
        std::copy(extents_array.begin(), extents_array.end(), extents);
    } else {
        assert(false);
        return -1;
    }
    return num_dimensions;
}

int dalotia_load_tensor_dense(DalotiaTensorFile *file, const char *tensor_name,
                              char *tensor, dalotia_WeightFormat format,
                              dalotia_Ordering ordering) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    auto byte_tensor = reinterpret_cast<dalotia_byte *>(tensor);
    try {
        dalotia_file->load_tensor_dense(tensor_name, format, ordering,
                                        byte_tensor);
    } catch (const std::exception &e) {
        std::cerr << "dalotia_load_tensor_dense: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int dalotia_load_tensor_dense_with_permutation(DalotiaTensorFile *file,
                                               const char *tensor_name,
                                               char *tensor,
                                               dalotia_WeightFormat format,
                                               dalotia_Ordering ordering,
                                               const int *permutation) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    auto byte_tensor = reinterpret_cast<dalotia_byte *>(tensor);
    try {
        // copy permutation to vector
        auto num_dimensions = dalotia_file->get_num_dimensions(tensor_name);
        std::vector<int> permutation_vector(permutation,
                                            permutation + num_dimensions);
        dalotia_file->load_tensor_dense(tensor_name, format, ordering,
                                        byte_tensor, permutation_vector);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "dalotia_load_tensor_dense_with_permutation: " << e.what()
                  << std::endl;
        return -1;
    }
}

// TODO with named tensors?

int dalotia_load_tensor_sparse(DalotiaTensorFile *file, const char *tensor_name,
                               char *values, int *first_indices,
                               int *second_indices, dalotia_SparseFormat format,
                               dalotia_WeightFormat weightFormat,
                               dalotia_Ordering ordering) {
    auto byte_tensor = reinterpret_cast<dalotia_byte *>(values);
    try {
        if (format == dalotia_SparseFormat::dalotia_CSR &&
            weightFormat == dalotia_WeightFormat::dalotia_float_32 &&
            ordering == dalotia_Ordering::dalotia_C_ordering) {
            reinterpret_cast<dalotia::TensorFile *>(file)->load_tensor_sparse(
                tensor_name, dalotia_SparseFormat::dalotia_CSR,
                dalotia_WeightFormat::dalotia_float_32,
                dalotia_Ordering::dalotia_C_ordering, byte_tensor,
                first_indices, second_indices);
        } else {
            throw std::runtime_error(
                "dalotia_load_tensor_sparse: unsupported format combination");
        }
    } catch (const std::exception &e) {
        std::cerr << "dalotia_load_tensor_sparse: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
// TODO ...also with permutation and named tensors...
