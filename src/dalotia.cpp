#include "dalotia.hpp"

#include "dalotia.h"

namespace dalotia {
// factory function for the file, selected by file extension and
// available implementations
TensorFile *make_tensor_file(std::string filename) {
    // make sure the file exists
    if (!std::filesystem::exists(filename)) {
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

DalotiaTensorFile *open_file(const char *filename) {
    return reinterpret_cast<DalotiaTensorFile *>(
        dalotia::make_tensor_file(std::string(filename)));
}

void close_file(DalotiaTensorFile *file) {
    delete reinterpret_cast<dalotia::TensorFile *>(file);
}

bool is_sparse(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->is_sparse(
        tensor_name);
}

int get_num_tensors(DalotiaTensorFile *file) {
    return static_cast<int>(reinterpret_cast<dalotia::TensorFile *>(file)
                                ->get_tensor_names()
                                .size());
}

int get_tensor_name(DalotiaTensorFile *file, int index, char *name) {
    auto tensor_names =
        reinterpret_cast<dalotia::TensorFile *>(file)->get_tensor_names();
    const std::string &tensor_name = tensor_names.at(index);
    std::copy(tensor_name.begin(), tensor_name.end(), name);
    name[tensor_name.size()] = '\0'; // zero-terminate
    // return the length of the string
    //TODO find out if safetensors specifies a maximum length??
    return static_cast<int>(tensor_name.size());
}

int get_num_dimensions(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->get_num_dimensions(
        tensor_name);
}

int get_num_tensor_elements(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)
        ->get_num_tensor_elements(tensor_name);
}

int get_nnz(DalotiaTensorFile *file, const char *tensor_name) {
    return reinterpret_cast<dalotia::TensorFile *>(file)->get_nnz(tensor_name);
}

int get_tensor_extents(DalotiaTensorFile *file, const char *tensor_name,
                       int *extents) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    int num_dimensions = dalotia_file->get_num_dimensions(tensor_name);

    std::array<int, 10> extents_array =
        dalotia_file->get_tensor_extents(tensor_name);

    std::copy(extents_array.begin(), extents_array.end(), extents);
    return num_dimensions;
}

int get_sparse_tensor_extents(DalotiaTensorFile *file, const char *tensor_name,
                              int *extents, dalotia_SparseFormat format) {
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

int load_tensor_dense(DalotiaTensorFile *file, const char *tensor_name,
                      char *tensor, dalotia_WeightFormat format,
                      dalotia_Ordering ordering) {
    auto dalotia_file = reinterpret_cast<dalotia::TensorFile *>(file);
    auto byte_tensor = reinterpret_cast<std::byte *>(tensor);
    dalotia_file->load_tensor_dense(tensor_name, format, ordering, byte_tensor);
    return 0;
}

void load_tensor_dense_with_permutation(
    DalotiaTensorFile *file, const char *tensor_name, char *tensor,
    dalotia_WeightFormat format, dalotia_Ordering ordering,
    const int
        *permutation) { /* ... same as above, but with added argument... */ }

// TODO with named tensors?

void load_tensor_sparse(DalotiaTensorFile *file, const char *tensor_name,
                        char *values, int *first_indices, int *second_indices,
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
