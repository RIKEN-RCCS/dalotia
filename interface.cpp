#include <stdio.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string>
#include <vector>

// for safetensors, should definitely use
// https://github.com/syoyo/safetensors-cpp

// C++ library code

enum dalotia_SparseFormat  // cannot be scoped to allow for C interface
{
    dalotia_CSR,
    dalotia_COO
};  //? compressed formats for d > 2? -> NO common ones
//  pytorch uses M+K, but M is always 2, K is dense
//  onnx also has only 2d sparse tensors, same for tensorflow
//  compressed-tensors (recent, based on safetensors): only bitmask compression,
//  but for arbitrary dimensions
//  ALTO https://github.com/IntelLabs/ALTO /
//  https://dl.acm.org/doi/abs/10.1145/3447818.3461703 could be interesting

enum dalotia_WeightFormat {
    dalotia_float_32,
    dalotia_float_16,
    dalotia_float_8,
    dalotia_bfloat_16,
    dalotia_int_2,
};

enum dalotia_Ordering {
    dalotia_C_ordering,  // row-major: last index is most contiguous
    dalotia_F_ordering,  // column-major: first index is most contiguous
};

namespace dalotia {

template <dalotia_WeightFormat format>
constexpr uint8_t sizeof_weight_format() {
    if constexpr (format == dalotia_float_32) {
        return 4;
    } else if constexpr (format == dalotia_float_16) {
        return 2;
    } else if constexpr (format == dalotia_float_8) {
        return 1;
    } else if constexpr (format == dalotia_bfloat_16) {
        return 2;
    } else if constexpr (format == dalotia_int_2) {
        return 1;  // TODO a bit unhappy with this one
    }
}

//? better pass around file pointer to opened file for efficiency? instead of
// every function taking filename
// then could have a class in C++ that opens the file and closes it in the
// destructor
class TensorFile {
   public:
    TensorFile(std::string filename) {
        this->file_ = std::fopen(filename.c_str(), "rb");
    }
    ~TensorFile() { std::fclose(this->file_); }

    bool is_sparse(std::string tensor_name) {
        // This function will (lazily) read the file and return true if the
        // tensor is sparse
        return true;
    }

    size_t get_num_dimensions(std::string tensor_name) {
        // This function will (lazily) read the file and return the number of
        // dimensions
        return 3;
    }

    std::array<int, 10> get_tensor_extents(
        std::string tensor_name =
            "")  //? have the maximum number of dimensions = 10?
    {
        // This function will (lazily) read the file and return the tensor
        // extents, passing -1 for "unused" dimensions
        return {
            5, 4, 3, -1, -1, -1, -1, -1, -1, -1,
        };
    }

    size_t get_num_tensor_elements(std::string tensor_name) {
        // ?

        auto long_extents = this->get_tensor_extents(tensor_name);
        // shorten extents to non -1
        auto num_nonzero =
            long_extents.size() -
            std::count(long_extents.begin(), long_extents.end(), -1);
        std::pmr::vector<int> extents(long_extents.begin(),
                                      long_extents.begin() + num_nonzero);
        auto total_size = std::accumulate(extents.begin(), extents.end(), 1,
                                          std::multiplies<size_t>());
        return total_size;
    }

    size_t get_nnz(std::string tensor_name) {
        // This function will read the file and return the number of non-zero
        // elements ? may take a while for dense tensors, only allow for sparse?
        return 12;
    }

    template <dalotia_SparseFormat format>
    std::array<int, 10> get_sparse_tensor_extents(std::string tensor_name) {
        // This function will (lazily) read the file and return the tensor
        // extents
        return {
            12, 12, 13, -1, -1, -1, -1, -1, -1, -1,
        };
    }

    template <dalotia_WeightFormat weightFormat, dalotia_Ordering ordering>
    void load_tensor_dense(std::string tensor_name, std::byte *tensor,
                           const int *permutation = nullptr) {
        // This function will read the whole file and load the tensor,
        // optionally transposing it according to the permutation
        for (size_t i = 0; i < this->get_num_tensor_elements(tensor_name);
             i++) {
            tensor[i] = static_cast<std::byte>(i);  // chunk from weightFormat
        }
    }

    template <dalotia_SparseFormat sparseFormat,
              dalotia_WeightFormat weightFormat, dalotia_Ordering ordering>
    void load_tensor_sparse(std::string tensor_name, std::byte *values,
                            int *first_indices, int *second_indices) {
        // This function will read the whole file and load the tensor into the
        // three arrays
        for (int i = 0; i < 10; i++) {
            values[i] = static_cast<std::byte>(i);  // chunk from weightFormat;
            first_indices[i] = i;
            second_indices[i] = i;
        }
    }

    // no private section to allow visibility from C
    std::FILE *file_;
};

// C++17 version -> will not compile on Fugaku...
// -- pmr vector types can accept different allocators
//? more memory interface than that? detect if CUDA device pointer through
// unified access... how about other devices?
template <
    dalotia_WeightFormat weight_format, typename value_type = std::byte,
    dalotia_Ordering ordering = dalotia_C_ordering>  //? or have no defaults?
[[nodiscard]] std::pair<std::pmr::vector<int>, std::pmr::vector<value_type>>
load_tensor_dense(std::string filename, std::string tensor_name,
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
    // true_extents.assign();
    auto total_size = std::accumulate(true_extents.begin(), true_extents.end(),
                                      1, std::multiplies<size_t>());
    std::pmr::vector<value_type> tensor(allocator);
    if constexpr (std::is_same_v<value_type, std::byte>) {
    tensor.resize(total_size * sizeof_weight_format<weight_format>());
    } else {
        tensor.resize(total_size);
    }
    dalotia_file.load_tensor_dense<weight_format, ordering>(
        tensor_name, tensor.data(), permutation.data());
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
        new dalotia::TensorFile(filename));
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
    int num_dimensions =
        reinterpret_cast<dalotia::TensorFile *>(file)->get_num_dimensions(
            tensor_name);
    std::array<int, 10> extents_array =
        reinterpret_cast<dalotia::TensorFile *>(file)->get_tensor_extents(
            tensor_name);
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
            dalotia_file
                ->get_sparse_tensor_extents<dalotia_SparseFormat::dalotia_CSR>(
                    tensor_name);
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
    if (format == dalotia_float_32 && ordering == dalotia_C_ordering) {
        dalotia_file->load_tensor_dense<dalotia_float_32, dalotia_C_ordering>(
            tensor_name, byte_tensor);
    } else if (format == dalotia_float_16 && ordering == dalotia_C_ordering) {
        dalotia_file->load_tensor_dense<dalotia_float_16, dalotia_C_ordering>(
            tensor_name, byte_tensor);
    } else if (format == dalotia_float_8 && ordering == dalotia_C_ordering) {
        dalotia_file->load_tensor_dense<dalotia_float_8, dalotia_C_ordering>(
            tensor_name, byte_tensor);
    } else if (format == dalotia_bfloat_16 && ordering == dalotia_C_ordering) {
        dalotia_file->load_tensor_dense<dalotia_bfloat_16, dalotia_C_ordering>(
            tensor_name, byte_tensor);
    } else if (format == dalotia_int_2 && ordering == dalotia_C_ordering) {
        dalotia_file->load_tensor_dense<dalotia_int_2, dalotia_C_ordering>(
            tensor_name, byte_tensor);
    } else {
        return 1;
    }
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
            ->load_tensor_sparse<dalotia_SparseFormat::dalotia_CSR,
                                 dalotia_WeightFormat::dalotia_float_32,
                                 dalotia_Ordering::dalotia_C_ordering>(
                tensor_name, byte_tensor, first_indices, second_indices);
    } else {
        assert(false);
    }
}
// TODO ...also with permutation and named tensors...

// application code

int main(int argc, char *argv[]) {
    char *filename = "data.txt";
    char *tensor_name = "layer_12";
    DalotiaTensorFile *file = open_file(filename);
    bool tensor_is_sparse =
        is_sparse(file, "dense_layer_12");  //...repeat later
    char *tensor;
    dalotia_WeightFormat weightFormat = dalotia_WeightFormat::dalotia_float_32;
    dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;
    if (!tensor_is_sparse) {
        // get the tensor extents
        int extents[10];  // ? or call get_num_dimensions before?
                          // file formats: gguf, safetensors, onnx? channel
                          // orders? gguf with quantized ops?
                          // -> look at dnnl, darknet, safetensors-cpp, tinyml?
                          // torch: named dimensions tensor name data format
                          // data shape offsets
        int num_dimensions = get_tensor_extents(file, tensor_name, extents);

        // calculate the total number of elements
        int total_size = 1;
        for (int i = 0; i < 10; i++) {
            if (extents[i] == -1) {
                assert(i > 0);
                break;
            }
            total_size *= extents[i];
        }

        assert(total_size == get_num_tensor_elements(file, tensor_name));

        // I want to store the tensor as a very long array
        // allocate memory for the tensor
        tensor = (char *)malloc(sizeof(float) * total_size);

        // load the tensor
        int permutation[3] = {2, 1, 0};

        // load_tensor_dense_with_permutation(file, tensor_name, tensor,
        //                                    weightFormat, ordering,
        //                                    permutation);
        load_tensor_dense(file, tensor_name, tensor, weightFormat, ordering);
    } else {
        dalotia_SparseFormat format = dalotia_SparseFormat::dalotia_CSR;
        // get the tensor extents
        int extents[10];
        int num_dimensions = get_tensor_extents(file, tensor_name, extents);

        int sparse_extents[10];
        get_sparse_tensor_extents(file, tensor_name, sparse_extents,
                                  dalotia_CSR);

        // I want to store the tensor as compressed sparse row
        char *values = reinterpret_cast<char *>(
            new float[sparse_extents[0]]);  // blah blah malloc...
        int *first_indices = new int[sparse_extents[1]];
        int *second_indices = new int[sparse_extents[2]];
        load_tensor_sparse(file, tensor_name, values, first_indices,
                           second_indices, format, weightFormat, ordering);
    }
    close_file(file);

    // alternative: the C++17 version
    auto [extents, tensor_cpp] =
        dalotia::load_tensor_dense<dalotia_float_32>(filename, tensor_name);

    //... // do something with the tensor
    // everything alex calls a runtime = possibly jit compiled

    // example: nicam ai, genesis, ...reconstruct?
    // run here with cuDNN, or BLIS, or...

    return 0;
}
