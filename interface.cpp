#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <memory>
#include <string>
#include <vector>

// C++ library code

enum dalotia_SparseFormat // cannot be scoped to allow for C interface
{
    CSR,
    COO
}; //? compressed formats for d > 2? -> NO
//  pytorch uses M+K, but M is always 2, K is dense
//  onnx also has only 2d sparse tensors, same for tensorflow
//  compressed-tensors (recent, based on safetensors): only bitmask compression, but for arbitrary dimensions

enum dalotia_WeightFormat
{
    dalotia_float_16,
    dalotia_float_8,
    dalotia_bfloat16,
    dalotia_int2,
};

enum dalotia_Ordering
{
    dalotia_C_ordering,       // row-major: last index is most contiguous
    dalotia_Fortran_ordering, // column-major: first index is most contiguous
};

namespace dalotia
{
    //? better pass around file pointer to opened file for efficiency? instead of every function taking filename
    // then could have a class in C++ that opens the file and closes it in the destructor
    // class TensorFile {...};
    // and c-like functions
    // TensorFile* open_file(const char* filename);
    // // and
    // void close_file(TensorFile* file);
    // made visible by a c struct, cf. https://isocpp.org/wiki/faq/mixing-c-and-cpp

    bool is_sparse(std::string filename, std::string tensor_name)
    {
        // This function will (lazily) read the file and return true if the tensor is sparse
        return false;
    }

    size_t get_num_dimensions(std::string filename, std::string tensor_name)
    {
        // This function will (lazily) read the file and return the number of dimensions
        return 3;
    }

    std::array<int, 10> get_tensor_extents(std::string filename, std::string tensor_name = "") //? have the maximum number of dimensions = 10?
    {
        // This function will (lazily) read the file and return the tensor extents, passing -1 for "unused" dimensions
        return {
            5,
            4,
            3,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        };
    }

    size_t get_num_tensor_elements(std::string filename, std::string tensor_name)
    {
        // ?

        auto long_extents = get_tensor_extents(filename, tensor_name);
        // shorten extents to non -1
        auto num_nonzero = long_extents.size() - std::count(long_extents.begin(), long_extents.end(), -1);
        std::pmr::vector<int> extents(long_extents.begin(), long_extents.begin() + num_nonzero);
        auto total_size = std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<size_t>());
        return total_size;
    }

    size_t get_nnz(std::string filename, std::string tensor_name)
    {
        // This function will read the file and return the number of non-zero elements
        // ? may take a while for dense tensors, only allow for sparse?
        return 12;
    }

    template <dalotia_SparseFormat format>
    std::array<int, 10> get_sparse_tensor_extents(std::string filename, std::string tensor_name)
    {
        // This function will (lazily) read the file and return the tensor extents
        return {
            8,
            8,
            9,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        };
    }

    void load_tensor_dense(std::string filename, std::string tensor_name, char *tensor, const int *permutation = nullptr)
    {
        // This function will read the whole file and load the tensor, optionally transposing it according to the permutation
        for (size_t i = 0; i < dalotia::get_num_tensor_elements(filename, tensor_name); i++)
        {
            tensor[i] = i;
        }
    }

    template <dalotia_SparseFormat format>
    void load_tensor_sparse(std::string filename, std::string tensor_name, char *values, int *first_indices, int *second_indices)
    {
        // This function will read the whole file and load the tensor into the three arrays
        for (int i = 0; i < 10; i++)
        {
            values[i] = i;
            first_indices[i] = i;
            second_indices[i] = i;
        }
    }

    // C++17 version -- pmr vector types can accept different allocators -- will not compile on Fugaku...
    //? more memory interface than that? detect if CUDA device pointer through unified access... how about other devices?
    template<typename value_type = char, dalotia_Ordering = dalotia_C_ordering> //? or have no defaults?
    [[nodiscard]] std::pair<std::pmr::vector<int>, std::pmr::vector<value_type>> load_tensor_dense(std::string filename, std::string tensor_name, const std::pmr::vector<int> &permutation = {})
    {
        auto long_extents = get_tensor_extents(filename, tensor_name);
        // shorten extents to nonzeros
        auto num_nonzero = long_extents.size() - std::count(long_extents.begin(), long_extents.end(), -1);
        std::pmr::vector<int> true_extents(long_extents.begin(), long_extents.begin() + num_nonzero);
        auto total_size = std::accumulate(true_extents.begin(), true_extents.end(), 1, std::multiplies<size_t>());
        std::pmr::vector<char> tensor;
        tensor.resize(total_size);
        load_tensor_dense(filename.c_str(), tensor_name.c_str(), tensor.data(), permutation.data());
        return std::make_pair(true_extents, tensor);
    }

    // TODO allow md-range sub-tensor requests
}

// C / Fortran interface
//? can float be "templated" here somehow for other precision formats? -> pass enum, get bytes out, cf. llama.cpp

extern "C" bool is_sparse(const char *filename, const char *tensor_name)
{
    return dalotia::is_sparse(filename, tensor_name);
}

extern "C" int get_num_dimensions(const char *filename, const char *tensor_name)
{
    return dalotia::get_num_dimensions(filename, tensor_name);
}

extern "C" int get_num_tensor_elements(const char *filename, const char *tensor_name)
{
    return dalotia::get_num_tensor_elements(filename, tensor_name);
}

extern "C" int get_nnz(const char *filename, const char *tensor_name)
{
    return dalotia::get_nnz(filename, tensor_name);
}

extern "C" int get_tensor_extents(const char *filename, const char *tensor_name, int *extents)
{
    int num_dimensions = dalotia::get_num_dimensions(filename, tensor_name);
    std::array<int, 10> extents_array = dalotia::get_tensor_extents(filename, tensor_name);
    std::copy(extents_array.begin(), extents_array.end(), extents);
    return num_dimensions;
}

extern "C" int get_sparse_tensor_extents(const char *filename, const char *tensor_name, int *extents, dalotia_SparseFormat format)
{
    int num_dimensions = dalotia::get_num_dimensions(filename, tensor_name);
    if (format == dalotia_SparseFormat::CSR)
    {
        std::array<int, 10> extents_array = dalotia::get_sparse_tensor_extents<dalotia_SparseFormat::CSR>(filename, tensor_name);
        assert(extents_array[0] == dalotia::get_nnz(filename, tensor_name));
        std::copy(extents_array.begin(), extents_array.end(), extents);
    }
    return num_dimensions;
}

extern "C" void load_tensor_dense(const char *filename, const char *tensor_name, char *tensor)
{
    return dalotia::load_tensor_dense(filename, tensor_name, tensor);
}

extern "C" void load_tensor_dense_with_permutation(const char *filename, const char *tensor_name, char *tensor, const int *permutation)
{
    return dalotia::load_tensor_dense(filename, tensor_name, tensor, permutation);
}

extern "C" void load_tensor_sparse(const char *filename, const char *tensor_name, char *values, int *first_indices, int *second_indices, dalotia_SparseFormat format)
{
    if (format == dalotia_SparseFormat::CSR)
    {
        return dalotia::load_tensor_sparse<dalotia_SparseFormat::CSR>(filename, tensor_name, values, first_indices, second_indices);
    }
}

// application code

int main(int argc, char *argv[])
{
    bool tensor_is_sparse = is_sparse("data.txt", "dense_layer_12"); //...repeat later
    char *tensor;

    if (~tensor_is_sparse)
    {
        char *filename = "data.txt";
        char *tensor_name = "dense_layer_12";

        // get the tensor extents
        int extents[10]; // ? or call get_num_dimensions before?
                         // file formats: gguf, safetensors, onnx? channel orders? gguf with quantized ops?
                         // -> look at dnnl, darknet, safetensors-cpp, tinyml? torch: named dimensions
                         // tensor name data format data shape offsets
        int num_dimensions = get_tensor_extents(filename, tensor_name, extents);

        // calculate the total number of elements
        int total_size = 1;
        for (int i = 0; i < 10; i++)
        {
            if (extents[i] == -1)
            {
                assert(i > 0);
                break;
            }
            total_size *= extents[i];
        }

        assert(total_size == get_num_tensor_elements(filename, tensor_name));

        // I want to store the tensor as a very long array
        // allocate memory for the tensor
        tensor = (char *)malloc(sizeof(float) * total_size);

        // load the tensor
        int permutation[2, 0, 1];
        load_tensor_dense_with_permutation(filename, tensor_name, tensor, permutation);
    }
    else
    {
        char *filename = "data.txt";
        char *tensor_name = "dropout_layer_14";

        // get the tensor extents
        int extents[10];
        int num_dimensions = get_tensor_extents(filename, tensor_name, extents);

        int sparse_extents[10];
        get_sparse_tensor_extents(filename, tensor_name, sparse_extents, CSR);

        // I want to store the tensor as compressed sparse row
        char *values = reinterpret_cast<char *>(new float[sparse_extents[0]]); // blah blah malloc...
        int *first_indices = new int[sparse_extents[1]];
        int *second_indices = new int[sparse_extents[2]];
        load_tensor_sparse(filename, tensor_name, values, first_indices, second_indices, CSR);
    }

    //... // do something with the tensor
    // everything alex calls a runtime = possibly jit compiled

    // example: nicam ai, genesis, ...reconstruct?
    // run here with cuDNN, or BLIS, or...

    return 0;
}
