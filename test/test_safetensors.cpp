#include "../dalotia.hpp"
#include "../safetensors_file.hpp"

void test_simple_linear_load() {
    // the C version
    char *filename = "../data/model.safetensors";
    char *tensor_name = "embedding";
    DalotiaTensorFile *file = open_file(filename);
    bool tensor_is_sparse = is_sparse(file, tensor_name);
    assert(!tensor_is_sparse);
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_64;
    dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;

    int extents[10];
    int num_dimensions = get_tensor_extents(file, tensor_name, extents);
    assert(num_dimensions == 3);
    assert(extents[0] == 3);
    assert(extents[1] == 4);
    assert(extents[2] == 5);
    assert(extents[3] == -1);

    int total_size = 1;
    for (int i = 0; i < 10; i++) {
        if (extents[i] == -1) {
            assert(i > 0);
            break;
        }
        total_size *= extents[i];
    }
    assert(total_size == 60);
    assert(total_size == get_num_tensor_elements(file, tensor_name));

    auto tensor = (char *)malloc(dalotia::sizeof_weight_format<weightFormat>() *
                                 total_size);

    load_tensor_dense(file, tensor_name, tensor, weightFormat, ordering);

    auto double_tensor = reinterpret_cast<double *>(tensor);
    for (int i = 0; i < total_size; i++) {
        assert(double_tensor[i] == i);
    }
}

void test_permutation() {
    int permutation[3] = {0, 1, 2};
    auto final_permutation =
        dalotia::final_c_permutation_from_permutation_and_order(
            permutation, dalotia_Ordering::dalotia_C_ordering, 3);
    assert(final_permutation.empty());

    permutation[0] = 2;
    permutation[2] = 0;
    final_permutation = dalotia::final_c_permutation_from_permutation_and_order(
        permutation, dalotia_Ordering::dalotia_F_ordering, 3);
    assert(final_permutation.empty());

    int permutation2[3] = {1, 0, 2};
    final_permutation = dalotia::final_c_permutation_from_permutation_and_order(
        permutation2, dalotia_Ordering::dalotia_F_ordering, 3);
    assert(final_permutation.size() == 3);
    assert(final_permutation[0] == 2);
    assert(final_permutation[1] == 0);
    assert(final_permutation[2] == 1);

    final_permutation = dalotia::final_c_permutation_from_permutation_and_order(
        permutation2, dalotia_Ordering::dalotia_C_ordering, 3);
    assert(final_permutation.size() == 3);
    assert(final_permutation[0] == 1);
    assert(final_permutation[1] == 0);
    assert(final_permutation[2] == 2);
}

void test_permuted_load() {
    // the C++ 17 version
    std::string filename = "../data/model.safetensors";
    std::string tensor_name = "embedding_firstchanged";
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_64;
    dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;
    {
        // first test linear load
        auto [extents, tensor_cpp] = dalotia::load_tensor_dense<double>(
            filename, tensor_name, weightFormat, ordering);
        assert(extents.size() == 3);
        assert(extents[0] == 4);
        assert(extents[1] == 3);
        assert(extents[2] == 5);
        assert(tensor_cpp.size() == 60);
    }
    {
        // then with permutation
        auto permutation = std::pmr::vector<int>{1, 0, 2};

        auto [extents, tensor_cpp] = dalotia::load_tensor_dense<double>(
            filename, tensor_name, weightFormat, ordering,
            std::pmr::polymorphic_allocator<std::byte>(), permutation);
        assert(extents.size() == 3);
        assert(extents[0] == 3);
        assert(extents[1] == 4);
        assert(extents[2] == 5);
        assert(tensor_cpp.size() == 60);
        for (int i = 0; i < 60; i++) {
            assert(tensor_cpp[i] == i);
        }
    }
}

int main(int, char **) {
    test_simple_linear_load();
    test_permutation();
    test_permuted_load();
    return 0;
}