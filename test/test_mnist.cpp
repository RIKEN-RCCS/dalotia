#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

#include "dalotia.hpp"
#include "dalotia_safetensors_file.hpp"

void test_get_tensor_names(std::string filename) {
    auto dalotia_file = std::unique_ptr<dalotia::TensorFile>(
        dalotia::make_tensor_file(filename));
    auto tensor_names = dalotia_file->get_tensor_names();
    assert(tensor_names.size() == 6);
    assert(tensor_names[0] == "conv1.bias");
    assert(tensor_names[1] == "conv1.weight");
    assert(tensor_names[2] == "conv2.bias");
    assert(tensor_names[3] == "conv2.weight");
    assert(tensor_names[4] == "fc1.bias");
    assert(tensor_names[5] == "fc1.weight");
}

void assert_close(float a, float b) {
    if (std::abs(a - b) > 1e-4) {
        throw std::runtime_error("assert_close: expected " + std::to_string(b) +
                                 " but got " + std::to_string(a));
    }
}

template <int dim>
std::function<int(std::array<int, dim>)> get_tensor_indexer(
    const std::array<int, dim> &extents) {
    std::array<int, dim> strides;
    // std::exclusive_scan(extents.rbegin(), extents.rend(), strides.rbegin(),
    // 1,
    //                     std::multiplies<int>());
    strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * extents[i + 1];
    }
    const std::array<int, dim> const_strides = strides;
    return [const_strides, &extents](std::array<int, dim> indices) {
#ifndef NDEBUG
        for (int i = 0; i < dim; ++i) {
            if (indices[i] < 0 || indices[i] >= extents[i]) {
                std::cerr << "index " << i << " out of bounds: " << indices[i]
                          << " not in [0, " << extents[i] << ")" << std::endl;
            }
            assert(indices[i] >= 0 && indices[i] < extents[i]);
        }
#endif  // NDEBUG
        return std::inner_product(indices.begin(), indices.end(),
                                  const_strides.begin(), 0);
    };
}

std::pair<dalotia::vector<float>, dalotia::vector<float>> test_load(
    std::string filename, std::string layer_name) {
    std::string tensor_name_weight = layer_name + ".weight";
    std::string tensor_name_bias = layer_name + ".bias";
    const dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_32;
    auto [extents_weight, tensor_weight_cpp] =
        dalotia::load_tensor_dense<float>(filename, tensor_name_weight,
                                          weightFormat, ordering);
    auto [extents_bias, tensor_bias_cpp] = dalotia::load_tensor_dense<float>(
        filename, tensor_name_bias, weightFormat, ordering);
    if (layer_name == "conv1") {
        assert(tensor_weight_cpp.size() == 72);
        assert(extents_weight[0] == 8);
        assert(extents_weight[1] == 1);
        assert(extents_weight[2] == 3);
        assert(extents_weight[3] == 3);
        assert(extents_weight.size() == 4);
        assert(tensor_bias_cpp.size() == 8);
        assert(extents_bias[0] == 8);
        assert(extents_bias.size() == 1);
        assert_close(tensor_weight_cpp[0], 0.944823);
        assert_close(tensor_weight_cpp[1], 1.25045);
        assert_close(tensor_weight_cpp.back(), 0.211111);
        assert_close(tensor_bias_cpp[0], 0.1796);
        assert_close(tensor_bias_cpp[7], 0.6550);

        std::array<int, 4> extents_weight_array;
        std::copy(extents_weight.begin(), extents_weight.end(),
                  extents_weight_array.begin());
        const auto plain_indexer = get_tensor_indexer<4>(extents_weight_array);

        // check permutations
        dalotia::vector<int> permutation = {0, 1, 2, 3};
        do {
            auto [extents_weight_perm, tensor_weight_cpp_perm] =
                dalotia::load_tensor_dense<float>(filename, tensor_name_weight,
                                                  weightFormat, ordering,
                                                  permutation);
            assert(extents_weight_perm.size() == extents_weight.size());
            assert(tensor_weight_cpp_perm.size() == tensor_weight_cpp.size());
            assert(extents_weight_perm[0] == extents_weight[permutation[0]]);
            assert(extents_weight_perm[1] == extents_weight[permutation[1]]);
            assert(extents_weight_perm[2] == extents_weight[permutation[2]]);
            assert(extents_weight_perm[3] == extents_weight[permutation[3]]);
            std::array<int, 4> extents_weight_perm_array;
            std::copy(extents_weight_perm.begin(), extents_weight_perm.end(),
                      extents_weight_perm_array.begin());
            const auto permuted_indexer =
                get_tensor_indexer<4>(extents_weight_perm_array);

            std::array<int, 4> indices;
            for (int i = 0; i < extents_weight[permutation[0]]; ++i) {
                for (int j = 0; j < extents_weight[permutation[1]]; ++j) {
                    for (int k = 0; k < extents_weight[permutation[2]]; ++k) {
                        for (int l = 0; l < extents_weight[permutation[3]];
                             ++l) {
                            indices[permutation[0]] = i;
                            indices[permutation[1]] = j;
                            indices[permutation[2]] = k;
                            indices[permutation[3]] = l;
                            const int index = plain_indexer(indices);
                            const int index_perm =
                                permuted_indexer({i, j, k, l});
                            assert(tensor_weight_cpp[index] ==
                                   tensor_weight_cpp_perm[index_perm]);
                        }
                    }
                }
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    } else if (layer_name == "conv2") {
        assert(tensor_weight_cpp.size() == 1152);
        assert(extents_weight[0] == 16);
        assert(extents_weight[1] == 8);
        assert(extents_weight[2] == 3);
        assert(extents_weight[3] == 3);
        assert(extents_weight.size() == 4);
        assert(tensor_bias_cpp.size() == 16);
        assert(extents_bias[0] == 16);
        assert(extents_bias.size() == 1);
        assert_close(tensor_weight_cpp[0], -0.79839);
        assert_close(tensor_weight_cpp[1], -1.3640);
        assert_close(tensor_weight_cpp.back(), 0.32985);
        assert_close(tensor_bias_cpp[0], -0.2460);
        assert_close(tensor_bias_cpp[15], -0.3158);
    } else if (layer_name == "fc1") {
        assert(tensor_weight_cpp.size() == 784 * 10);
        assert(extents_weight[0] == 10);
        assert(extents_weight[1] == 784);
        assert(extents_weight.size() == 2);
        assert(tensor_bias_cpp.size() == 10);
        assert(extents_bias[0] == 10);
        assert(extents_bias.size() == 1);
        assert_close(tensor_weight_cpp[0], 0.3420);
        assert_close(tensor_weight_cpp[1], 0.7881);
        assert_close(tensor_weight_cpp.back(), 0.3264);
        assert_close(tensor_bias_cpp[0], 0.3484);
        assert_close(tensor_bias_cpp[9], -0.2224);
    } else {
        assert(false);
    }
    return std::make_pair(tensor_weight_cpp, tensor_bias_cpp);
}

int main(int, char **) {
    // the model used here is generated as in
    // https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
    // (but trained for 100 epochs)
    // and then saved with safetensors.torch.save_model(model,
    // "model.safetensors")
    std::string filename = "../data/model-mnist.safetensors";

    test_get_tensor_names(filename);
    test_load(filename, "conv1");
    test_load(filename, "conv2");
    test_load(filename, "fc1");
    std::cout << "test_mnist succeded" << std::endl;
    return 0;
}