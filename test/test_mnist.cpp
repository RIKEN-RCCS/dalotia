#include <cassert>

#include "../dalotia.hpp"
#include "../safetensors_file.hpp"

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

std::pair<std::pmr::vector<float>, std::pmr::vector<float>> test_load(
    std::string filename, std::string layer_name) {
    std::string tensor_name_weight = layer_name + ".weight";
    std::string tensor_name_bias = layer_name + ".bias";
    const dalotia_Ordering ordering = dalotia_Ordering::dalotia_C_ordering;
    constexpr dalotia_WeightFormat weightFormat =
        dalotia_WeightFormat::dalotia_float_32;
    // unpermuted for now
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
    } else if (layer_name == "fc1") {
        assert(tensor_weight_cpp.size() == 784 * 10);
        assert(extents_weight[0] == 10);
        assert(extents_weight[1] == 784);
        assert(extents_weight.size() == 2);
        assert(tensor_bias_cpp.size() == 10);
        assert(extents_bias[0] == 10);
        assert(extents_bias.size() == 1);
    } else {
        assert(false);
    }
    return std::make_pair(tensor_weight_cpp, tensor_bias_cpp);
}

void test_inference(std::string filename) {
    auto conv1 = test_load(filename, "conv1");
    auto conv2 = test_load(filename, "conv2");
    auto fc1 = test_load(filename, "fc1");

    // TODO
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
    test_inference(filename);
    std::cout << "test_mnist succeded" << std::endl;
    return 0;
}