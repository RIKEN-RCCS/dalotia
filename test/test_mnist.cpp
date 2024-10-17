#include <cassert>
#include <fstream>

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

// cf. https://stackoverflow.com/a/10409376/7272382
int reverseEndianness(int i) {
    int c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
std::vector<uint8_t> read_mnist(std::string full_path) {
    std::vector<uint8_t> vector_of_images;
    size_t image_index = 0;
    std::ifstream file(full_path, std::ios::binary);
    std::cout << "Reading " << full_path << std::endl;
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseEndianness(magic_number);
        assert(magic_number == 2051 || magic_number == 2049);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseEndianness(number_of_images);
        assert(number_of_images == 10000);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = reverseEndianness(n_rows);
        assert(n_rows == 28);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = reverseEndianness(n_cols);
        assert(n_cols == 28);
        vector_of_images.resize(number_of_images * n_rows * n_cols);
        for (int i = 0; i < number_of_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    file.read((char *)&(vector_of_images[image_index]), 1);
                    ++image_index;
                }
            }
        }
    }
    assert(image_index == vector_of_images.size());
    return vector_of_images;
}

void test_inference(std::string filename) {

    auto conv2 = test_load(filename, "conv2");
    auto fc1 = test_load(filename, "fc1");

    return;  // early return for CI, to avoid data handling ;)
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