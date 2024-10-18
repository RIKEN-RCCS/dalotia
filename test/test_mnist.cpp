#include <cassert>
#include <fstream>

#include "../dalotia.hpp"
#include "../safetensors_file.hpp"
// #include "mdspan/mdspan.hpp"
#include <boost/multi/array.hpp>

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
std::pmr::vector<uint8_t> read_mnist(std::string full_path) {
    std::pmr::vector<uint8_t> vector_of_images;
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
        file.read((char *)vector_of_images.data(), vector_of_images.size());
    }
    return vector_of_images;
}

namespace multi = boost::multi;

void test_inference(std::string filename) {
    using span_4d_float = multi::array_ref<float, 4>;
    using span_3d_float = multi::array_ref<float, 3>;
    using span_2d_float = multi::array_ref<float, 2>;
    using span_2d_char = multi::array_ref<uint8_t, 2>;

    auto [conv1_weight, conv1_bias] = test_load(filename, "conv1");
    const auto conv1_weight_span =
        span_4d_float({8, 1, 3, 3}, conv1_weight.data());
    const auto conv1_bias_span = span_2d_float({8, 1}, conv1_bias.data());
    assert(conv1_weight_span.sizes().get<1>() == 1);  // 1 input channel

    auto conv2 = test_load(filename, "conv2");
    auto fc1 = test_load(filename, "fc1");
    const auto fc1_weight_span = span_2d_float({10, 784}, fc1.first.data());
    const auto fc1_bias_span = span_2d_float({10, 1}, fc1.second.data());

    return;  // early return for CI, to avoid data handling ;)

    // load the mnist test data // as in
    // https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
    // too
    std::string mnist_test_images_filename =
        "../../python-fiddling/dataset/MNIST/raw/t10k-images-idx3-ubyte";
    std::string mnist_test_labels_filename =
        "../../python-fiddling/dataset/MNIST/raw/t10k-labels-idx3-ubyte";

    auto images = read_mnist(mnist_test_images_filename);
    // auto labels = read_mnist(mnist_test_labels_filename);
    auto num_images = images.size() / (28 * 28);

    // for (auto &image : images) {  // todo minibatching
    std::cout << "num_images: " << num_images << std::endl;
    for (size_t image_index = 0; image_index < num_images; ++image_index) {
        // copy data to larger array for zero-padding at the edges
        auto image_vector_padded = std::pmr::vector<uint8_t>(30 * 30);
        auto image_padded_span =
            span_2d_char({30, 30}, image_vector_padded.data());
        image_padded_span({1, 29}, {1, 29}) =
            span_2d_char({28, 28}, images.data() + image_index * (28 * 28));

        auto conv1_output = std::pmr::vector<float>(8 * 28 * 28);
        auto conv1_output_span =
            span_3d_float({8, 28, 28}, conv1_output.data());
// apply first convolution
#pragma omp parallel for
        for (int i = 1; i < image_padded_span.sizes().get<0>() - 1; ++i) {
            for (int j = 1; j < image_padded_span.sizes().get<1>() - 1; ++j) {
                for (int k = 0; k < conv1_weight_span.sizes().get<0>(); ++k) {
                    // sum_m_n(conv1_weight_span[k, 0, m, n] *
                    // image_padded_span[i + m -1, j + n -1] ) + bias[k] (10
                    // terms per k)
                    conv1_output_span(k, i - 1, j - 1) =
                        conv1_weight_span(k, 0, 0, 0) *
                            image_padded_span(i - 1, j - 1) +
                        conv1_weight_span(k, 0, 0, 1) *
                            image_padded_span(i - 1, j + 0) +
                        conv1_weight_span(k, 0, 0, 2) *
                            image_padded_span(i - 1, j + 1) +
                        conv1_weight_span(k, 0, 1, 0) *
                            image_padded_span(i + 0, j - 1) +
                        conv1_weight_span(k, 0, 1, 1) *
                            image_padded_span(i + 0, j + 0) +
                        conv1_weight_span(k, 0, 1, 2) *
                            image_padded_span(i + 0, j + 1) +
                        conv1_weight_span(k, 0, 2, 0) *
                            image_padded_span(i + 1, j - 1) +
                        conv1_weight_span(k, 0, 2, 1) *
                            image_padded_span(i + 1, j + 0) +
                        conv1_weight_span(k, 0, 2, 2) *
                            image_padded_span(i + 1, j + 1) +
                        conv1_bias_span(k, 0);
                    // apply first activation function (relu)
                    if (conv1_output_span(k, i - 1, j - 1) < 0.) {
                        conv1_output_span(k, i - 1, j - 1) = 0.;
                    }
                }
            }
        }

        // apply max pooling
        std::pmr::vector<float> conv1_output_pooled(8 * 14 * 14);
        auto conv1_output_pooled_span =
            span_3d_float({8, 14, 14}, conv1_output_pooled.data());
#pragma omp parallel for
        for (int i = 0; i < 14; ++i) {
            for (int j = 0; j < 14; ++j) {
                for (int k = 0; k < 8; ++k) {
                    auto window = conv1_output_span(k, {2 * i, 2 * i + 1},
                                                    {2 * j, 2 * j + 1});
                    auto max_val =
                        (*std::max_element(window.begin(), window.end()))[0];
                    conv1_output_pooled_span(k, i, j) = max_val;
                }
            }
        }
    }
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