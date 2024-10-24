#include <assert.h>

#include "../dalotia.h"

void test_get_tensor_names(const char* filename) {
    DalotiaTensorFile* dalotia_file = open_file(filename);
    int num_tensors = get_num_tensors(dalotia_file);
    assert (num_tensors == 6);
    // auto tensor_names = dalotia_file->get_tensor_names(); //TODO
    // assert(tensor_names[0] == "conv1.bias");
    // assert(tensor_names[1] == "conv1.weight");
    // assert(tensor_names[2] == "conv2.bias");
    // assert(tensor_names[3] == "conv2.weight");
    // assert(tensor_names[4] == "fc1.bias");
    // assert(tensor_names[5] == "fc1.weight");
    close_file(dalotia_file);
}
int main(int, char **) {
    char filename[] = "../data/model-mnist.safetensors";

    test_get_tensor_names(filename);
    // test_load(filename, "conv1"); // TODO
    // test_load(filename, "conv2");
    // test_load(filename, "fc1");
    // test_inference(filename);
    return 0;
}