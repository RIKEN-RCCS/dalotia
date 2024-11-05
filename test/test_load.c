#include <assert.h>
#include <stdio.h> //TODO remove
#include <stdlib.h>
#include <string.h>

#include "dalotia.h"

void test_get_tensor_names(const char* filename) {
    DalotiaTensorFile* dalotia_file = open_file(filename);
    int num_tensors = get_num_tensors(dalotia_file);
    assert(num_tensors == 6);
    for (int i = 0; i < num_tensors; i++) {
        char name[100];
        get_tensor_name(dalotia_file, i, name);
        // fprintf(stderr, "name: %s , %d \n", name, i);
        if (i == 0) {
            assert(strcmp(name, "conv1.bias") == 0);
        } else if (i == 1) {
            assert(strcmp(name, "conv1.weight") == 0);
        } else if (i == 2) {
            assert(strcmp(name, "conv2.bias") == 0);
        } else if (i == 3) {
            assert(strcmp(name, "conv2.weight") == 0);
        } else if (i == 4) {
            assert(strcmp(name, "fc1.bias") == 0);
        } else if (i == 5) {
            assert(strcmp(name, "fc1.weight") == 0);
        }
    }
    close_file(dalotia_file);
}

void assert_close(volatile float a, volatile float b) {
    assert(abs(a - b) < 1e-5);
}

void test_load(const char* filename, const char* tensor_name) {
    DalotiaTensorFile* dalotia_file = open_file(filename);
    {
        const dalotia_Ordering ordering = dalotia_C_ordering;
        const dalotia_WeightFormat weightFormat = dalotia_float_32;
        char tensor_name_weight[255];
        char tensor_name_bias[255];
        sprintf(tensor_name_weight, "%s.weight", tensor_name);
        sprintf(tensor_name_bias, "%s.bias", tensor_name);
        int extents_weight[10], extents_bias[10];
        // initialize to -1
        for (int i = 0; i < 10; i++) {
            extents_weight[i] = -1;
            extents_bias[i] = -1;
        }
        int num_dimensions_weight = get_tensor_extents(
            dalotia_file, tensor_name_weight, extents_weight);
        int num_elements_weight =
            get_num_tensor_elements(dalotia_file, tensor_name_weight);
        int num_dimensions_bias =
            get_tensor_extents(dalotia_file, tensor_name_bias, extents_bias);
        int num_elements_bias =
            get_num_tensor_elements(dalotia_file, tensor_name_bias);

        if (strcmp(tensor_name, "conv1") == 0) {
            assert(num_dimensions_weight == 4);
            assert(num_elements_weight == 72);
            assert(extents_weight[0] == 8);
            assert(extents_weight[1] == 1);
            assert(extents_weight[2] == 3);
            assert(extents_weight[3] == 3);
            assert(num_dimensions_bias == 1);
            assert(num_elements_bias == 8);
            assert(extents_bias[0] == 8);
        } else if (strcmp(tensor_name, "conv2") == 0) {
            assert(num_dimensions_weight == 4);
            assert(num_elements_weight == 1152);
            assert(extents_weight[0] == 16);
            assert(extents_weight[1] == 8);
            assert(extents_weight[2] == 3);
            assert(extents_weight[3] == 3);
            assert(num_dimensions_bias == 1);
            assert(num_elements_bias == 16);
            assert(extents_bias[0] == 16);
        } else if (strcmp(tensor_name, "fc1") == 0) {
            assert(num_dimensions_weight == 2);
            assert(num_elements_weight == 784 * 10);
            assert(extents_weight[0] == 10);
            assert(extents_weight[1] == 784);
            assert(num_dimensions_bias == 1);
            assert(num_elements_bias == 10);
            assert(extents_bias[0] == 10);
        } else {
            assert(0);
        }

        float *tensor_weight, *tensor_bias;
        tensor_weight = (float*)malloc(num_elements_weight * sizeof(float));
        load_tensor_dense(dalotia_file, tensor_name_weight,
                          (char*)tensor_weight, weightFormat, ordering);

        tensor_bias = (float*)malloc(num_elements_bias * sizeof(float));
        load_tensor_dense(dalotia_file, tensor_name_bias, (char*)tensor_bias,
                          weightFormat, ordering);

        // check if the first, second, and last values are as expected
        if (strcmp(tensor_name, "conv1") == 0) {
            assert_close(tensor_weight[0], 0.944823);
            assert_close(tensor_weight[1], 1.25045);
            assert_close(tensor_weight[71], 0.211111);
            assert_close(tensor_weight[0],
                         0.0);  // TODO why is this not failing?
            assert_close(tensor_weight[1], 1.0);
            assert_close(tensor_weight[71], 0.0);
            fprintf(stderr, "tensor_bias[0]: %f\n", tensor_bias[0]);
            assert_close(tensor_bias[0], 0.0);
            fprintf(stderr, "tensor_bias[7]: %f\n", tensor_bias[7]);
            assert_close(tensor_bias[7], 0.0);
        } else {
            assert(0);
        }

        free(tensor_weight);
        free(tensor_bias);
    }
    close_file(dalotia_file);
}

int main(int, char**) {
    char filename[] = "../data/model-mnist.safetensors";

    test_get_tensor_names(filename);
    test_load(filename, "conv1");
    // test_load(filename, "conv2"); // TODO
    // test_load(filename, "fc1");
    return 0;
}