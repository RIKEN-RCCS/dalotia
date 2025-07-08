#include "dalotia_tensorflow_file.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"

namespace dalotia {

TF_Output get_operation_from_name(const std::string &tensor_name,
                                  std::shared_ptr<TF_Graph> graph) {
    TF_Operation *oper = TF_GraphOperationByName(graph.get(), tensor_name.c_str());
    return {oper, 0};
}

// parts of this code are intensely based on cppflow, esp. tf_status_check and the
// constructor -- so here goes their license for the respective parts:

// MIT License
//
// Copyright (c) 2019 Sergio Izquierdo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

inline bool tf_status_check(std::shared_ptr<TF_Status> status) {
    // cf. https://github.com/serizba/cppflow/blob/master/include/cppflow/context.h#L45
    if (TF_GetCode(status.get()) != TF_OK) {
        throw std::runtime_error(TF_Message(status.get()));
    }
    return true;
}

int tf_get_num_dimensions(TF_Output output, std::shared_ptr<TF_Graph> graph,
                          std::shared_ptr<TF_Status> status) {
    // TF_DataType dtype = TF_OperationOutputType(output);
    int num_dimensions = TF_GraphGetTensorNumDims(graph.get(), output, status.get());
    tf_status_check(status);
    return num_dimensions;
}

TensorflowSavedModel::TensorflowSavedModel(const std::string &filename)
    : TensorFile(filename) {
    // cf.
    // https://github.com/serizba/cppflow/blob/master/include/cppflow/model.h
    this->status_ = {TF_NewStatus(), &TF_DeleteStatus};
    this->graph_ = {TF_NewGraph(), TF_DeleteGraph};

    // Create the session.
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>
        session_options = {TF_NewSessionOptions(), TF_DeleteSessionOptions};

    auto session_deleter = [this](TF_Session *sess) {
        TF_DeleteSession(sess, this->status_.get());
        tf_status_check(this->status_);
    };

    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> run_options = {
        TF_NewBufferFromString("", 0), TF_DeleteBuffer};
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> meta_graph = {TF_NewBuffer(),
                                                                         TF_DeleteBuffer};

    int tag_len = 1;
    const char *tag = "serve";
    this->session_ = {TF_LoadSessionFromSavedModel(session_options.get(),
                                                   run_options.get(), filename.c_str(),
                                                   &tag, tag_len, this->graph_.get(),
                                                   meta_graph.get(), this->status_.get()),
                      session_deleter};
    tf_status_check(this->status_);

    {  // create and fill the tensor names vector
        size_t pos = 0;
        TF_Operation *oper;
        while ((oper = TF_GraphNextOperation(graph_.get(), &pos)) != nullptr) {
            const char *op_name = TF_OperationName(oper);
            tensor_names_.emplace_back(op_name);
        }
    }
}

TensorflowSavedModel::~TensorflowSavedModel() = default;

const std::vector<std::string> &TensorflowSavedModel::get_tensor_names() const {
    return tensor_names_;
}

bool TensorflowSavedModel::is_sparse(const std::string & /*tensor_name*/) const {
    return false;
}

size_t TensorflowSavedModel::get_num_dimensions(const std::string &tensor_name) const {
    TF_Output output = get_operation_from_name(tensor_name, this->graph_);
    if (output.oper == nullptr) {
        throw std::runtime_error(
            "Tensor not found: " + tensor_name +
            ". Tensor names in the file: " + to_string(tensor_names_));
    }
    if (tensor_name == "NoOp") {
        // NoOp is a special operation in TensorFlow, it has no dimensions
        // (weird vector error otherwise)
        return 0;
    }
    int num_dimensions = tf_get_num_dimensions(output, this->graph_, this->status_);
    if (num_dimensions < 0) {
        throw std::runtime_error("Failed to get number of dimensions for tensor: " +
                                 tensor_name);
    }
    return num_dimensions;
}

std::vector<int>
TensorflowSavedModel::get_tensor_extents(const std::string &tensor_name,
                                         const std::vector<int> &permutation) const {
    TF_Output output = get_operation_from_name(tensor_name, this->graph_);
    if (output.oper == nullptr) {
        throw std::runtime_error(
            "Tensor not found: " + tensor_name +
            ". Tensor names in the file: " + to_string(tensor_names_));
    }

    int num_dimensions = tf_get_num_dimensions(output, this->graph_, this->status_);
    std::vector<int64_t> extents_read(num_dimensions);
    // use tensor_name only from the last slash onward
    size_t last_slash_pos = tensor_name.find_last_of('/');
    auto shortened_tensor_name = (last_slash_pos != std::string::npos)
                                     ? tensor_name.substr(last_slash_pos + 1)
                                     : tensor_name;
    TF_GraphGetTensorShape(this->graph_.get(), output, extents_read.data(),
                           extents_read.size(), this->status_.get());
    tf_status_check(this->status_);

    std::vector<int> extents(extents_read.size());

    if (!permutation.empty()) {
        auto final_permutation_in_c_order =
            final_c_permutation_from_permutation_and_order(
                permutation, dalotia_Ordering::dalotia_C_ordering, extents.size());
        if (!final_permutation_in_c_order.empty()) {
            for (size_t i = 0; i < extents.size(); i++) {
                extents[i] = extents_read[final_permutation_in_c_order[i]];
            }
        }
    } else {
        extents.assign(extents_read.begin(), extents_read.end());
    }
    return extents;
}

void TensorflowSavedModel::load_tensor_dense(const std::string &tensor_name,
                                             dalotia_WeightFormat weightFormat,
                                             dalotia_Ordering ordering,
                                             dalotia_byte *__restrict__ tensor,
                                             const std::vector<int> &permutation) {
    TF_Output output = get_operation_from_name(tensor_name, this->graph_);
    if (output.oper == nullptr) {
        throw std::runtime_error(
            "Tensor not found: " + tensor_name +
            ". Tensor names in the file: " + to_string(tensor_names_));
    }
    const size_t num_tensor_elements = this->get_num_tensor_elements(tensor_name);
    std::cout << "dalotia: loading tensor " << tensor_name << " with "
              << num_tensor_elements << std::endl;

    TF_Tensor *tf_tensor = nullptr;
    TF_SessionRun(this->session_.get(), nullptr, nullptr, nullptr, 0, &output, &tf_tensor,
                  1, nullptr, 0, nullptr, this->status_.get());
    std::cout << "dalotia: loaded tensor " << tensor_name << std::endl;
    if (tf_tensor == nullptr) {
        throw std::runtime_error("Failed to load tensor: " + tensor_name);
    }
    tf_status_check(this->status_);

    void *databuffer = TF_TensorData(tf_tensor);
    int num_dimensions = TF_NumDims(tf_tensor);

    std::cout << "dalotia: loading tensor " << tensor_name << " with "
              << TF_TensorElementCount(tf_tensor)
              << " elements, num_dimensions: " << num_dimensions << std::endl;
    TF_DataType tf_type = TF_TensorType(tf_tensor);
    const dalotia_WeightFormat input_weight_format = tensorflow_type_map.at(tf_type);
#ifndef NDEBUG
    assert(databuffer != nullptr);
    assert(tf_tensor != nullptr);
    assert(num_dimensions == static_cast<int>(this->get_num_dimensions(tensor_name)));
    assert(num_dimensions >= 0);
    assert(TF_TensorElementCount(tf_tensor) == static_cast<int64_t>(num_tensor_elements));
    size_t num_bytes = TF_TensorByteSize(tf_tensor);
    assert(num_bytes ==
           static_cast<size_t>(dalotia::sizeof_weight_format(input_weight_format)) *
               num_tensor_elements);
#endif  // NDEBUG

    auto *tensor_start = reinterpret_cast<const dalotia_byte *__restrict__>(databuffer);

    auto final_permutation_in_c_order = final_c_permutation_from_permutation_and_order(
        permutation, ordering, num_dimensions);
    if (!final_permutation_in_c_order.empty()) {
        std::vector<int> input_shape = this->get_tensor_extents(tensor_name);
        dalotia::assign_permuted(num_dimensions, tensor, weightFormat, input_shape.data(),
                                 tensor_start, input_weight_format,
                                 final_permutation_in_c_order.data());
    } else {
        dalotia::assign_linearly(tensor, weightFormat, num_tensor_elements, tensor_start,
                                 input_weight_format);
    }
    TF_DeleteTensor(tf_tensor);
}
}  // namespace dalotia