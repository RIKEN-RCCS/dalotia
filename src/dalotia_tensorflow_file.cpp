#include "dalotia_tensorflow_file.hpp"

#include <algorithm>
#include <cassert>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"

namespace dalotia {

TF_Output get_operation_from_name(const std::string& tensor_name,
                                  std::shared_ptr<TF_Graph> graph) {
    TF_Operation* oper =
        TF_GraphOperationByName(graph.get(), tensor_name.c_str());
    return {oper, 0};
}

// parts of this code are intensely based on cppflow, esp. tf_status_check and
// the constructor -- so here goes their license for the respective parts:

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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

inline bool tf_status_check(std::shared_ptr<TF_Status> status) {
    // cf.
    // https://github.com/serizba/cppflow/blob/master/include/cppflow/context.h#L45
    if (TF_GetCode(status.get()) != TF_OK) {
        throw std::runtime_error(TF_Message(status.get()));
    }
    return true;
}

int tf_get_num_dimensions(TF_Output output, std::shared_ptr<TF_Graph> graph,
                          std::shared_ptr<TF_Status> status) {
    // TF_DataType dtype = TF_OperationOutputType(output);
    int num_dimensions =
        TF_GraphGetTensorNumDims(graph.get(), output, status.get());
    tf_status_check(status);
    return num_dimensions;
}

TensorflowSavedModel::TensorflowSavedModel(const std::string& filename)
    : TensorFile(filename) {
    // cf.
    // https://github.com/serizba/cppflow/blob/master/include/cppflow/model.h
    this->status_ = {TF_NewStatus(), &TF_DeleteStatus};
    this->graph_ = {TF_NewGraph(), TF_DeleteGraph};

    // Create the session.
    std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>
        session_options = {TF_NewSessionOptions(), TF_DeleteSessionOptions};

    auto session_deleter = [this](TF_Session* sess) {
        TF_DeleteSession(sess, this->status_.get());
        tf_status_check(this->status_);
    };

    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> run_options = {
        TF_NewBufferFromString("", 0), TF_DeleteBuffer};
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> meta_graph = {
        TF_NewBuffer(), TF_DeleteBuffer};

    int tag_len = 1;
    const char* tag = "serve";
    this->session_ = {
        TF_LoadSessionFromSavedModel(
            session_options.get(), run_options.get(), filename.c_str(), &tag,
            tag_len, this->graph_.get(), meta_graph.get(), this->status_.get()),
        session_deleter};
    tf_status_check(this->status_);

    {  // create and fill the tensor names vector
        size_t pos = 0;
        TF_Operation* oper;
        while ((oper = TF_GraphNextOperation(graph_.get(), &pos)) != nullptr) {
            const char* op_name = TF_OperationName(oper);
            tensor_names_.emplace_back(op_name);
        }
    }
}

TensorflowSavedModel::~TensorflowSavedModel() = default;

const std::vector<std::string>& TensorflowSavedModel::get_tensor_names() const {
    return tensor_names_;
}

bool TensorflowSavedModel::is_sparse(const std::string& /*tensor_name*/) const {
    return false;
}

std::vector<int> TensorflowSavedModel::get_tensor_extents_raw(
    const std::string& tensor_name) const {
    TF_Output output = get_operation_from_name(tensor_name, this->graph_);
    if (output.oper == nullptr) {
        throw std::runtime_error(
            "Tensor not found: " + tensor_name +
            ". Tensor names in the file: " + to_string(tensor_names_));
    }
    if (tensor_name == "NoOp") {
        // NoOp is a special operation in TensorFlow, it has no dimensions
        // (weird vector error otherwise)
        return {};
    }
    int num_dimensions =
        tf_get_num_dimensions(output, this->graph_, this->status_);
    if (num_dimensions < 0) {
        throw std::runtime_error(
            "Failed to get number of dimensions for tensor: " + tensor_name);
    }
    std::vector<int64_t> extents_read(num_dimensions);
    TF_GraphGetTensorShape(this->graph_.get(), output, extents_read.data(),
                           extents_read.size(), this->status_.get());
    tf_status_check(this->status_);
    return {extents_read.begin(), extents_read.end()};
}

TensorFile::TensorInfo TensorflowSavedModel::get_tensor_info(
    const std::string& tensor_name) const {
    // const_cast needed because get_tensor_pointer_from_name caches the
    // tensor internally (mutates tensors_ map on first access).
    const TF_Tensor* tf_tensor =
        const_cast<TensorflowSavedModel*>(this)->get_tensor_pointer_from_name(
            tensor_name);
    void* databuffer = TF_TensorData(tf_tensor);
    TF_DataType tf_type = TF_TensorType(tf_tensor);
    int num_dimensions = TF_NumDims(tf_tensor);

    std::vector<int> shape(num_dimensions);
    for (int i = 0; i < num_dimensions; ++i) {
        shape[i] = static_cast<int>(TF_Dim(tf_tensor, i));
    }

    return {
        reinterpret_cast<const dalotia_byte*>(databuffer),
        tensorflow_type_map.at(tf_type),
        std::move(shape),
        static_cast<size_t>(TF_TensorElementCount(tf_tensor)),
    };
}

std::vector<const dalotia_byte*> TensorflowSavedModel::get_tensor_pointers(
    const std::string& tensor_name) {
    const TF_Tensor* tf_tensor =
        this->get_tensor_pointer_from_name(tensor_name);
    return std::vector<const dalotia_byte*>(
        1, reinterpret_cast<const dalotia_byte*>(TF_TensorData(tf_tensor)));
}

const TF_Tensor* TensorflowSavedModel::get_tensor_pointer_from_name(
    const std::string& tensor_name) {
    // check if it is already in the cache
    auto it = tensors_.find(tensor_name);
    if (it != tensors_.end()) {
        return it->second.get();
    } else {
        // if not, load it from the graph
        TF_Output output = get_operation_from_name(tensor_name, this->graph_);
        if (output.oper == nullptr) {
            throw std::runtime_error(
                "Tensor not found: " + tensor_name +
                ". Tensor names in the file: " + to_string(tensor_names_));
        }

        TF_Tensor* tf_tensor = nullptr;
        TF_SessionRun(this->session_.get(), nullptr, nullptr, nullptr, 0,
                      &output, &tf_tensor, 1, nullptr, 0, nullptr,
                      this->status_.get());
        if (tf_tensor == nullptr) {
            throw std::runtime_error("Failed to load tensor: " + tensor_name);
        }
        tf_status_check(this->status_);
        auto [position, inserted] = this->tensors_.emplace(
            tensor_name, std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>(
                             tf_tensor, &TF_DeleteTensor));
        assert(inserted);  // should not already exist
        return position->second.get();
    }
}
}  // namespace dalotia
