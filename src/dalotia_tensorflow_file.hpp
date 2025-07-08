#pragma once
#include <tensorflow/c/c_api.h>

#include <array>
#include <string>

#include "dalotia_formats.hpp"
#include "dalotia_tensor_file.hpp"

namespace dalotia {

class TensorflowSavedModel : public TensorFile {
  public:
    explicit TensorflowSavedModel(const std::string &filename);

    ~TensorflowSavedModel();

    const std::vector<std::string> &get_tensor_names() const override;

    bool is_sparse(const std::string &tensor_name) const override;

    size_t get_num_dimensions(const std::string &tensor_name) const override;

    std::vector<int>
    get_tensor_extents(const std::string &tensor_name = "",
                       const std::vector<int> &permutation = {}) const override;

    // void load_tensor_dense(const std::string &tensor_name,
    //                        dalotia_WeightFormat weightFormat,
    //                        dalotia_Ordering ordering,
    //                        dalotia_byte *__restrict__ tensor,
    //                        const std::vector<int>& permutation = {}) override;
    
    // cf. https://github.com/serizba/cppflow/blob/master/include/cppflow/model.h
    std::shared_ptr<TF_Status> status_;
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Session> session_;
    std::vector<std::string> tensor_names_;
};

}  // namespace dalotia