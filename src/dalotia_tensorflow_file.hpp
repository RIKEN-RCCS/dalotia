#pragma once
#include <tensorflow/c/c_api.h>

#include <array>
#include <string>

#include "dalotia_formats.hpp"
#include "dalotia_tensor_file.hpp"

namespace dalotia {
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_datatype.h
static const std::map<TF_DataType, dalotia_WeightFormat> tensorflow_type_map{
    {TF_DOUBLE, dalotia_WeightFormat::dalotia_float_64},
    {TF_FLOAT, dalotia_WeightFormat::dalotia_float_32},
    {TF_HALF, dalotia_WeightFormat::dalotia_float_16},
    {TF_BFLOAT16, dalotia_WeightFormat::dalotia_bfloat_16},
    // {TF_BOOL,    dalotia_WeightFormat::dalotia_bool},
    {TF_INT8, dalotia_WeightFormat::dalotia_int_8},
    {TF_UINT8, dalotia_WeightFormat::dalotia_uint_8},
    {TF_INT16, dalotia_WeightFormat::dalotia_int_16},
    {TF_UINT16, dalotia_WeightFormat::dalotia_uint_16},
    {TF_INT32, dalotia_WeightFormat::dalotia_int_32},
    {TF_UINT32, dalotia_WeightFormat::dalotia_uint_32},
    // {TF_INT64,   dalotia_WeightFormat::dalotia_int_64},
    // {TF_UINT64,  dalotia_WeightFormat::dalotia_uint_64},
    // {TF_FLOAT8_E5M2, dalotia_WeightFormat::dalotia_float_8_e5m2},
    // {TF_INT2,   dalotia_WeightFormat::dalotia_int_2},
};

class TensorflowSavedModel : public TensorFile {
  public:
    explicit TensorflowSavedModel(const std::string &filename);

    ~TensorflowSavedModel() override;

    const std::vector<std::string> &get_tensor_names() const override;

    bool is_sparse(const std::string &tensor_name) const override;

    size_t get_num_dimensions(const std::string &tensor_name) const override;

    std::vector<int>
    get_tensor_extents(const std::string &tensor_name = "",
                       const std::vector<int> &permutation = {}) const override;

    void load_tensor_dense(const std::string &tensor_name,
                           dalotia_WeightFormat weightFormat, dalotia_Ordering ordering,
                           dalotia_byte *__restrict__ tensor,
                           const std::vector<int> &permutation = {}) override;

    std::vector<const dalotia_byte *> get_tensor_pointers(const std::string &tensor_name);

    // cf. https://github.com/serizba/cppflow/blob/master/include/cppflow/model.h
    std::shared_ptr<TF_Status> status_;
    std::shared_ptr<TF_Graph> graph_;
    std::shared_ptr<TF_Session> session_;
    std::vector<std::string> tensor_names_;
    std::map<std::string, std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>>
        tensors_;  // cache for loaded tensor pointers

  private:
    const TF_Tensor *get_tensor_pointer_from_name(const std::string &tensor_name);
};

}  // namespace dalotia