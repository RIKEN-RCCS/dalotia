#pragma once
#include <array>
#include <map>
#include <string>
#include <vector>

#include "dalotia_formats.hpp"
#include "dalotia_tensor_file.hpp"
#include "safetensors.hh"

namespace dalotia {

const std::map<safetensors::dtype,  dalotia_WeightFormat> safetensors_type_map{
    {safetensors::dtype::kFLOAT64,  dalotia_WeightFormat::dalotia_float_64},
    {safetensors::dtype::kFLOAT32,  dalotia_WeightFormat::dalotia_float_32},
    {safetensors::dtype::kFLOAT16,  dalotia_WeightFormat::dalotia_float_16},
    {safetensors::dtype::kBFLOAT16, dalotia_WeightFormat::dalotia_bfloat_16},
    // {kBOOL, dalotia_bool},
    {safetensors::dtype::kUINT8,    dalotia_WeightFormat::dalotia_uint_8},
    {safetensors::dtype::kINT8,     dalotia_WeightFormat::dalotia_int_8},
    {safetensors::dtype::kUINT16,   dalotia_WeightFormat::dalotia_uint_16},
    {safetensors::dtype::kINT32,    dalotia_WeightFormat::dalotia_int_32},
    {safetensors::dtype::kUINT32,   dalotia_WeightFormat::dalotia_uint_32},
    // {safetensors::dtype::kINT64,    dalotia_WeightFormat::dalotia_int_64},
    // {safetensors::dtype::kUINT64,   dalotia_WeightFormat::dalotia_uint_64},
    // {dalotia_float_8},
    // {dalotia_int_2},
};

class SafetensorsFile : public TensorFile {
   public:
    explicit SafetensorsFile(const std::string &filename);

    ~SafetensorsFile() override;

    const std::vector<std::string> &get_tensor_names() const override;

    bool is_sparse(const std::string &tensor_name) const override;

    size_t get_num_dimensions(const std::string &tensor_name) const override;

    size_t get_num_tensor_elements(const std::string &tensor_name) const override;

    std::vector<int> get_tensor_extents(
        const std::string &tensor_name = "",
        const std::vector<int>& permutation = {}) const override;

    void load_tensor_dense(const std::string &tensor_name,
                           dalotia_WeightFormat weightFormat,
                           dalotia_Ordering ordering,
                           dalotia_byte *__restrict__ tensor,
                           const std::vector<int>& permutation = {}) override;
    std::vector<const dalotia_byte*> get_mmap_tensor_pointers(
        const std::string &tensor_name) const override;
    
    // cf. https://github.com/syoyo/safetensors-cpp/blob/main/safetensors.hh
    safetensors::safetensors_t st_;
};

}  // namespace dalotia