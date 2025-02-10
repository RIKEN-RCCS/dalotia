#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#ifdef DALOTIA_WITH_CPP_PMR
#include <memory_resource>
#endif  // DALOTIA_WITH_CPP_PMR
#include <numeric>
#include <string>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"
#include "dalotia_tensor_file.hpp"

#ifdef DALOTIA_WITH_SAFETENSORS_CPP
#include "dalotia_safetensors_file.hpp"
#endif

namespace dalotia {
// factory function for the file, selected by file extension and
// available implementations
TensorFile *make_tensor_file(const std::string & filename);

// C++17 version -> will not compile on Fugaku...
// -- pmr vector types can accept different allocators
//? more memory interface than that? detect if CUDA device pointer through
// unified access... how about other devices?
template <typename value_type = dalotia_byte>  //? or have no defaults?
[[nodiscard]] std::pair<std::vector<int>, dalotia::vector<value_type>>
load_tensor_dense(
    const std::string &filename, const std::string &tensor_name,
    dalotia_WeightFormat weight_format,
    dalotia_Ordering ordering = dalotia_C_ordering,
    const std::vector<int> &permutation = {}
#ifdef DALOTIA_WITH_CPP_PMR
    ,
    const std::pmr::polymorphic_allocator<dalotia_byte> &allocator =
        std::pmr::polymorphic_allocator<dalotia_byte>()
#endif  // DALOTIA_WITH_CPP_PMR
) {
    auto dalotia_file = std::unique_ptr<TensorFile>(make_tensor_file(filename));
    auto long_extents =
        dalotia_file->get_tensor_extents(tensor_name, permutation);
    // shorten extents to nonzeros
    auto num_nonzero = long_extents.size() -
                       std::count(long_extents.begin(), long_extents.end(), -1);

    std::vector<int> true_extents(long_extents.begin(),
                                      long_extents.begin() + num_nonzero);
    auto total_size = std::accumulate(true_extents.begin(), true_extents.end(),
                                      1, std::multiplies<size_t>());
#ifdef DALOTIA_WITH_CPP_PMR
    dalotia::vector<value_type> tensor(allocator);
#else
    dalotia::vector<value_type> tensor;
#endif  // DALOTIA_WITH_CPP_PMR
    if constexpr (std::is_same_v<value_type, dalotia_byte>) {
        tensor.resize(total_size * sizeof_weight_format(weight_format));
    } else {
        tensor.resize(total_size);
    }
    dalotia_file->load_tensor_dense(
        tensor_name, weight_format, ordering,
        reinterpret_cast<dalotia_byte *>(tensor.data()), permutation);
    return std::make_pair(true_extents, tensor);
}

// TODO same for sparse

// TODO allow md-range sub-tensor requests

}  // namespace dalotia
