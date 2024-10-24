#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string>
#include <vector>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"
#include "tensor_file.hpp"

#ifdef DALOTIA_WITH_SAFETENSORS_CPP
#include "safetensors_file.hpp"
#endif

namespace dalotia {
// factory function for the file, selected by file extension and
// available implementations
TensorFile *make_tensor_file(std::string filename);

// C++17 version -> will not compile on Fugaku...
// -- pmr vector types can accept different allocators
//? more memory interface than that? detect if CUDA device pointer through
// unified access... how about other devices?
template <typename value_type = std::byte>  //? or have no defaults?
[[nodiscard]] std::pair<std::pmr::vector<int>, std::pmr::vector<value_type>>
load_tensor_dense(std::string filename, std::string tensor_name,
                  dalotia_WeightFormat weight_format,
                  dalotia_Ordering ordering = dalotia_C_ordering,
                  const std::pmr::polymorphic_allocator<std::byte> &allocator =
                      std::pmr::polymorphic_allocator<std::byte>(),
                  const std::pmr::vector<int> &permutation = {}) {
    auto dalotia_file = std::unique_ptr<TensorFile>(make_tensor_file(filename));
    const int *permutation_ptr = nullptr;
    if (!permutation.empty()) {
        permutation_ptr = permutation.data();
    }
    auto long_extents =
        dalotia_file->get_tensor_extents(tensor_name, permutation_ptr);
    // shorten extents to nonzeros
    auto num_nonzero = long_extents.size() -
                       std::count(long_extents.begin(), long_extents.end(), -1);
    std::pmr::vector<int> true_extents(
        long_extents.begin(), long_extents.begin() + num_nonzero, allocator);
    auto total_size = std::accumulate(true_extents.begin(), true_extents.end(),
                                      1, std::multiplies<size_t>());

    std::pmr::vector<value_type> tensor(allocator);
    if constexpr (std::is_same_v<value_type, std::byte>) {
        tensor.resize(total_size * sizeof_weight_format(weight_format));
    } else {
        tensor.resize(total_size);
    }
    dalotia_file->load_tensor_dense(
        tensor_name, weight_format, ordering,
        reinterpret_cast<std::byte *>(tensor.data()), permutation_ptr);
    return std::make_pair(true_extents, tensor);
}

// TODO same for sparse

// TODO allow md-range sub-tensor requests

}  // namespace dalotia
