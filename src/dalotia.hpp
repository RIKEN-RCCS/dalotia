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
#ifdef DALOTIA_WITH_TENSORFLOW
#include "dalotia_tensorflow_file.hpp"
#endif

namespace dalotia {
// factory function for the file, selected by file extension and
// available implementations
[[nodiscard]] TensorFile *make_tensor_file(const std::string & filename);

// C++17 version -> will not compile on Fugaku...
// -- pmr vector types can accept different allocators
//? more memory interface than that? detect if CUDA device pointer through
// unified access... how about other devices?
template <typename value_type = dalotia_byte, typename... Ts>
[[nodiscard]] std::pair<std::vector<int>, dalotia::vector<value_type>>
load_tensor_dense(
    const std::string &filename, const std::string &tensor_name, Ts&&... params
) {
    auto dalotia_file = std::unique_ptr<TensorFile>(make_tensor_file(filename));
    return dalotia_file->load_tensor_dense<value_type>(tensor_name, std::forward<Ts>(params)...);
}

// TODO allow md-range sub-tensor requests

}  // namespace dalotia
