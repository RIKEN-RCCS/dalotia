#pragma once
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <stdexcept>

#ifdef DALOTIA_WITH_CPP_PMR
#include <memory_resource>
// define dalotia::vector as std::pmr::vector
namespace dalotia {
template <typename T>
using vector = std::pmr::vector<T>;
}  // namespace dalotia
#else
#include <vector>
// define dalotia::vector as std::vector
namespace dalotia {
template <typename T>
using vector = std::vector<T>;
}  // namespace dalotia
#endif

#include "dalotia_formats.hpp"

namespace dalotia {

std::vector<int> final_c_permutation_from_permutation_and_order(
    const std::vector<int>& permutation, dalotia_Ordering ordering,
    size_t num_dimensions);

template <typename InType, typename OutType>
std::function<void(dalotia_byte* __restrict__,
                   const dalotia_byte* __restrict__)>
cpp_type_assignment(size_t store_item_bytes) {
    // if both types are builtins, cast input and assign the resulting bytes
    auto fcn = [store_item_bytes](
                   dalotia_byte* __restrict__ output_bytes,
                   const dalotia_byte* __restrict__ input_bytes) {
        auto input_cast = reinterpret_cast<const InType*>(input_bytes);
        auto output_cast = static_cast<OutType>(*input_cast);
        assert(sizeof(output_cast) == store_item_bytes);
        auto copy_bytes = reinterpret_cast<dalotia_byte*>(&output_cast);
        for (size_t j = 0; j < store_item_bytes; ++j) {
            output_bytes[j] = copy_bytes[j];
        }
    };
    return fcn;
}

std::function<void(dalotia_byte* __restrict__,
                   const dalotia_byte* __restrict__)>
get_assignment_function(dalotia_WeightFormat weight_output_format,
                        dalotia_WeightFormat weight_input_format);

void assign_linearly(dalotia_byte* __restrict__ dest,
                     dalotia_WeightFormat weight_output_format,
                     size_t num_items,
                     const dalotia_byte* const __restrict__ tensor_start,
                     dalotia_WeightFormat weight_input_format);

// Recursive compile-time loop for stride-based permuted copy.
// Fully unrolled at compile time for each num_dimensions.
template <int num_dimensions, int dim = 0>
inline void assign_permuted_loop(
    dalotia_byte* __restrict__ dest,
    const dalotia_byte* __restrict__& input_pointer, size_t& store_index,
    size_t store_item_bytes, size_t load_item_bytes, const int* input_shape,
    const size_t* strides_permuted,
    const std::function<void(dalotia_byte* __restrict__,
                             const dalotia_byte* __restrict__)>& assign_fn) {
    for (int i = 0; i < input_shape[dim]; ++i) {
        if constexpr (dim + 1 == num_dimensions) {
            auto output_pointer = dest + store_index * store_item_bytes;
            assign_fn(output_pointer, input_pointer);
            input_pointer += load_item_bytes;
        } else {
            assign_permuted_loop<num_dimensions, dim + 1>(
                dest, input_pointer, store_index, store_item_bytes,
                load_item_bytes, input_shape, strides_permuted, assign_fn);
        }
        store_index += strides_permuted[dim];
    }
    store_index -=
        static_cast<size_t>(input_shape[dim]) * strides_permuted[dim];
}

// Compute C-order input strides, permuted output strides, and total size.
// Compile-time version (returns std::arrays).
template <int num_dimensions>
inline std::tuple<std::array<size_t, num_dimensions>,
                  std::array<size_t, num_dimensions>, size_t>
compute_permute_strides(const int* input_shape, const int* permutation) {
    // C-order input strides
    std::array<size_t, num_dimensions> input_strides;
    input_strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }
    // Permuted output shape and strides
    std::array<int, num_dimensions> output_shape;
    size_t total_size = 1;
    for (int i = 0; i < num_dimensions; ++i) {
        output_shape[i] = input_shape[permutation[i]];
        total_size *= output_shape[i];
    }
    std::array<size_t, num_dimensions> output_strides;
    output_strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; --i) {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }
    std::array<size_t, num_dimensions> permuted_strides;
    for (int i = 0; i < num_dimensions; ++i) {
        permuted_strides[permutation[i]] = output_strides[i];
    }
    return {input_strides, permuted_strides, total_size};
}

// (returns only permuted strides).
template <int num_dimensions>
inline std::pair<std::array<size_t, num_dimensions>, size_t>
get_new_strides_permuted(const int* input_shape, const int* permutation) {
    auto [input_strides, permuted_strides, total_size] =
        compute_permute_strides<num_dimensions>(input_shape, permutation);
    return {permuted_strides, total_size};
}

// Runtime (vector-based) version. Delegates to the compile-time template.
// Returns {input_strides, permuted_output_strides, total_size}.
namespace detail {
template <int N>
inline std::tuple<std::vector<size_t>, std::vector<size_t>, size_t>
compute_permute_strides_dispatch(const int* input_shape,
                                 const int* permutation) {
    auto [in_s, perm_s, total] =
        compute_permute_strides<N>(input_shape, permutation);
    // convert std::array → std::vector
    return {std::vector<size_t>(in_s.begin(), in_s.end()),
            std::vector<size_t>(perm_s.begin(), perm_s.end()), total};
}
}  // namespace detail

inline std::tuple<std::vector<size_t>, std::vector<size_t>, size_t>
compute_permute_strides(int ndims, const int* input_shape,
                        const int* permutation) {
    switch (ndims) {
    case 1:
        return detail::compute_permute_strides_dispatch<1>(input_shape,
                                                           permutation);
    case 2:
        return detail::compute_permute_strides_dispatch<2>(input_shape,
                                                           permutation);
    case 3:
        return detail::compute_permute_strides_dispatch<3>(input_shape,
                                                           permutation);
    case 4:
        return detail::compute_permute_strides_dispatch<4>(input_shape,
                                                           permutation);
    case 5:
        return detail::compute_permute_strides_dispatch<5>(input_shape,
                                                           permutation);
    case 6:
        return detail::compute_permute_strides_dispatch<6>(input_shape,
                                                           permutation);
    case 7:
        return detail::compute_permute_strides_dispatch<7>(input_shape,
                                                           permutation);
    case 8:
        return detail::compute_permute_strides_dispatch<8>(input_shape,
                                                           permutation);
    default:
        throw std::runtime_error("compute_permute_strides: unsupported ndims " +
                                 std::to_string(ndims));
    }
}

// Primary template for assign_permuted: stride-based permuted copy
// using the recursive compile-time loop. Works for any num_dimensions >= 2.
template <uint8_t num_dimensions>
void assign_permuted(dalotia_byte* __restrict__ dest,
                     dalotia_WeightFormat weight_output_format,
                     const int* const input_shape,
                     const dalotia_byte* __restrict__ tensor_start,
                     dalotia_WeightFormat weight_input_format,
                     const int* permutation) {
    auto [strides_permuted, total_size] =
        get_new_strides_permuted<num_dimensions>(input_shape, permutation);

    const size_t load_item_bytes =
        dalotia::sizeof_weight_format(weight_input_format);
    const size_t store_item_bytes =
        dalotia::sizeof_weight_format(weight_output_format);
    auto assign_fn =
        get_assignment_function(weight_output_format, weight_input_format);

    auto input_pointer = tensor_start;
    size_t store_index = 0;
    assign_permuted_loop<num_dimensions>(
        dest, input_pointer, store_index, store_item_bytes, load_item_bytes,
        input_shape, strides_permuted.data(), assign_fn);

    assert(static_cast<size_t>(std::distance(tensor_start, input_pointer)) /
               load_item_bytes ==
           total_size);
}

// specialization for 1d — just a linear copy (defined in .cpp).
template <>
void assign_permuted<1>(dalotia_byte* __restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const int* const input_shape,
                        const dalotia_byte* __restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int* permutation);

// Runtime dispatcher — maps num_dimensions to the compile-time template.
template <typename... Args>
void assign_permuted(uint8_t num_dimensions, Args&&... args) {
    switch (num_dimensions) {
    case 1:
        return assign_permuted<1>(std::forward<Args>(args)...);
    case 2:
        return assign_permuted<2>(std::forward<Args>(args)...);
    case 3:
        return assign_permuted<3>(std::forward<Args>(args)...);
    case 4:
        return assign_permuted<4>(std::forward<Args>(args)...);
    case 5:
        return assign_permuted<5>(std::forward<Args>(args)...);
    case 6:
        return assign_permuted<6>(std::forward<Args>(args)...);
    case 7:
        return assign_permuted<7>(std::forward<Args>(args)...);
    case 8:
        return assign_permuted<8>(std::forward<Args>(args)...);
    default:
        throw std::runtime_error(
            "dalotia: assign_permuted not implemented for " +
            std::to_string(num_dimensions) + " dimensions");
    }
}

}  // namespace dalotia