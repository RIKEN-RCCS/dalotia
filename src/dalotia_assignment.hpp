#pragma once
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory_resource>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dalotia_formats.hpp"

namespace dalotia {

std::pmr::vector<int> final_c_permutation_from_permutation_and_order(
    const int *permutation, dalotia_Ordering ordering, size_t num_dimensions);

template <typename InType, typename OutType>
std::function<void(dalotia_byte *__restrict__, const dalotia_byte *__restrict__)>
cpp_type_assignment(size_t store_item_bytes) {
    // if both types are builtins, cast input and assign the resulting bytes
    auto fcn = [store_item_bytes](dalotia_byte *__restrict__ output_bytes,
                                  const dalotia_byte *__restrict__ input_bytes) {
        auto input_cast =
            reinterpret_cast<const InType *__restrict__>(input_bytes);
        auto output_cast = static_cast<OutType>(*input_cast);
        assert(sizeof(output_cast) == store_item_bytes);
        auto copy_bytes =
            reinterpret_cast<dalotia_byte *__restrict__>(&output_cast);
        for (size_t j = 0; j < store_item_bytes; ++j) {
            output_bytes[j] = copy_bytes[j];
        }
    };
    return fcn;
}

std::function<void(dalotia_byte *__restrict__, const dalotia_byte *__restrict__)>
get_assignment_function(dalotia_WeightFormat weight_output_format,
                        dalotia_WeightFormat weight_input_format);

void assign_linearly(dalotia_byte *__restrict__ dest,
                     dalotia_WeightFormat weight_output_format,
                     size_t num_items,
                     const dalotia_byte *const __restrict__ tensor_start,
                     dalotia_WeightFormat weight_input_format);

template <uint8_t num_dimensions>
void assign_permuted(dalotia_byte *__restrict__ /*dest*/,
                     dalotia_WeightFormat /*weight_output_format*/,
                     const size_t *const /*input_shape*/,
                     const dalotia_byte *__restrict__ /*tensor_start*/,
                     dalotia_WeightFormat /*weight_input_format*/,
                     const int * /*permutation*/) {
    throw std::runtime_error("assign_permuted not yet implemented for " +
                             std::to_string(num_dimensions) + " dimensions");
}

// specialization for 1d
template <>
void assign_permuted<1>(dalotia_byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const size_t *const input_shape,
                        const dalotia_byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int *permutation);

// specialization for 2d
template <>
void assign_permuted<2>(dalotia_byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const size_t *const input_shape,
                        const dalotia_byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int *permutation);

// specialization for 3d
template <>
void assign_permuted<3>(dalotia_byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const size_t *const input_shape,
                        const dalotia_byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int *permutation);

// specialization for 4d
template <>
void assign_permuted<4>(dalotia_byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const size_t *const input_shape,
                        const dalotia_byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int *permutation);

// TODO use BOOST_PP_* to generate something like Julia @nloops macro for
// arbitrary dimensions?
// ->
// https://github.com/JuliaLang/julia/blob/master/base/multidimensional.jl#L1685
// https://stackoverflow.com/questions/77130743/boost-preprocessor-boost-pp-local-iterate-nested-loops

template <typename... Args>
void assign_permuted(uint8_t num_dimensions, Args &&...args) {
    if (num_dimensions == 1) {
        return assign_permuted<1>(std::forward<Args>(args)...);
    } else if (num_dimensions == 2) {
        return assign_permuted<2>(std::forward<Args>(args)...);
    } else if (num_dimensions == 3) {
        return assign_permuted<3>(std::forward<Args>(args)...);
    } else if (num_dimensions == 4) {
        return assign_permuted<4>(std::forward<Args>(args)...);
    } else {
        throw std::runtime_error("assign_permuted not yet implemented for " +
                                 std::to_string(num_dimensions) +
                                 " dimensions");
    }
}

}  // namespace dalotia