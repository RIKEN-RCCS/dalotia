#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dalotia_formats.hpp"

namespace dalotia {

std::pmr::vector<int> final_c_permutation_from_permutation_and_order(
    const int *permutation, dalotia_Ordering ordering, size_t num_dimensions) {
    std::pmr::vector<int> final_permutation_in_c_order;
    if (permutation == nullptr) {
        if (ordering == dalotia_Ordering::dalotia_F_ordering) {
            final_permutation_in_c_order.resize(num_dimensions);
            // assign reverse iota
            std::iota(final_permutation_in_c_order.rbegin(),
                      final_permutation_in_c_order.rend(), 0);
        }  // else leave empty
    } else {
        // find out if the permutation ranges from 0 to d-1 or 1 to d
        const auto [min, max] =
            std::minmax_element(permutation, permutation + num_dimensions);
        if (*min == 0 && *max == num_dimensions - 1) {
            final_permutation_in_c_order.assign(permutation,
                                                permutation + num_dimensions);
        } else if (*min == 1 && *max == num_dimensions) {
            final_permutation_in_c_order.resize(num_dimensions);
            std::transform(permutation, permutation + num_dimensions,
                           final_permutation_in_c_order.begin(),
                           [](int x) { return x - 1; });
        } else {
            throw std::runtime_error("Invalid permutation");
        }

        if (ordering == dalotia_Ordering::dalotia_F_ordering) {
            std::reverse(final_permutation_in_c_order.begin(),
                         final_permutation_in_c_order.end());
        } else {  // assume that 1-indexed permutations are only requested w/
                  // Fortran
                  // (remove if wrong assumption)
            assert(final_permutation_in_c_order[0] == permutation[0]);
        }
        {
            auto sorted_permutation = final_permutation_in_c_order;
            std::sort(sorted_permutation.begin(), sorted_permutation.end());
            const auto duplicate = std::adjacent_find(
                sorted_permutation.begin(), sorted_permutation.end());

            if (duplicate != sorted_permutation.end()) {
                throw std::runtime_error("dalotia: Invalid permutation");
            }
        }

        // if it is the same as iota, we can leave it empty
        if (std::is_sorted(final_permutation_in_c_order.begin(),
                           final_permutation_in_c_order.end())) {
            final_permutation_in_c_order.clear();
        }
    }
    return final_permutation_in_c_order;
}

void assign_linearly(std::byte *__restrict__ dest,
                     dalotia_WeightFormat weight_output_format,
                     size_t num_items,
                     const std::byte *const __restrict__ tensor_start,
                     dalotia_WeightFormat weight_input_format) {
    const size_t file_item_bytes =
        dalotia::sizeof_weight_format(weight_input_format);
    const size_t load_item_bytes =
        dalotia::sizeof_weight_format(weight_output_format);
    for (size_t i = 0; i < num_items; i++) {
        auto element_pointer = tensor_start + i * file_item_bytes;
        // TODO cast from safetensor.dtype to weightFormat -- how ???
        // use gmpxx? use quantization things?
        assert(load_item_bytes == file_item_bytes);
        for (size_t j = 0; j < load_item_bytes; ++j) {
            dest[i * load_item_bytes + j] =
                static_cast<std::byte>(element_pointer[j]);
        }
    }
}

template <uint8_t num_dimensions>
void assign_permuted(std::byte *__restrict__ dest,
                     dalotia_WeightFormat weight_output_format,
                     const size_t *const input_shape,
                     const std::byte *__restrict__ tensor_start,
                     dalotia_WeightFormat weight_input_format,
                     const int *permutation) {
    throw std::runtime_error("assign_permuted not yet implemented for " +
                             std::to_string(num_dimensions) + " dimensions");
}

// specialization for 2d
template <>
void assign_permuted<2>(std::byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const size_t *const input_shape,
                        const std::byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int *permutation) {
    constexpr int num_dimensions = 2;
    auto desired_shape = std::vector<size_t>(num_dimensions);
    size_t total_size = 1;
    for (size_t i = 0; i < num_dimensions; ++i) {
        desired_shape[i] = input_shape[permutation[i]];
        total_size *= desired_shape[i];
    }
    assert(permutation[0] == 1);
    assert(permutation[1] == 0);
    const size_t file_item_bytes =
        dalotia::sizeof_weight_format(weight_input_format);  // TODO casting
    size_t load_index = 0;
    for (size_t i = 0; i < input_shape[1]; ++i) {
        for (size_t j = 0; j < input_shape[0]; ++j) {
            auto store_index = j * input_shape[1] + i;
            for (size_t k = 0; k < file_item_bytes; ++k) {
                auto element_pointer =
                    tensor_start + load_index * file_item_bytes;
                dest[store_index * file_item_bytes + j] =
                    static_cast<std::byte>(element_pointer[k]);
            }
            ++load_index;
        }
    }

    assert(load_index == total_size);
}

// specialization for 3d
template <>
void assign_permuted<3>(std::byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const size_t *const input_shape,
                        const std::byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        const int *permutation) {
    constexpr int num_dimensions = 3;
    auto desired_shape = std::vector<size_t>(num_dimensions);
    size_t total_size = 1;
    for (size_t i = 0; i < num_dimensions; ++i) {
        desired_shape[i] = input_shape[permutation[i]];
        total_size *= desired_shape[i];
    }
    auto new_strides = std::array<size_t, num_dimensions>();
    // C order -> last dimension is the most contiguous -> rbegin
    std::exclusive_scan(desired_shape.rbegin(), desired_shape.rend(),
                        new_strides.rbegin(), 1, std::multiplies<>{});
    auto new_strides_permuted = new_strides;
    for (size_t i = 0; i < num_dimensions; ++i) {
        new_strides_permuted[i] = new_strides[permutation[i]];
    }

    const size_t file_item_bytes =
        dalotia::sizeof_weight_format(weight_input_format);  // TODO casting
    size_t load_index = 0;
    std::array<size_t, num_dimensions> load_index_array{0, 0, 0};
    auto &[i, j, k] = load_index_array;
    // this is sequential load / permuted store
    // untested whether it would be faster the other way around
    for (i = 0; i < input_shape[0]; ++i) {
        for (j = 0; j < input_shape[1]; ++j) {
            for (k = 0; k < input_shape[2]; ++k) {
                auto store_index = std::inner_product(
                    new_strides_permuted.begin(), new_strides_permuted.end(),
                    load_index_array.begin(), 0);
                assert(store_index < total_size);

                for (size_t l = 0; l < file_item_bytes; ++l) {
                    auto element_pointer =
                        tensor_start + load_index * file_item_bytes;
                    dest[store_index * file_item_bytes + l] =
                        static_cast<std::byte>(element_pointer[l]);
                }
                ++load_index;
            }
        }
    }

    assert(load_index == total_size);
}

template <typename... Args>
void assign_permuted(uint8_t num_dimensions, Args &&...args) {
    if (num_dimensions == 2) {
        return assign_permuted<2>(std::forward<Args>(args)...);
    } else if (num_dimensions == 3) {
        return assign_permuted<3>(std::forward<Args>(args)...);
    } else {
        throw std::runtime_error("assign_permuted not yet implemented for " +
                                 std::to_string(num_dimensions) +
                                 " dimensions");
    }
}

// TODO use BOOST_PP_* to generate something like Julia @nloops macro?
// for all dimensions
// ->
// https://github.com/JuliaLang/julia/blob/master/base/multidimensional.jl#L1685
// https://stackoverflow.com/questions/77130743/boost-preprocessor-boost-pp-local-iterate-nested-loops

}  // namespace dalotia