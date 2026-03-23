#include "dalotia_assignment.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dalotia_formats.hpp"

namespace dalotia {

std::vector<int> final_c_permutation_from_permutation_and_order(
    const std::vector<int> &permutation, dalotia_Ordering ordering, size_t num_dimensions) {
    std::vector<int> final_permutation_in_c_order;
    if (permutation.empty()) {
        if (ordering == dalotia_Ordering::dalotia_F_ordering) {
            final_permutation_in_c_order.resize(num_dimensions);
            // assign reverse iota
            std::iota(final_permutation_in_c_order.rbegin(),
                        final_permutation_in_c_order.rend(), 0);
        }  // else leave empty
    } else {
        // find out if the permutation ranges from 0 to d-1 or 1 to d
        const auto [min, max] =
            std::minmax_element(permutation.begin(), permutation.end());
        if (*min == 0 && *max == static_cast<int>(num_dimensions - 1)) {
            final_permutation_in_c_order.assign(permutation.begin(),
                                                permutation.end());
        } else if (*min == 1 && *max == static_cast<int>(num_dimensions)) {
            final_permutation_in_c_order.resize(num_dimensions);
            std::transform(permutation.begin(), permutation.end(),
                            final_permutation_in_c_order.begin(),
                            [](int x) { return x - 1; });
        } else {
            throw std::runtime_error("Invalid permutation");
        }

        if (ordering == dalotia_Ordering::dalotia_F_ordering) {
            // invert both the numbers and the order
            std::transform(
                final_permutation_in_c_order.begin(),
                final_permutation_in_c_order.end(),
                final_permutation_in_c_order.begin(),
                [&num_dimensions](int x) { return num_dimensions - x - 1; });
            std::reverse(final_permutation_in_c_order.begin(),
                         final_permutation_in_c_order.end());
        } else {  // assume that 1-indexed permutations are only requested w/
                  // Fortran (remove if wrong assumption)
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

std::function<void(dalotia_byte *__restrict__,
                   const dalotia_byte *__restrict__)>
get_assignment_function(dalotia_WeightFormat weight_output_format,
                        dalotia_WeightFormat weight_input_format) {
    const size_t load_item_bytes =
        dalotia::sizeof_weight_format(weight_input_format);
    const size_t store_item_bytes =
        dalotia::sizeof_weight_format(weight_output_format);
    if (weight_input_format == weight_output_format) {
        // if they are the same, just assign them dalotia_byte by dalotia_byte
        assert(load_item_bytes == store_item_bytes);
        auto fcn = [load_item_bytes](
                       dalotia_byte *__restrict__ output_bytes,
                       const dalotia_byte *__restrict__ input_bytes) {
            for (size_t j = 0; j < load_item_bytes; ++j) {
                output_bytes[j] = input_bytes[j];
            }
        };
        return fcn;
    } else if (weight_input_format == dalotia_float_64 &&
               weight_output_format == dalotia_float_32) {
        // TODO abstract all the combinations, or copy paste this here
        // for all type combinations...
        // maybe std::type_index / typeid(...) + visitor could work?
        return cpp_type_assignment<double, float>(store_item_bytes);
    } else if (weight_input_format == dalotia_float_64 &&
               weight_output_format == dalotia_float_16) {
        return cpp_type_assignment<double, short>(store_item_bytes);
    } else if (weight_input_format == dalotia_float_32 &&
               weight_output_format == dalotia_float_16) {
        return cpp_type_assignment<float, short>(store_item_bytes);
    } else if (weight_input_format == dalotia_float_32 &&
               weight_output_format == dalotia_float_64) {
        return cpp_type_assignment<float, double>(store_item_bytes);
    } else if (weight_input_format == dalotia_float_16 &&
               weight_output_format == dalotia_float_64) {
        return cpp_type_assignment<short, double>(store_item_bytes);
    } else if (weight_input_format == dalotia_float_16 &&
               weight_output_format == dalotia_float_32) {
        return cpp_type_assignment<short, float>(store_item_bytes);
    } else if (weight_input_format == dalotia_uint_32 &&
               weight_output_format == dalotia_uint_16) {
        return cpp_type_assignment<uint32_t, uint16_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_uint_32 &&
               weight_output_format == dalotia_uint_16) {
        return cpp_type_assignment<uint32_t, uint16_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_uint_32 &&
               weight_output_format == dalotia_uint_8) {
        return cpp_type_assignment<uint32_t, uint8_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_uint_16 &&
               weight_output_format == dalotia_uint_8) {
        return cpp_type_assignment<uint16_t, uint8_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_int_32 &&
               weight_output_format == dalotia_int_16) {
        return cpp_type_assignment<int32_t, int16_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_int_32 &&
               weight_output_format == dalotia_int_16) {
        return cpp_type_assignment<int32_t, int16_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_int_32 &&
               weight_output_format == dalotia_int_8) {
        return cpp_type_assignment<int32_t, int8_t>(store_item_bytes);
    } else if (weight_input_format == dalotia_int_16 &&
               weight_output_format == dalotia_int_8) {
        return cpp_type_assignment<int16_t, int8_t>(store_item_bytes);
    } else {  // TODO chain builtin conversion and bfloat conversion?
        auto b_in = bfloat_compatible_float.find(weight_input_format);
        auto b_out = bfloat_compatible_float.find(weight_output_format);
        if (b_in != bfloat_compatible_float.end() &&
            b_in->second == weight_output_format) {
            // if the input format is bfloat and if the output format is
            // bfloat-compatible, assign and add zeros at the end (?)
            assert(2 * load_item_bytes == store_item_bytes);
            auto fcn = [load_item_bytes](
                           dalotia_byte *__restrict__ output_bytes,
                           const dalotia_byte *__restrict__ input_bytes) {
                for (size_t j = 0; j < load_item_bytes; ++j) {
                    output_bytes[j] = input_bytes[j];
                }
                for (size_t j = 0; j < load_item_bytes; ++j) {
                    output_bytes[j] = static_cast<dalotia_byte>(0);
                }
            };
            return fcn;
        } else if (b_out != bfloat_compatible_float.end() &&
                   b_out->second == weight_input_format) {
            // conversely, if the output format is bfloat and if the input
            // format is bfloat-compatible, assign only a few bytes and drop the
            // rest
            assert(load_item_bytes == 2 * store_item_bytes);
            auto fcn = [store_item_bytes](
                           dalotia_byte *__restrict__ output_bytes,
                           const dalotia_byte *__restrict__ input_bytes) {
                for (size_t j = 0; j < store_item_bytes; ++j) {
                    output_bytes[j] = input_bytes[j];
                }
            };
            return fcn;
        }
    }
    // use gmpxx? use floatx? use quantization things?
    throw std::runtime_error(
        "get_assignment_function: unsupported format combination");
}

void assign_linearly(dalotia_byte *__restrict__ dest,
                     dalotia_WeightFormat weight_output_format,
                     size_t num_items,
                     const dalotia_byte *const __restrict__ tensor_start,
                     dalotia_WeightFormat weight_input_format) {
    const size_t load_item_bytes =
        dalotia::sizeof_weight_format(weight_input_format);
    const size_t store_item_bytes =
        dalotia::sizeof_weight_format(weight_output_format);
    auto assign_function =
        get_assignment_function(weight_output_format, weight_input_format);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_items; ++i) {
        auto input_pointer = tensor_start + i * load_item_bytes;
        auto output_pointer = dest + i * store_item_bytes;
        assign_function(output_pointer, input_pointer);
    }
}

template <>
void assign_permuted<1>(dalotia_byte *__restrict__ dest,
                        dalotia_WeightFormat weight_output_format,
                        const int *const input_shape,
                        const dalotia_byte *__restrict__ tensor_start,
                        dalotia_WeightFormat weight_input_format,
                        [[maybe_unused]] const int *permutation) {
    assert(permutation[0] == 0);
    assign_linearly(dest, weight_output_format, input_shape[0], tensor_start,
                    weight_input_format);
}

// Explicit instantiations for dimensions 2..8.
template void assign_permuted<2>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);
template void assign_permuted<3>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);
template void assign_permuted<4>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);
template void assign_permuted<5>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);
template void assign_permuted<6>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);
template void assign_permuted<7>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);
template void assign_permuted<8>(dalotia_byte *, dalotia_WeightFormat,
    const int *, const dalotia_byte *, dalotia_WeightFormat, const int *);

}  // namespace dalotia