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

}  // namespace dalotia