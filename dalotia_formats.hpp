#pragma once

#include <cstdint>
#include <limits>
#include <map>

enum dalotia_SparseFormat  // cannot be scoped to allow for C interface
{
    dalotia_CSR,
    dalotia_COO
};  //? compressed formats for d > 2? -> NO common ones
//  pytorch uses M+K, but M is always 2, K is dense
//  onnx also has only 2d sparse tensors, same for tensorflow
//  compressed-tensors (recent, based on safetensors): only bitmask compression,
//  but for arbitrary dimensions
//  ALTO https://github.com/IntelLabs/ALTO /
//  https://dl.acm.org/doi/abs/10.1145/3447818.3461703 could be interesting

enum dalotia_WeightFormat {
    dalotia_float_64,
    dalotia_float_32,
    dalotia_float_16,
    dalotia_float_8,
    dalotia_bfloat_16,
    dalotia_int_8,
    dalotia_int_2,
};

enum dalotia_Ordering {
    dalotia_C_ordering,  // row-major: last index is most contiguous
    dalotia_F_ordering,  // column-major: first index is most contiguous
};

namespace dalotia {

template <dalotia_WeightFormat format>
constexpr int8_t sizeof_weight_format() {
    if constexpr (format == dalotia_float_64) {
        return 8;
    } else if constexpr (format == dalotia_float_32) {
        return 4;
    } else if constexpr (format == dalotia_float_16) {
        return 2;
    } else if constexpr (format == dalotia_float_8) {
        return 1;
    } else if constexpr (format == dalotia_bfloat_16) {
        return 2;
    } else if constexpr (format == dalotia_int_8) {
        return 1;
    } else if constexpr (format == dalotia_int_2) {
        return 1;  // TODO a bit unhappy with this one
    }
}

// runtime version
int8_t sizeof_weight_format(dalotia_WeightFormat format);

const std::map<dalotia_WeightFormat, dalotia_WeightFormat> bfloat_compatible_float{
    //   {dalotia_bfloat_8, dalotia_float_16},
    {dalotia_bfloat_16, dalotia_float_32},
    //   {dalotia_bfloat_32, dalotia_float_64},
};

}  // namespace dalotia