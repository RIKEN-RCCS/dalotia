#pragma once

#include <cstdint>
#include <limits>
#include <map>

#include "dalotia_formats.h"

#if __cpp_lib_byte
using dalotia_byte = std::byte;
#else
using dalotia_byte = unsigned char;
#endif

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