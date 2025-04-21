#pragma once

#include <cstdint>
#include <limits>
#include <map>

#include "dalotia_formats.h"

// #if __cpp_lib_byte //TODO why is github actions unhappy with this?
// using dalotia_byte = std::byte;
// #else
using dalotia_byte = unsigned char;
// #endif

namespace dalotia {

static_assert(sizeof(double) == 8);
static_assert(sizeof(float) == 4);
static_assert(sizeof(short) == 2);

template <dalotia_WeightFormat format>
constexpr int8_t sizeof_weight_format() {
    if constexpr (format == dalotia_float_64) {
        return 8;
    } else if constexpr (format == dalotia_float_32) {
        return 4;
    } else if constexpr (format == dalotia_float_16) {
        return 2;
    // } else if constexpr (format == dalotia_float_8) {
    //     return 1;
    } else if constexpr (format == dalotia_bfloat_16) {
        return 2;
    // } else if constexpr (format == dalotia_uint_64) {
    //     return 8;
    } else if constexpr (format == dalotia_uint_32) {
        return 4;
    } else if constexpr (format == dalotia_uint_16) {
        return 2;
    } else if constexpr (format == dalotia_uint_8) {
        return 1;
    // } else if constexpr (format == dalotia_int_64) {
    //     return 8;
    } else if constexpr (format == dalotia_int_32) {
        return 4;
    } else if constexpr (format == dalotia_int_16) {
        return 2;
    } else if constexpr (format == dalotia_int_8) {
        return 1;
    } else if constexpr (format == dalotia_int_2) {
        return 1;  // TODO a bit unhappy with this one, oneDNN also gives denominators
    }
}

// runtime version
int8_t sizeof_weight_format(dalotia_WeightFormat format);

const std::map<dalotia_WeightFormat, dalotia_WeightFormat>
    bfloat_compatible_float{
        //   {dalotia_bfloat_8, dalotia_float_16},
        {dalotia_bfloat_16, dalotia_float_32},
        //   {dalotia_bfloat_32, dalotia_float_64},
    };

}  // namespace dalotia