#include "dalotia_formats.hpp"

#include <stdexcept>

namespace dalotia {

int8_t sizeof_weight_format(dalotia_WeightFormat format) {
    // TODO is there a nicer (visitor-like) way to do this?
    switch (format) {
        case dalotia_float_64:
            return sizeof_weight_format<dalotia_float_64>();
        case dalotia_float_32:
            return sizeof_weight_format<dalotia_float_32>();
        case dalotia_float_16:
            return sizeof_weight_format<dalotia_float_16>();
        case dalotia_float_8:
            return sizeof_weight_format<dalotia_float_8>();
        case dalotia_bfloat_16:
            return sizeof_weight_format<dalotia_bfloat_16>();
        case dalotia_int_8:
            return sizeof_weight_format<dalotia_int_8>();
        case dalotia_int_2:
            return sizeof_weight_format<dalotia_int_2>();

        default:
            throw std::runtime_error("Invalid weight format");
            return std::numeric_limits<int8_t>::min();
    }
}
}  // namespace dalotia