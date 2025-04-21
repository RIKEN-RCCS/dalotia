#pragma once

typedef enum  // cannot be scoped to allow for C interface
{
    dalotia_CSR,
    dalotia_COO
} dalotia_SparseFormat;  //? compressed formats for d > 2? -> NO common ones
//  pytorch uses M+K, but M is always 2, K is dense
//  onnx also has only 2d sparse tensors, same for tensorflow
//  compressed-tensors (recent, based on safetensors): only bitmask compression,
//  but for arbitrary dimensions
//  ALTO https://github.com/IntelLabs/ALTO /
//  https://dl.acm.org/doi/abs/10.1145/3447818.3461703 could be interesting

typedef enum {
    dalotia_float_64,
    dalotia_float_32,
    dalotia_float_16,
    // dalotia_float_8, #TODO distinguish 4-3 and 5-2 etc.
    dalotia_bfloat_16,
    // dalotia_uint_64,
    dalotia_uint_32,
    dalotia_uint_16,
    dalotia_uint_8,
    // dalotia_int_64,
    dalotia_int_32,
    dalotia_int_16,
    dalotia_int_8,
    dalotia_int_2,
} dalotia_WeightFormat;

typedef enum {
    dalotia_C_ordering,  // row-major: last index is most contiguous
    dalotia_F_ordering,  // column-major: first index is most contiguous
} dalotia_Ordering;