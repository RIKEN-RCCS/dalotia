#pragma once

#include "dalotia_formats.h"

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#include <stdbool.h>
#endif

// C interface

// file class made visible by a c struct, cf.
// https://isocpp.org/wiki/faq/mixing-c-and-cpp
typedef struct DalotiaTensorFile DalotiaTensorFile;

EXTERNC DalotiaTensorFile *dalotia_open_file(const char *filename);

EXTERNC void dalotia_close_file(DalotiaTensorFile *file);

EXTERNC int dalotia_sizeof_weight_format(dalotia_WeightFormat format);

EXTERNC bool dalotia_is_sparse(DalotiaTensorFile *file,
                               const char *tensor_name);

EXTERNC int dalotia_get_num_tensors(DalotiaTensorFile *file);

EXTERNC int dalotia_get_tensor_name(DalotiaTensorFile *file, int index,
                                    char *name);

EXTERNC int dalotia_get_num_dimensions(DalotiaTensorFile *file,
                                       const char *tensor_name);

EXTERNC int dalotia_get_num_tensor_elements(DalotiaTensorFile *file,
                                            const char *tensor_name);

EXTERNC int dalotia_get_nnz(DalotiaTensorFile *file, const char *tensor_name);

EXTERNC int dalotia_get_tensor_extents(DalotiaTensorFile *file,
                                       const char *tensor_name, int *extents);

EXTERNC int dalotia_get_sparse_tensor_extents(DalotiaTensorFile *file,
                                              const char *tensor_name,
                                              int *extents,
                                              dalotia_SparseFormat format);

EXTERNC int dalotia_load_tensor_dense(DalotiaTensorFile *file,
                                      const char *tensor_name, char *tensor,
                                      dalotia_WeightFormat format,
                                      dalotia_Ordering ordering);

EXTERNC int dalotia_load_tensor_dense_with_permutation(
    DalotiaTensorFile *file, const char *tensor_name, char *tensor,
    dalotia_WeightFormat format, dalotia_Ordering ordering,
    const int *permutation);

EXTERNC int dalotia_load_tensor_sparse(DalotiaTensorFile *file,
                                       const char *tensor_name, char *values,
                                       int *first_indices, int *second_indices,
                                       dalotia_SparseFormat format,
                                       dalotia_WeightFormat weightFormat,
                                       dalotia_Ordering ordering);

#undef EXTERNC