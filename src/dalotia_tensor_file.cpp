#include "dalotia_tensor_file.hpp"

namespace dalotia {

void TensorFile::load_tensor_dense(const std::string& tensor_name,
                                   dalotia_WeightFormat weightFormat,
                                   dalotia_Ordering ordering,
                                   dalotia_byte* __restrict__ tensor,
                                   const std::vector<int>& permutation) {
    // Host path — delegate to the format-specific implementation.
    load_tensor_dense_impl(tensor_name, weightFormat, ordering, tensor,
                           permutation);
}

}  // namespace dalotia
