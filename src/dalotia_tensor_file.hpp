#pragma once

#include <stdio.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "dalotia_formats.hpp"
#include "dalotia_assignment.hpp"

namespace dalotia {
class TensorFile {
   public:
    explicit TensorFile(const std::string &/* filename */) {
        // bool opened = (this->file_ = fopen(filename.c_str(), "rb"));
        // if (!opened) {
        //     throw std::runtime_error("Could not open file " + filename);
        // }
    }

    TensorFile(const TensorFile &) = delete;
    TensorFile &operator=(const TensorFile &) = delete;
    TensorFile(TensorFile &&) = delete;
    TensorFile &operator=(TensorFile &&) = delete;

    virtual ~TensorFile() {
        // assert(this->file_ != nullptr);
        // fclose(this->file_);
    }

    [[nodiscard]] virtual const std::vector<std::string> &get_tensor_names() const {
        throw std::runtime_error(
            "get_tensor_names not implemented for this tensor type");
    }

    [[nodiscard]] virtual bool is_sparse(const std::string &/*tensor_name*/) const {
        throw std::runtime_error(
            "is_sparse not implemented for this tensor type");
        return false;
    }

    [[nodiscard]] virtual size_t get_num_dimensions(const std::string &tensor_name) const {
        return this->get_tensor_extents(tensor_name).size();
    }

    [[nodiscard]] virtual std::vector<int> get_tensor_extents(
        const std::string &/*tensor_name*/,
        const std::vector<int>& /*permutation*/ = {}) const
    {
        throw std::runtime_error(
            "get_tensor_extents not implemented for this tensor type");
        return {};
    }

    [[nodiscard]] virtual size_t get_num_tensor_elements(const std::string &tensor_name) const {
        // ?
        auto extents = this->get_tensor_extents(tensor_name);
        return std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<size_t>());
    }

    [[nodiscard]] virtual size_t get_nnz(const std::string &/* tensor_name*/) const {
        // This function will read the file and return the number of non-zero
        // elements ? may take a while for dense tensors, only allow for sparse?
        throw std::runtime_error(
            "get_nnz not implemented for this tensor type");
        return 0;
    }

    [[nodiscard]] virtual std::vector<int> get_sparse_tensor_extents(
        const std::string &/*tensor_name*/, dalotia_SparseFormat /*format*/) const {
        // This function will (lazily) read the file and return the tensor
        // extents
        throw std::runtime_error(
            "get_sparse_tensor_extents not implemented for this tensor type");
        return {};
    }

    virtual void load_tensor_dense(const std::string &/*tensor_name */,
                                   dalotia_WeightFormat /*weightFormat */,
                                   dalotia_Ordering /* ordering */,
                                   dalotia_byte *__restrict__ /*tensor */,
                                   const std::vector<int>& /* permutation */ = {}) {
        // This function will read the whole file and load the tensor,
        // optionally transposing it according to the permutation
        throw std::runtime_error(
            "load_tensor_dense not implemented for this tensor type");
    }

    template <typename value_type = dalotia_byte>  //? or have no defaults?
    [[nodiscard]] std::pair<std::vector<int>, dalotia::vector<value_type>>
    load_tensor_dense(const std::string &tensor_name,
        dalotia_WeightFormat weight_format,
        dalotia_Ordering ordering = dalotia_C_ordering,
        const std::vector<int>& permutation = {}
#ifdef DALOTIA_WITH_CPP_PMR
        ,
        const std::pmr::polymorphic_allocator<dalotia_byte> &allocator =
            std::pmr::polymorphic_allocator<dalotia_byte>()
#endif  // DALOTIA_WITH_CPP_PMR
    ) {
        auto extents = this->get_tensor_extents(tensor_name, permutation);
        auto total_size = std::accumulate(extents.begin(), extents.end(),
                                          1, std::multiplies<size_t>());
#ifdef DALOTIA_WITH_CPP_PMR
        dalotia::vector<value_type> tensor(allocator);
#else
        dalotia::vector<value_type> tensor;
#endif  // DALOTIA_WITH_CPP_PMR

        if constexpr (std::is_same_v<value_type, dalotia_byte>) {
            tensor.resize(total_size * sizeof_weight_format(weight_format));
        } else {
            tensor.resize(total_size);
        }
        this->load_tensor_dense(tensor_name, weight_format, ordering,
            reinterpret_cast<dalotia_byte *>(tensor.data()), permutation);
        return std::make_pair(extents, tensor);
    }

    template <typename value_type>
    [[nodiscard]] std::pair<std::vector<int>, dalotia::vector<value_type>>
    load_tensor_dense(const std::string &tensor_name,
        dalotia_Ordering ordering = dalotia_C_ordering,
        const std::vector<int>& permutation = {}
#ifdef DALOTIA_WITH_CPP_PMR
        ,
        const std::pmr::polymorphic_allocator<dalotia_byte> &allocator =
            std::pmr::polymorphic_allocator<dalotia_byte>()
#endif  // DALOTIA_WITH_CPP_PMR
    ) {
        // TODO is there an elegant way to map types to values?
        if constexpr (std::is_same_v<value_type, float>) {
            return this->load_tensor_dense<float>(tensor_name, dalotia_float_32,
                ordering, permutation
#ifdef DALOTIA_WITH_CPP_PMR
                , allocator
#endif  // DALOTIA_WITH_CPP_PMR
            );
        } else if constexpr (std::is_same_v<value_type, double>) {
            return this->load_tensor_dense<double>(tensor_name, dalotia_float_64,
                ordering, permutation
#ifdef DALOTIA_WITH_CPP_PMR
                    , allocator
#endif  // DALOTIA_WITH_CPP_PMR
                );
        } else {
            throw std::runtime_error(
                "load_tensor_dense cannot derive the weight format \
                    from the value type");
        }

    }

    virtual void load_tensor_sparse(const std::string &/*tensor_name */,
                                    dalotia_SparseFormat /*sparseFormat */,
                                    dalotia_WeightFormat /* weightFormat*/,
                                    dalotia_Ordering /* ordering */,
                                    dalotia_byte *__restrict__ /*values*/,
                                    int *__restrict__ /* first_indices*/,
                                    int *__restrict__ /* second_indices*/) {
        // This function will read the whole file and load the tensor into the
        // three arrays
        throw std::runtime_error(
            "load_tensor_sparse not implemented for this tensor type");
    }

    virtual std::vector<const dalotia_byte*> get_mmap_tensor_pointers(
        const std::string &/*tensor_name*/) const {
        // This function will return the pointer(s) to the mmaped tensor
        // (single for a dense, potentially multiple for a sparse tensor);
        // empty if not implemented or not available (e.g. if not mmapped)
        return std::vector<const dalotia_byte*>();
    }

    // no private section to allow visibility from C
    // FILE *file_ = nullptr;
};

// helper function to output iterables
template <typename Iterable>
inline std::string to_string(const Iterable &iterable) {
    std::string result;
    for (const auto &item : iterable) {
        if (!result.empty()) {
            result += ", ";
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, std::string>) {
            result += item;  // for strings, just append
        } else {
            result += std::to_string(item);  // for other types, convert to string
        }
    }
    return result;
}

}  // namespace dalotia
