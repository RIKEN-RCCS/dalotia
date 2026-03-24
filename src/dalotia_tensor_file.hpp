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
#include "dalotia_datasource.hpp"

#ifdef DALOTIA_WITH_CUDA
#include "dalotia_cuda.hpp"
#endif

namespace dalotia {

class TensorFile {
   public:
    explicit TensorFile(const std::string& /* filename */) {
        // bool opened = (this->file_ = fopen(filename.c_str(), "rb"));
        // if (!opened) {
        //     throw std::runtime_error("Could not open file " + filename);
        // }
    }

    TensorFile(const TensorFile&) = delete;
    TensorFile& operator=(const TensorFile&) = delete;
    TensorFile(TensorFile&&) = delete;
    TensorFile& operator=(TensorFile&&) = delete;

    virtual ~TensorFile() {
        // assert(this->file_ != nullptr);
        // fclose(this->file_);
    }

    [[nodiscard]] virtual const std::vector<std::string>& get_tensor_names()
        const {
        throw std::runtime_error(
            "get_tensor_names not implemented for this tensor type");
    }

    [[nodiscard]] virtual bool is_sparse(
        const std::string& /*tensor_name*/) const {
        throw std::runtime_error(
            "is_sparse not implemented for this tensor type");
        return false;
    }

    [[nodiscard]] size_t get_num_dimensions(
        const std::string& tensor_name) const {
        return this->get_tensor_extents(tensor_name).size();
    }

    // Returns the tensor extents, optionally permuted.
    [[nodiscard]] std::vector<int> get_tensor_extents(
        const std::string& tensor_name,
        const std::vector<int>& permutation = {}) const {
        auto extents = get_tensor_extents_raw(tensor_name);
        if (!permutation.empty()) {
            auto final_perm = final_c_permutation_from_permutation_and_order(
                permutation, dalotia_C_ordering, extents.size());
            if (!final_perm.empty()) {
                auto raw = extents;
                for (size_t i = 0; i < extents.size(); i++) {
                    extents[i] = raw[final_perm[i]];
                }
            }
        }
        return extents;
    }

    [[nodiscard]] size_t get_num_tensor_elements(
        const std::string& tensor_name) const {
        auto extents = this->get_tensor_extents(tensor_name);
        return std::accumulate(extents.begin(), extents.end(), size_t{1},
                               std::multiplies<size_t>());
    }

    [[nodiscard]] virtual size_t get_nnz(
        const std::string& /* tensor_name*/) const {
        // This function will read the file and return the number of non-zero
        // elements ? may take a while for dense tensors, only allow for sparse?
        throw std::runtime_error(
            "get_nnz not implemented for this tensor type");
        return 0;
    }

    [[nodiscard]] virtual std::vector<int> get_sparse_tensor_extents(
        const std::string& /*tensor_name*/,
        dalotia_SparseFormat /*format*/) const {
        // This function will (lazily) read the file and return the tensor
        // extents
        throw std::runtime_error(
            "get_sparse_tensor_extents not implemented for this tensor type");
        return {};
    }

    void load_tensor_dense(const std::string& tensor_name,
                           dalotia_WeightFormat weightFormat,
                           dalotia_Ordering ordering,
                           dalotia_byte* __restrict__ tensor,
                           const std::vector<int>& permutation = {}
#ifdef DALOTIA_WITH_CUDA
                           ,
                           cudaStream_t stream = 0
#endif
    );

    template <typename value_type = dalotia_byte>  //? or have no defaults?
    [[nodiscard]] std::pair<std::vector<int>, dalotia::vector<value_type>>
    load_tensor_dense(
        const std::string& tensor_name, dalotia_WeightFormat weight_format,
        dalotia_Ordering ordering = dalotia_C_ordering,
        const std::vector<int>& permutation = {}
#ifdef DALOTIA_WITH_CPP_PMR
        ,
        const std::pmr::polymorphic_allocator<dalotia_byte>& allocator =
            std::pmr::polymorphic_allocator<dalotia_byte>()
#endif  // DALOTIA_WITH_CPP_PMR
    ) {
        auto extents = this->get_tensor_extents(tensor_name, permutation);
        auto total_size = std::accumulate(extents.begin(), extents.end(), 1,
                                          std::multiplies<size_t>());
#ifdef DALOTIA_WITH_CPP_PMR
        dalotia::vector<value_type> tensor(allocator);
#else
        dalotia::vector<value_type> tensor;
#endif  // DALOTIA_WITH_CPP_PMR

        if constexpr (std::is_same_v<value_type, dalotia_byte>) {
            tensor.resize(total_size * sizeof_weight_format(weight_format));
        } else {
            if (dalotia::sizeof_weight_format(weight_format) !=
                sizeof(value_type)) {
                throw std::runtime_error(
                    "load_tensor_dense: weight format size does not match "
                    "value type size");
            }
            tensor.resize(total_size);
        }
        this->load_tensor_dense(tensor_name, weight_format, ordering,
                                reinterpret_cast<dalotia_byte*>(tensor.data()),
                                permutation);
        return {std::move(extents), std::move(tensor)};
    }

    template <typename value_type>
    [[nodiscard]] std::pair<std::vector<int>, dalotia::vector<value_type>>
    load_tensor_dense(
        const std::string& tensor_name,
        dalotia_Ordering ordering = dalotia_C_ordering,
        const std::vector<int>& permutation = {}
#ifdef DALOTIA_WITH_CPP_PMR
        ,
        const std::pmr::polymorphic_allocator<dalotia_byte>& allocator =
            std::pmr::polymorphic_allocator<dalotia_byte>()
#endif  // DALOTIA_WITH_CPP_PMR
    ) {
        // TODO is there an elegant way to map types to values?
        if constexpr (std::is_same_v<value_type, float>) {
            return this->load_tensor_dense<float>(tensor_name, dalotia_float_32,
                                                  ordering, permutation
#ifdef DALOTIA_WITH_CPP_PMR
                                                  ,
                                                  allocator
#endif  // DALOTIA_WITH_CPP_PMR
            );
        } else if constexpr (std::is_same_v<value_type, double>) {
            return this->load_tensor_dense<double>(
                tensor_name, dalotia_float_64, ordering, permutation
#ifdef DALOTIA_WITH_CPP_PMR
                ,
                allocator
#endif  // DALOTIA_WITH_CPP_PMR
            );
        } else {
            throw std::runtime_error(
                "load_tensor_dense cannot derive the weight format \
                    from the value type");
        }
    }

    virtual void load_tensor_sparse(const std::string& /*tensor_name */,
                                    dalotia_SparseFormat /*sparseFormat */,
                                    dalotia_WeightFormat /* weightFormat*/,
                                    dalotia_Ordering /* ordering */,
                                    dalotia_byte* __restrict__ /*values*/,
                                    int* __restrict__ /* first_indices*/,
                                    int* __restrict__ /* second_indices*/) {
        // This function will read the whole file and load the tensor into the
        // three arrays
        throw std::runtime_error(
            "load_tensor_sparse not implemented for this tensor type");
    }

    // Set the host data source. Subclasses call this in their constructor
    // to provide host-accessible access to the file's data section.
    void set_data_source(std::unique_ptr<DataSource> source) {
        data_source_ = std::move(source);
    }

    DataSource* data_source() const noexcept { return data_source_.get(); }

#ifdef DALOTIA_WITH_CUFILE
    // Set the GPU data source. Subclasses call this in their constructor
    // to enable direct file-to-device loading. Offsets must be
    // data-section-relative (same convention as the host data source).
    void set_gpu_data_source(std::unique_ptr<DataSource> source) {
        gpu_data_source_ = std::move(source);
    }

    DataSource* gpu_data_source() const noexcept {
        return gpu_data_source_.get();
    }
#endif  // DALOTIA_WITH_CUFILE

    // Information needed to read a tensor from the file.
    struct TensorInfo {
        const dalotia_byte* data;     // pointer to raw tensor bytes
        dalotia_WeightFormat format;  // format of the data in the file
        std::vector<int> shape;       // unpermuted shape
        size_t num_elements;          // total number of elements
    };

    // Subclasses override this to provide access to a tensor's raw data.
    [[nodiscard]] virtual TensorInfo get_tensor_info(
        const std::string& /*tensor_name*/) const {
        throw std::runtime_error(
            "get_tensor_info not implemented for this tensor type");
    }

    // Returns unpermuted extents. Subclasses override this.
    [[nodiscard]] virtual std::vector<int> get_tensor_extents_raw(
        const std::string& /*tensor_name*/) const {
        throw std::runtime_error(
            "get_tensor_extents_raw not implemented for this tensor type");
    }

    // Default implementation of host tensor loading using get_tensor_info.
    // Subclasses typically don't need to override this.
    virtual void load_tensor_dense_impl(const std::string& tensor_name,
                                        dalotia_WeightFormat weightFormat,
                                        dalotia_Ordering ordering,
                                        dalotia_byte* __restrict__ tensor,
                                        const std::vector<int>& permutation);

    std::unique_ptr<DataSource> data_source_;
#ifdef DALOTIA_WITH_CUFILE
    std::unique_ptr<DataSource> gpu_data_source_;
#endif  // DALOTIA_WITH_CUFILE

    virtual std::vector<const dalotia_byte*> get_mmap_tensor_pointers(
        const std::string& /*tensor_name*/) const {
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
inline std::string to_string(const Iterable& iterable) {
    std::string result;
    for (const auto& item : iterable) {
        if (!result.empty()) {
            result += ", ";
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>,
                                     std::string>) {
            result += item;  // for strings, just append
        } else {
            result +=
                std::to_string(item);  // for other types, convert to string
        }
    }
    return result;
}

}  // namespace dalotia
