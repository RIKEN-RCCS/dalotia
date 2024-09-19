#include "dalotia.hpp"
#include "safetensors.hh"

namespace dalotia {
class SafetensorsFile : public TensorFile {
   public:
    SafetensorsFile(std::string filename) : TensorFile(filename) {
        std::string warn, err;
        bool ret = safetensors::mmap_from_file(filename, &st, &warn, &err);
        if (warn.size() > 0) {
            std::cout << "safetensors-cpp WARN: " << warn << "\n";
        }
        if (ret == false) {
            std::cerr << "Failed to load: " << filename << "\n";
            std::cerr << "  ERR: " << err << "\n";
            throw std::runtime_error("Could not open file " + filename);
        }
        // Check if data_offsets are valid
        if (!safetensors::validate_data_offsets(st, err)) {
            std::cerr << "Invalid data_offsets\n";
            std::cerr << err << "\n";
            throw std::runtime_error("Invalid safetensors file " + filename);
        }
    }

    ~SafetensorsFile() {
        if (st.st_file != nullptr) {
            //?free  // TODO not sure where ownership is
        }
    }

    safetensors::tensor_t get_only_tensor() {
        safetensors::tensor_t tensor;
        assert(st.tensors.size() == 1);
        st.tensors.at(0, &tensor);
        return tensor;
    }

    safetensors::tensor_t get_tensor_from_name(std::string tensor_name) {
        if (tensor_name == "") {
            return get_only_tensor();
        }
        for (size_t i = 0; i < st.tensors.size(); i++) {
            std::string key = st.tensors.keys()[i];
            if (key == tensor_name) {
                safetensors::tensor_t tensor;
                st.tensors.at(i, &tensor);
                return tensor;
            }
        }
        throw std::runtime_error("Tensor " + tensor_name + " not found");
    }

    bool is_sparse(std::string tensor_name) override {
        return false;  // TODO figure out how sparsity works / could work
    }

    size_t get_num_dimensions(std::string tensor_name) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        return safetensor.shape.size();
    }

    size_t get_num_tensor_elements(std::string tensor_name) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        return safetensors::get_shape_size(safetensor);
    }

    std::array<int, 10> get_tensor_extents(
        std::string tensor_name = "") override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);
        std::array<int, 10> extents;
        for (size_t i = 0; i < safetensor.shape.size(); i++) {
            extents[i] = safetensor.shape[i];
        }
        for (size_t i = safetensor.shape.size(); i < 10; i++) {
            extents[i] = -1;
        }
        return extents;
    }

    void load_tensor_dense(std::string tensor_name,
                           dalotia_WeightFormat weightFormat,
                           dalotia_Ordering ordering, std::byte *tensor,
                           const int *permutation = nullptr) override {
        safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name);

        const uint8_t *databuffer = st.databuffer_addr;
        const size_t nitems = safetensors::get_shape_size(safetensor);
        const size_t file_item_bytes =
            safetensors::get_dtype_bytes(safetensor.dtype);
        const size_t load_item_bytes =
            dalotia::sizeof_weight_format(weightFormat);
        if (permutation != nullptr) {
            throw std::runtime_error("Permutation not yet implemented");
        } else {
            for (size_t i = 0; i < nitems; i++) {
                auto element_pointer = databuffer + safetensor.data_offsets[0] +
                                       i * file_item_bytes;
                // TODO cast from safetensor.dtype to weightFormat -- how ???
                // use gmpxx? use quantization things?
                assert(load_item_bytes == file_item_bytes);
                for (size_t j = 0; j < load_item_bytes; ++j) {
                    tensor[i * load_item_bytes + j] =
                        static_cast<std::byte>(element_pointer[j]);
                }
            }
        }
    }

    void load_tensor_sparse(std::string tensor_name,
                            dalotia_SparseFormat sparseFormat,
                            dalotia_WeightFormat weightFormat,
                            dalotia_Ordering ordering, std::byte *values,
                            int *first_indices, int *second_indices) override {
        throw std::runtime_error(
            "Sparse tensors for safetensors not implemented");
    }
    safetensors::safetensors_t st;
};

}  // namespace dalotia