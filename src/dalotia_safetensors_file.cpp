#include "dalotia_safetensors_file.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"
#include "safetensors.hh"
#ifdef DALOTIA_WITH_CUFILE
#include "dalotia_cufile.hpp"
#endif

namespace dalotia {

safetensors::tensor_t get_only_tensor(const safetensors::safetensors_t& st) {
    safetensors::tensor_t tensor;
    assert(st.tensors.size() == 1);
    st.tensors.at(0, &tensor);
    return tensor;
}

safetensors::tensor_t get_tensor_from_name(
    const std::string& tensor_name, const safetensors::safetensors_t& st) {
    if (tensor_name.empty()) {
        return get_only_tensor(st);
    }
    for (size_t i = 0; i < st.tensors.size(); i++) {
        std::string key = st.tensors.keys()[i];
        if (key == tensor_name) {
            safetensors::tensor_t tensor;
            st.tensors.at(i, &tensor);
            return tensor;
        }
    }

    const std::string joined_keys = std::accumulate(
        st.tensors.keys().begin(), st.tensors.keys().end(), std::string(),
        [](const std::string& a, const std::string& b) -> std::string {
            return a + (a.length() > 0 ? "," : "") + b;
        });

    throw std::runtime_error("Tensor " + tensor_name +
                             " not found; available: " + joined_keys);
}

const std::vector<std::string>& SafetensorsFile::get_tensor_names() const {
    return st_.tensors.keys();
}

void SafetensorsFile::init_data_source() {
    // Wrap the parsed data buffer in a MemoryDataSource for host reads.
    // databuffer_addr points into the mmap'd region (file constructor) or
    // the caller-provided buffer (memory constructor).
    set_data_source(std::make_unique<MemoryDataSource>(st_.databuffer_addr,
                                                       st_.databuffer_size));
}

SafetensorsFile::SafetensorsFile(const std::string& filename)
    : TensorFile(filename) {
    // as far as I can tell, safetensors are saved in C order
    std::string warn, err;
    bool ret = safetensors::mmap_from_file(filename, &st_, &warn, &err);
    if (warn.size() > 0) {
        std::cout << "safetensors-cpp WARN: " << warn << "\n";
    }
    if (ret == false) {
        std::cerr << "Failed to load: " << filename << "\n";
        std::cerr << "  ERR: " << err << "\n";
        throw std::runtime_error("Could not open file " + filename);
    }
#ifndef NDEBUG
    // Check if data_offsets are valid
    if (!safetensors::validate_data_offsets(st_, err)) {
        std::cerr << "Invalid data_offsets\n";
        std::cerr << err << "\n";
        throw std::runtime_error("Invalid safetensors file " + filename);
    }
#endif  // NDEBUG
    init_data_source();
#ifdef DALOTIA_WITH_CUFILE
    // Only attempt to open a GDS data source if the cuFile driver is active.
    if (CuFileDriver::is_open()) {
        try {
            const size_t base_offset = 8 + st_.header_size;
            set_gpu_data_source(
                std::make_unique<GDSDataSource>(filename, base_offset));
        } catch (const std::exception& e) {
            std::cerr << "dalotia: GDS unavailable for " << filename << " ("
                      << e.what() << "), will use cudaMemcpy fallback\n";
        } catch (...) {
            std::cerr << "dalotia: GDS unavailable for " << filename
                      << " (unknown error), will use cudaMemcpy fallback\n";
        }
    }
#endif
}

SafetensorsFile::SafetensorsFile(const void* const address, size_t num_bytes)
    : TensorFile("") {
    // as far as I can tell, safetensors are saved in C order
    std::string warn, err;
    bool ret = safetensors::mmap_from_memory(
        static_cast<const uint8_t*>(address), num_bytes, "", &st_, &warn, &err);
    if (warn.size() > 0) {
        std::cout << "safetensors-cpp WARN: " << warn << "\n";
    }
    if (ret == false) {
        std::cerr << "  ERR: " << err << "\n";
        throw std::runtime_error("Could not load safetensors from address");
    }
#ifndef NDEBUG
    // Check if data_offsets are valid
    if (!safetensors::validate_data_offsets(st_, err)) {
        std::cerr << "Invalid data_offsets\n";
        std::cerr << err << "\n";
        throw std::runtime_error("Invalid safetensors address");
    }
#endif  // NDEBUG
    init_data_source();
}

SafetensorsFile::~SafetensorsFile() {
    if (st_.st_file != nullptr) {
        // delete st_.st_file;
    }
}

bool SafetensorsFile::is_sparse(const std::string& /*tensor_name*/) const {
    return false;
}

std::vector<int> SafetensorsFile::get_tensor_extents_raw(
    const std::string& tensor_name) const {
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
    return {safetensor.shape.begin(), safetensor.shape.end()};
}

TensorFile::TensorInfo SafetensorsFile::get_tensor_info(
    const std::string& tensor_name) const {
    if (!data_source_) {
        throw std::runtime_error(
            "SafetensorsFile::get_tensor_info: data source not initialized");
    }
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
    return {
        reinterpret_cast<const dalotia_byte*>(
            data_source_->host_data(safetensor.data_offsets[0])),
        safetensors_type_map.at(safetensor.dtype),
        {safetensor.shape.begin(), safetensor.shape.end()},
        safetensors::get_shape_size(safetensor),
    };
}

std::vector<const dalotia_byte*> SafetensorsFile::get_mmap_tensor_pointers(
    const std::string& tensor_name) const {
    safetensors::tensor_t safetensor = get_tensor_from_name(tensor_name, st_);
    auto* tensor_start = reinterpret_cast<const dalotia_byte* __restrict__>(
        data_source_->host_data(safetensor.data_offsets[0]));
    return std::vector<const dalotia_byte*>(1, tensor_start);
}
}  // namespace dalotia