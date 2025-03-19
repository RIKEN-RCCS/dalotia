#pragma once
#include <array>
#include <string>

#include "dalotia_formats.hpp"
#include "safetensors.hh"
#include "dalotia_tensor_file.hpp"
#ifdef DALOTIA_WITH_CUDA
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h> // fdopen/close

// #include <cstdlib>
#include <cstring>
#include <iostream>

#include "cufile.h"
#endif // DALOTIA_WITH_CUDA

namespace dalotia {

const std::map<safetensors::dtype, dalotia_WeightFormat> safetensors_type_map{
    {safetensors::dtype::kFLOAT64, dalotia_WeightFormat::dalotia_float_64},
    {safetensors::dtype::kFLOAT32, dalotia_WeightFormat::dalotia_float_32},
    {safetensors::dtype::kFLOAT16, dalotia_WeightFormat::dalotia_float_16},
    {safetensors::dtype::kBFLOAT16, dalotia_WeightFormat::dalotia_bfloat_16},
    // {kBOOL, dalotia_bool},
    // {kUINT8, dalotia_uint_8},
    // {kINT8, dalotia_int_8},
    // {kUINT16, dalotia_uint_16},
    // {kINT32, dalotia_int_32},
    // {kUINT32, dalotia_uint_32},
    // {kINT64, dalotia_int_64},
    // {kUINT64, dalotia_uint_64},
    // {dalotia_float_8},
    // {dalotia_int_2},
};

class SafetensorsFile : public TensorFile {
   public:
    explicit SafetensorsFile(const std::string &filename);

    virtual ~SafetensorsFile();

    const std::vector<std::string> &get_tensor_names() const override;

    bool is_sparse(const std::string &tensor_name) const override;

    size_t get_num_dimensions(const std::string &tensor_name) const override;

    size_t get_num_tensor_elements(const std::string &tensor_name) const override;

    std::vector<int> get_tensor_extents(
        const std::string &tensor_name = "",
        const std::vector<int>& permutation = {}) const override;

    void load_tensor_dense(const std::string &tensor_name,
                           dalotia_WeightFormat weightFormat,
                           dalotia_Ordering ordering,
                           dalotia_byte *__restrict__ tensor,
                           const std::vector<int>& permutation = {}) override;

    std::vector<const dalotia_byte*> get_mmap_tensor_pointers(
        const std::string &tensor_name) const override;
    
    // cf. https://github.com/syoyo/safetensors-cpp/blob/main/safetensors.hh
    safetensors::safetensors_t st_;
};

#ifdef DALOTIA_WITH_CUDA
// based on sample code from
// https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#sample-program-with-cufile-apis

struct CuFileDriver{

    explicit CuFileDriver(const std::string& filename) {
        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("could not open " + filename + 
                                     " because of " + std::to_string(errno));
        }
        CUfileError_t status = cuFileDriverOpen();
        if (status.err != CU_FILE_SUCCESS) {
            close(fd_);
            throw std::runtime_error("cuFile driver failed to open on " + 
                                     filename + " because of " + std::to_string(status.err));
        }
    }
    
    CuFileDriver() = delete;
    CuFileDriver(const CuFileDriver&) = delete;
    CuFileDriver(CuFileDriver&&) = delete;
    CuFileDriver& operator=(const CuFileDriver&) = delete;
    CuFileDriver& operator=(CuFileDriver&&) = delete;
    ~CuFileDriver() {
        if (fd_ != 0) {
            close(fd_);
            cuFileDriverClose();
        }
    }

    int fd_ = 0;
};

struct CuFileDescriptorHandle{
    explicit CuFileDescriptorHandle(const dalotia::CuFileDriver & driver) {
        memset((void *)&cf_descr_, 0, sizeof(CUfileDescr_t));
        cf_descr_.handle.fd = driver.fd_;
        cf_descr_.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        CUfileError_t status = cuFileHandleRegister(&cf_handle_, &cf_descr_);
        if (status.err == CU_FILE_INVALID_FILE_OPEN_FLAG) {
            throw std::runtime_error("cuFile handle registration failed because of invalid flags");
        }
        if (status.err == CU_FILE_INVALID_VALUE) {
            throw std::runtime_error("cuFile handle registration failed because of invalid API args");
        }
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("cuFile handle registration failed because of " + std::to_string(status.err));
        }
    }

    CuFileDescriptorHandle() = delete;
    CuFileDescriptorHandle(const CuFileDescriptorHandle&) = delete;
    CuFileDescriptorHandle(CuFileDescriptorHandle&&) = delete;
    CuFileDescriptorHandle& operator=(const CuFileDescriptorHandle&) = delete;
    CuFileDescriptorHandle& operator=(CuFileDescriptorHandle&&) = delete;

    ~CuFileDescriptorHandle() {
        if (cf_handle_ != nullptr) {
            cuFileHandleDeregister(cf_handle_);
        }
    }

    CUfileDescr_t cf_descr_;
    CUfileHandle_t cf_handle_ = nullptr;
};

struct CuHostFileBuf{
    explicit CuHostFileBuf(size_t buff_size) {
        // cout << "Allocating CUDA buffer of " << buff_size << " bytes." << std::endl;
        unsigned int flags = cudaHostAllocDefault;
        int cuda_result = cudaHostAlloc(&hostPtr_base_, buff_size, flags);
        if (cuda_result != CUDA_SUCCESS) {
            throw std::runtime_error("cuFile buffer allocation failed because of " + std::to_string(cuda_result));
        }
        CUfileError_t status = cuFileBufRegister(hostPtr_base_, buff_size, 0);
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("cuFile buffer registration failed because of " + std::to_string(status.err));
        }
#ifndef NDEBUG
        // try zeroing the buffer
        memset(hostPtr_base_, 0, buff_size);
#endif
    }

    CuHostFileBuf() = delete;
    CuHostFileBuf(const CuHostFileBuf&) = delete;
    CuHostFileBuf(CuHostFileBuf&&) = delete;
    CuHostFileBuf& operator=(const CuHostFileBuf&) = delete;
    CuHostFileBuf& operator=(CuHostFileBuf&&) = delete;

    ~CuHostFileBuf() {
        if (hostPtr_base_ != nullptr){
            // release the GPU memory pinning
            CUfileError_t status = cuFileBufDeregister(hostPtr_base_);
            if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "CUDA file buffer deregister failed" << std::endl;
            }
            cudaFree(hostPtr_base_);
        }
    }

    void *hostPtr_base_ = nullptr;
};


class CudaSafetensorsFile : public TensorFile {
  public:
    explicit CudaSafetensorsFile(const std::string &filename) : TensorFile(filename) {
    // cf. https://github.com/syoyo/safetensors-cpp/blob/b63b28dff8bd011bfed2d1abc182498984f64d0b/safetensors.hh#L4403
    std::string warn, err;
    
    cuDriver_ = std::make_unique<CuFileDriver>(filename);
    cuFileDescriptorHandle_ = std::make_unique<CuFileDescriptorHandle>(*cuDriver_);

    // first, read only header into host memory
    // cf. https://github.com/syoyo/safetensors-cpp/blob/b63b28dff8bd011bfed2d1abc182498984f64d0b/safetensors.hh#L4256

    bool header_fits = false;
    // optimistically assume the header fits into 0.5MiB, increase if it doesn't
    size_t buffer_size = 512*1024;
    uint64_t header_size = 0;
    do {
        cuHostFileBufHeader_ = std::make_unique<CuHostFileBuf>(buffer_size);
        std::cout << "buffer" << std::endl;
        // read first 16 bytes to know the header size
        ssize_t return_value = cuFileRead(cuFileDescriptorHandle_->cf_handle_, cuHostFileBufHeader_->hostPtr_base_, 16, 0, 0);
        if (return_value != 16) {
            throw std::runtime_error("could not read cuFile because " + std::to_string(return_value));
        }
        header_size = *reinterpret_cast<uint64_t*>(cuHostFileBufHeader_->hostPtr_base_);
        if (header_size + 8 <= buffer_size) {
            header_fits = true;
        } else {
            buffer_size = header_size + 8;
            std::cout << "resizing host buffer" << std::endl;
        }
    } while (!header_fits);
    // read the header
    ssize_t return_value = cuFileRead(cuFileDescriptorHandle_->cf_handle_, cuHostFileBufHeader_->hostPtr_base_, header_size + 8, 0, 0);
    if (static_cast<uint64_t>(return_value) != header_size + 8) {
        throw std::runtime_error("could not read cuFile because " + std::to_string(return_value));
    }

    std::unique_ptr<CuFileDriver> cuDriver_ = nullptr;
    std::unique_ptr<CuFileDescriptorHandle> cuFileDescriptorHandle_ = nullptr;
    // CuDeviceFileBuf cuFileBuf_;
    std::unique_ptr<CuHostFileBuf> cuHostFileBufHeader_;
    safetensors::safetensors_t st_;
 };
 #endif // def DALOTIA_WITH_CUDA

}  // namespace dalotia