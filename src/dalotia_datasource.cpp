#include "dalotia_datasource.hpp"

#include <cstring>
#include <stdexcept>

#include <fcntl.h>   // for open() and O_RDONLY/O_DIRECT
#include <unistd.h>  // for close()

#ifdef DALOTIA_WITH_CUFILE
#include <cerrno>
#include <cuda_runtime.h>
#include <cufile.h>
#endif  // DALOTIA_WITH_CUFILE

namespace dalotia {

const uint8_t* DataSource::host_data(size_t /*offset*/) const {
    return nullptr;
}

MemoryDataSource::MemoryDataSource(const uint8_t* base, size_t size)
    : base_(base), size_(size) {}

void MemoryDataSource::read_into(size_t offset, size_t nbytes, void* dest) {
    if (offset + nbytes > size_) {
        throw std::runtime_error(
            "MemoryDataSource::read_into: read past end of buffer");
    }
    std::memcpy(dest, base_ + offset, nbytes);
}

const uint8_t* MemoryDataSource::host_data(size_t offset) const {
    return base_ + offset;
}

class ScopedFd {
   public:
    explicit ScopedFd(const std::string& path, int flags)
        : fd_(open(path.c_str(), flags)) {
        if (fd_ < 0) {
            throw std::runtime_error("failed to open file " + path + ": " +
                                     std::string(strerror(errno)));
        }
    }
    ~ScopedFd() {
        if (fd_ >= 0)
            close(fd_);
    }
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    int get() const noexcept { return fd_; }

   private:
    int fd_;
};

#ifdef DALOTIA_WITH_CUFILE

class ScopedCuFileHandle {
   public:
    explicit ScopedCuFileHandle(int fd) {
        CUfileDescr_t descr{};
        descr.handle.fd = fd;
        descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        CUfileError_t status = cuFileHandleRegister(&handle_, &descr);
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("cuFileHandleRegister failed with error " +
                                     std::to_string(status.err));
        }
    }
    ~ScopedCuFileHandle() { cuFileHandleDeregister(handle_); }
    ScopedCuFileHandle(const ScopedCuFileHandle&) = delete;
    ScopedCuFileHandle& operator=(const ScopedCuFileHandle&) = delete;
    CUfileHandle_t get() const noexcept { return handle_; }

   private:
    CUfileHandle_t handle_;
};

class ScopedCuFileBuf {
   public:
    ScopedCuFileBuf(void* d_ptr, size_t nbytes) : d_ptr_(d_ptr) {
        CUfileError_t status = cuFileBufRegister(d_ptr_, nbytes, 0);
        if (status.err != CU_FILE_SUCCESS) {
            throw std::runtime_error("cuFileBufRegister failed with error " +
                                     std::to_string(status.err));
        }
    }
    ~ScopedCuFileBuf() { cuFileBufDeregister(d_ptr_); }
    ScopedCuFileBuf(const ScopedCuFileBuf&) = delete;
    ScopedCuFileBuf& operator=(const ScopedCuFileBuf&) = delete;

   private:
    void* d_ptr_;
};

// GDSDataSource pimpl

struct GDSDataSource::Impl {
    ScopedFd fd;
    ScopedCuFileHandle handle;

    explicit Impl(const std::string& filepath)
        : fd(filepath, O_RDONLY | O_DIRECT), handle(fd.get()) {}
};

GDSDataSource::GDSDataSource(const std::string& filepath, size_t base_offset)
    : impl_(std::make_unique<Impl>(filepath)), base_offset_(base_offset) {}

GDSDataSource::~GDSDataSource() = default;

void GDSDataSource::read_into(size_t offset, size_t nbytes, void* d_ptr) {
    // Buffer registration is per-call: each tensor goes to a different
    // device pointer, but the file handle is reused.
    ScopedCuFileBuf buf_guard(d_ptr, nbytes);

    const size_t file_offset = base_offset_ + offset;
    ssize_t bytes_read =
        cuFileRead(impl_->handle.get(), d_ptr, nbytes, file_offset, 0);
    if (bytes_read < 0) {
        throw std::runtime_error("cuFileRead failed with error " +
                                 std::to_string(bytes_read));
    }
    if (static_cast<size_t>(bytes_read) != nbytes) {
        throw std::runtime_error("cuFileRead: short read (" +
                                 std::to_string(bytes_read) + " of " +
                                 std::to_string(nbytes) + " bytes)");
    }
}

#endif  // DALOTIA_WITH_CUFILE

}  // namespace dalotia
