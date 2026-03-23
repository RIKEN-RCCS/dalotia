#include "dalotia_datasource.hpp"

#include <cstring>
#include <stdexcept>

#include <fcntl.h>   // for open() and O_RDONLY/O_DIRECT
#include <unistd.h>  // for close()

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

}  // namespace dalotia
