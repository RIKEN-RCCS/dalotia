#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace dalotia {

// Abstract interface for reading data from a tensor file's data section.
// A DataSource is opened once per file and reused for all tensor reads.
// All offsets are relative to the start of the data section.
class DataSource {
   public:
    virtual ~DataSource() = default;

    // Copy `nbytes` starting at byte `offset` into `dest`.
    virtual void read_into(size_t offset, size_t nbytes, void* dest) = 0;

    // Return a host-accessible pointer to the data at byte `offset`, or
    // nullptr if the data is not host-accessible (e.g. lives on a GPU).
    [[nodiscard]] virtual const uint8_t* host_data(size_t offset) const;
};

// Data source backed by a host-accessible memory region (mmap or user buffer).
class MemoryDataSource : public DataSource {
   public:
    MemoryDataSource(const uint8_t* base, size_t size);

    void read_into(size_t offset, size_t nbytes, void* dest) override;
    [[nodiscard]] const uint8_t* host_data(size_t offset) const override;

   private:
    const uint8_t* base_;
    size_t size_;
};

#ifdef DALOTIA_WITH_CUFILE
// GDS-backed data source.
// Individual read_into() calls register/deregister the destination device
// buffer (since each tensor goes to a different cudaMalloc'd pointer).
//
// `base_offset` is the byte offset from the start of the file to the data
// section (e.g. 8 + header_size for safetensors).
class GDSDataSource : public DataSource {
   public:
    GDSDataSource(const std::string& filepath, size_t base_offset);
    ~GDSDataSource() override;

    GDSDataSource(const GDSDataSource&) = delete;
    GDSDataSource& operator=(const GDSDataSource&) = delete;

    void read_into(size_t offset, size_t nbytes, void* d_ptr) override;
    // host_data() returns nullptr — GDS data is not host-accessible.

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t base_offset_;
};
#endif  // DALOTIA_WITH_CUFILE

}  // namespace dalotia
