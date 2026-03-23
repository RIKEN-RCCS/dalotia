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

}  // namespace dalotia
