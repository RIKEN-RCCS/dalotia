#pragma once

#ifdef DALOTIA_WITH_CUFILE

#include "dalotia_datasource.hpp"

namespace dalotia {

// RAII guard for the cuFile (GDS) driver lifetime.
// multiple and previous calls to driver_open are OK.
class CuFileDriver {
   public:
    CuFileDriver();
    ~CuFileDriver();

    CuFileDriver(const CuFileDriver&) = delete;
    CuFileDriver& operator=(const CuFileDriver&) = delete;

    [[nodiscard]] static bool is_open() noexcept;
};

// Keep the old name as an alias for backwards compatibility.
using CuFileDriver = CuFileDriver;

// Returns true if `ptr` is a CUDA device pointer (cudaMalloc'd or managed).
// Returns false for host pointers (including cudaMallocHost pinned memory).
bool is_device_pointer(const void* ptr) noexcept;

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUFILE
