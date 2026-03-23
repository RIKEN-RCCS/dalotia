#pragma once

#ifdef DALOTIA_WITH_CUFILE

#include "dalotia_datasource.hpp"

namespace dalotia {

// RAII guard for the cuFile (GDS) driver lifetime.
// Multiple and previous calls to driver_open are OK.
class CuFileDriver {
   public:
    CuFileDriver();
    ~CuFileDriver();

    CuFileDriver(const CuFileDriver&) = delete;
    CuFileDriver& operator=(const CuFileDriver&) = delete;

    [[nodiscard]] static bool is_open() noexcept;
};

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUFILE
