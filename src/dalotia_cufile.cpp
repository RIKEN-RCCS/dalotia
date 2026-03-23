#ifdef DALOTIA_WITH_CUFILE

#include "dalotia_cufile.hpp"

#include <stdexcept>
#include <string>

#include <cufile.h>

namespace dalotia {

CuFileDriver::CuFileDriver() {
    CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("cuFileDriverOpen failed with error " +
                                 std::to_string(status.err));
    }
}

CuFileDriver::~CuFileDriver() {
    cuFileDriverClose();
}

bool CuFileDriver::is_open() noexcept {
    return cuFileUseCount() > 0;
}

}  // namespace dalotia

#endif  // DALOTIA_WITH_CUFILE
