// Compile-only test: verifies that dalotia's public headers compile
// with C++17 and don't accidentally require C++20 features.

#include "dalotia.hpp"
#include "dalotia_datasource.hpp"
#include "dalotia_tensor_file.hpp"
#include "dalotia_assignment.hpp"
#include "dalotia_formats.hpp"

// Optional headers (only when their backend is enabled)
#ifdef DALOTIA_WITH_SAFETENSORS_CPP
#include "dalotia_safetensors_file.hpp"
#endif
#ifdef DALOTIA_WITH_CUFILE
#include "dalotia_cufile.hpp"
#endif
#ifdef DALOTIA_WITH_TENSORFLOW
#include "dalotia_tensorflow_file.hpp"
#endif

// Minimal usage to force template instantiation
static_assert(__cplusplus >= 201703L, "C++17 required");
static_assert(__cplusplus < 202002L,
              "This test must be compiled with -std=c++17, not c++20");

int main() {
    return 0;
}
