cmake_minimum_required(VERSION 3.24)

project(DALOTIA_CPP CXX C)

option(DALOTIA_CPP_BUILD_EXAMPLES "Build examples" ON)
option(DALOTIA_BUILD_TESTS "Build tests" ON)
option(DALOTIA_WITH_SAFETENSORS_CPP "use safetensors-cpp for tensor I/O" ON)
option(DALOTIA_WITH_FORTRAN "Build Fortran interface" ON)
if (DALOTIA_WITH_FORTRAN)
    enable_language(Fortran)
    set( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif (DALOTIA_WITH_FORTRAN)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_MAKE_PROGRAM "make")

if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the build type, options are: Debug Release RelWithDebInfo")
endif()
message(STATUS "CMAKE BUILD TYPE: ${CMAKE_BUILD_TYPE}")

add_subdirectory(src) # target dalotia_cpp is generated here

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(dalotia_cpp OpenMP::OpenMP_CXX)
endif ()

# copy example / test files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

# safetensors-cpp
if (DALOTIA_WITH_SAFETENSORS_CPP)
  include(FetchContent)
  FetchContent_Declare(
    safetensors-cpp
    GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp.git
    GIT_TAG 82571c11c2dc47a6663fc9ec0251241baf339c16
    FIND_PACKAGE_ARGS # allow finding system installations
  )
  set(SAFETENSORS_CPP_CXX_EXCEPTIONS true)
  FetchContent_MakeAvailable(safetensors-cpp)
  FetchContent_GetProperties(safetensors-cpp SOURCE_DIR SAFETENSORS_CPP_INCLUDE_DIR BINARY_DIR SAFETENSORS_CPP_LIBRARY)
  # add dependecy from safetensors_cpp to sources 
  # so that everything that links to safetensors_cpp will also get the headers
  target_include_directories(safetensors_cpp PUBLIC  
      $<BUILD_INTERFACE:${SAFETENSORS_CPP_INCLUDE_DIR}>
    )

  target_include_directories(dalotia_cpp PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR} ${safetensors-cpp_DIR})
  target_link_libraries(dalotia_cpp safetensors_cpp)
  target_compile_options(dalotia_cpp PUBLIC "-DDALOTIA_WITH_SAFETENSORS_CPP")
endif (DALOTIA_WITH_SAFETENSORS_CPP)

if (DALOTIA_CPP_BUILD_EXAMPLES)
  add_executable(dalotia-example example.cpp)
  target_link_libraries(dalotia-example dalotia_cpp)
endif (DALOTIA_CPP_BUILD_EXAMPLES)

if (DALOTIA_BUILD_TESTS)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  enable_testing()
  add_subdirectory(test)
endif (DALOTIA_BUILD_TESTS)