cmake_minimum_required(VERSION 3.24)

project(DALOTIA_CPP CXX C)

option(DALOTIA_CPP_BUILD_EXAMPLES "Build examples" ON)
option(DALOTIA_WITH_SAFETENSORS_CPP "use safetensors-cpp for tensor I/O" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MAKE_PROGRAM "make")

set(DALOTIA_CPP_SOURCES dalotia_assignment.cpp dalotia_formats.cpp safetensors_file.cpp)
add_library(dalotia_cpp ${DALOTIA_CPP_SOURCES})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)

# boost::multi instead of multi
include(FetchContent)
FetchContent_Declare(
  multi
  GIT_REPOSITORY https://github.com/correaa/boost-multi.git
  GIT_TAG f549688718345435aabaa1ac49c392965ca082a5
  FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(multi)
FetchContent_GetProperties(multi SOURCE_DIR MULTI_CPP_INCLUDE_DIR BINARY_DIR MULTI_CPP_LIBRARY)

target_include_directories(dalotia_cpp PUBLIC ${MULTI_CPP_INCLUDE_DIR} ${MULTI_DIR})
target_link_libraries(dalotia_cpp multi)

# safetensors-cpp
if (DALOTIA_WITH_SAFETENSORS_CPP)
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
  target_compile_definitions(dalotia-example PRIVATE "DALOTIA_CPP_NO_IMPLEMENTATION")
  target_link_libraries(dalotia-example dalotia_cpp)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

  if (DALOTIA_CPP_BUILD_C_API)
    add_executable(example-c example-c.c)
    target_compile_definitions(example-c PRIVATE "DALOTIA_C_NO_IMPLEMENTATION")
    target_link_libraries(example-c dalotia_c)
  endif ()
endif ()

enable_testing()
add_subdirectory(test)
