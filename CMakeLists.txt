cmake_minimum_required(VERSION 3.24)

project(DALOTIA CXX C)

option(DALOTIA_CPP_BUILD_EXAMPLES "Build examples" ON)
option(DALOTIA_BUILD_TESTS "Build tests" ON)
option(DALOTIA_WITH_CPP_PMR "use polymorphic memory resources (pmr) C++17 feature for dalotia" ON)
option(DALOTIA_WITH_OPENMP "Build with OpenMP support" OFF)
option(DALOTIA_WITH_SAFETENSORS_CPP "use safetensors-cpp for tensor I/O" ON)
option(DALOTIA_WITH_TENSORFLOW "use the Tensorflow C backend for tensor I/O" OFF)
option(DALOTIA_WITH_FORTRAN "Build Fortran interface" ON)
if (DALOTIA_WITH_FORTRAN)
    enable_language(Fortran)
    set( CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # cf. https://enccs.github.io/cmake-workshop/cxx-fortran/
    include(FortranCInterface)
    FortranCInterface_VERIFY()
    FortranCInterface_VERIFY(CXX)
endif (DALOTIA_WITH_FORTRAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules")

# if this is empty, will be set at config time
# (build with --config Release or --config Debug etc.)
message(STATUS "CMAKE BUILD TYPE: ${CMAKE_BUILD_TYPE}")

# copy example / test files to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

# safetensors-cpp
if (DALOTIA_WITH_SAFETENSORS_CPP)
  if (NOT DEFINED safetensors-cpp_DIR OR safetensors-cpp_DIR MATCHES "fetch")
    include(FetchContent)
    FetchContent_Declare(
      safetensors-cpp
      GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp.git
      GIT_TAG a88953c981c6773760540592fa97f04619a8f825
      OVERRIDE_FIND_PACKAGE
    )
    set(SAFETENSORS_CPP_CXX_EXCEPTIONS true) #TODO maybe depend on language?
    set(SAFETENSORS_CPP_BUILD_EXAMPLES false)
    set(SAFETENSORS_CPP_BUILD_C_API false)
    FetchContent_MakeAvailable(safetensors-cpp)
    FetchContent_GetProperties(safetensors-cpp SOURCE_DIR SAFETENSORS_CPP_INCLUDE_DIR BINARY_DIR SAFETENSORS_CPP_LIBRARY)
    # add dependecy from safetensors_cpp library to its sources 
    # so that everything that links to safetensors_cpp will also get the headers
    target_include_directories(safetensors_cpp PUBLIC  
        $<BUILD_INTERFACE:${SAFETENSORS_CPP_INCLUDE_DIR}>
      )
    # TODO this is not good, every package should install its own targets
    # cf. https://discourse.cmake.org/t/propagation-of-fetchcontent-targets-when-installing/2559/2
    install(TARGETS safetensors_cpp EXPORT dalotia
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin
      INCLUDES DESTINATION include
    )
  else()
    # to pass safetensors-cpp_DIR on the command line:
    find_package(safetensors-cpp REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)
  endif()
endif (DALOTIA_WITH_SAFETENSORS_CPP)

# tensorflow
if (DALOTIA_WITH_TENSORFLOW)  
  if (NOT DEFINED tensorflow_DIR OR tensorflow_DIR MATCHES "fetch")
    include(FetchContent)
    FetchContent_Declare(
      tensorflow
      URL https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-gpu-linux-x86_64.tar.gz
    )
    FetchContent_MakeAvailable(tensorflow)

    FetchContent_GetProperties(tensorflow SOURCE_DIR tensorflow_SRC_DIR)
    set(tensorflow_INCLUDE_DIRS ${tensorflow_SRC_DIR}/include)
    find_library(tensorflow_LIBRARIES
      NAMES tensorflow
      PATHS ${tensorflow_SRC_DIR}/lib
      NO_DEFAULT_PATH
    )
    message(STATUS "tensorflow_LIBRARIES: ${tensorflow_LIBRARIES}")

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(tensorflow DEFAULT_MSG tensorflow_INCLUDE_DIRS tensorflow_LIBRARIES)
  else()
    find_package(tensorflow REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)
  endif()
  if(NOT TARGET tensorflow::tensorflow)
    add_library(tensorflow::tensorflow SHARED IMPORTED)
    set_target_properties(
      tensorflow::tensorflow 
      PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${tensorflow_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${tensorflow_INCLUDE_DIRS}"
    )
    # cf. https://stackoverflow.com/a/41179630/7272382
    install(FILES ${tensorflow_INCLUDE_DIRS}/tensorflow/c/c_api.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorflow/c
    )
    install(FILES ${tensorflow_LIBRARIES}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  endif()
endif (DALOTIA_WITH_TENSORFLOW)

add_subdirectory(src) # target dalotia_cpp is generated here
if (DALOTIA_WITH_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(dalotia_cpp PRIVATE OpenMP::OpenMP_CXX)
  else ()
    message(WARNING "OpenMP not found")
  endif()
endif(DALOTIA_WITH_OPENMP)

if (DALOTIA_CPP_BUILD_EXAMPLES)
  add_executable(dalotia-example example.cpp)
  target_link_libraries(dalotia-example dalotia_cpp)
  if (DALOTIA_WITH_SAFETENSORS_CPP)  
    target_compile_options(dalotia-example PUBLIC "-DDALOTIA_WITH_SAFETENSORS_CPP")
  endif (DALOTIA_WITH_SAFETENSORS_CPP)
endif (DALOTIA_CPP_BUILD_EXAMPLES)

if (DALOTIA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif (DALOTIA_BUILD_TESTS)

# install version info etc.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "dalotia-config-version.cmake"
  VERSION 0.0.1
  COMPATIBILITY SameMajorVersion
)
install(FILES $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/dalotia-config-version.cmake
  DESTINATION lib/cmake/dalotia
)

# export full library interface
install(EXPORT dalotia
  FILE dalotia-config.cmake
  NAMESPACE dalotia::
  DESTINATION lib/cmake/dalotia
)
