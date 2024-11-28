set(DALOTIA_CPP_SOURCES dalotia_assignment.cpp dalotia_formats.cpp dalotia_safetensors_file.cpp dalotia.cpp)
# set(DALOTIA_CPP_HEADERS dalotia_assignment.hpp dalotia_formats.hpp dalotia_formats.h dalotia_safetensors_file.hpp dalotia.hpp dalotia.h dalotia_tensor_file.hpp)
#TODO if DALOTIA_WITH_SAFETENSORS_CPP

add_library(dalotia_cpp ${DALOTIA_CPP_SOURCES})

if (DALOTIA_WITH_FORTRAN)
    set(DALOTIA_FORTRAN_HEADERS dalotia.f03)
    add_library(dalotia_fortran ${DALOTIA_FORTRAN_HEADERS})
    set_target_properties(dalotia_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(dalotia_fortran INTERFACE dalotia_cpp)
endif (DALOTIA_WITH_FORTRAN)
target_sources(dalotia_cpp PUBLIC dalotia.hpp dalotia.h)
target_include_directories(dalotia_cpp PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
else()
    message( FATAL_ERROR "Unknown compiler, please add matching compiler flags here." )
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
