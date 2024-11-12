set(DALOTIA_CPP_SOURCES dalotia_assignment.cpp dalotia_formats.cpp safetensors_file.cpp dalotia.cpp)
# set(DALOTIA_CPP_HEADERS dalotia_assignment.hpp dalotia_formats.hpp dalotia_formats.h safetensors_file.hpp dalotia.hpp dalotia.h tensor_file.hpp)
#TODO if DALOTIA_WITH_SAFETENSORS_CPP

add_library(dalotia_cpp ${DALOTIA_CPP_SOURCES})

if (DALOTIA_WITH_FORTRAN)
    set(DALOTIA_FORTRAN_HEADERS dalotia.f03)
    add_library(dalotia_fortran ${DALOTIA_FORTRAN_HEADERS})
    set_target_properties(dalotia_fortran PROPERTIES LINKER_LANGUAGE Fortran)
endif (DALOTIA_WITH_FORTRAN)
target_sources(dalotia_cpp PUBLIC dalotia.hpp dalotia.h)
target_include_directories(dalotia_cpp PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
