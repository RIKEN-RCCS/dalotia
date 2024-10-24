set(DALOTIA_CPP_SOURCES dalotia_assignment.cpp dalotia_formats.cpp safetensors_file.cpp dalotia.cpp)
set(DALOTIA_CPP_HEADERS dalotia_assignment.hpp dalotia_formats.hpp dalotia_formats.h safetensors_file.hpp dalotia.hpp dalotia.h tensor_file.hpp)

add_library(dalotia_cpp ${DALOTIA_CPP_SOURCES} ${DALOTIA_CPP_HEADERS})
target_sources(dalotia_cpp PUBLIC )
target_include_directories(dalotia_cpp PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
