add_library(dalotia_cpp dalotia.cpp) # Daniel Pfeifer says: no variables
target_sources(dalotia_cpp PRIVATE dalotia_assignment.cpp dalotia_formats.cpp dalotia_safetensors_file.cpp )
# have one dalotia library target that can be used in C++ and Fortran
add_library(dalotia INTERFACE)

if (DALOTIA_WITH_FORTRAN)
    add_library(dalotia_fortran dalotia.f03)
    set_target_properties(dalotia_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    target_link_libraries(dalotia_fortran INTERFACE dalotia_cpp)
    # target_link_libraries(dalotia INTERFACE dalotia_fortran)
endif (DALOTIA_WITH_FORTRAN)

if (DALOTIA_WITH_CPP_PMR)
    # pass DALOTIA_WITH_CPP_PMR to target
    target_compile_definitions(dalotia_cpp PUBLIC "-DDALOTIA_WITH_CPP_PMR")
endif (DALOTIA_WITH_CPP_PMR)

if (DALOTIA_WITH_SAFETENSORS_CPP)
    target_link_libraries(dalotia_cpp PUBLIC safetensors_cpp)
    target_compile_options(dalotia_cpp PUBLIC "-DDALOTIA_WITH_SAFETENSORS_CPP")
endif (DALOTIA_WITH_SAFETENSORS_CPP)

# not sure if this is elegant, but helps to make this compatible to all languages
target_sources(dalotia_cpp PRIVATE dalotia.hpp dalotia.h)
target_include_directories(dalotia_cpp PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
else()
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

install(TARGETS dalotia_cpp EXPORT dalotia
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)
install(TARGETS dalotia_fortran EXPORT dalotia
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)
