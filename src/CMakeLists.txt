add_library(dalotia_cpp dalotia.cpp) # Daniel Pfeifer says: no variables
target_sources(dalotia_cpp PRIVATE dalotia_assignment.cpp dalotia_formats.cpp dalotia_safetensors_file.cpp )

# not sure if this is elegant, but helps to make this compatible to all languages
target_sources(dalotia_cpp PUBLIC
  FILE_SET dalotia_headers
  TYPE HEADERS
  FILES 
    dalotia.h
    dalotia_formats.h
    dalotia.hpp 
    dalotia_assignment.hpp
    dalotia_formats.hpp
    dalotia_safetensors_file.hpp
    dalotia_tensor_file.hpp
)
target_include_directories(dalotia_cpp PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

install(TARGETS dalotia_cpp EXPORT dalotia
  FILE_SET dalotia_headers
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

if (DALOTIA_WITH_CPP_PMR)
    # pass DALOTIA_WITH_CPP_PMR to target
    target_compile_definitions(dalotia_cpp PUBLIC "-DDALOTIA_WITH_CPP_PMR")
endif (DALOTIA_WITH_CPP_PMR)

if (DALOTIA_WITH_SAFETENSORS_CPP)
    target_link_libraries(dalotia_cpp PUBLIC safetensors_cpp)
    target_compile_options(dalotia_cpp PUBLIC "-DDALOTIA_WITH_SAFETENSORS_CPP")
endif (DALOTIA_WITH_SAFETENSORS_CPP)

if (DALOTIA_WITH_FORTRAN)
  add_library(dalotia_fortran dalotia.f03)
  set_target_properties(dalotia_fortran PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(dalotia_fortran INTERFACE dalotia_cpp)
  target_include_directories(dalotia_cpp PUBLIC
          $<INSTALL_INTERFACE:include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

  install(TARGETS dalotia_fortran EXPORT dalotia
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
  )
  install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/dalotia_c_interface.mod DESTINATION include)
endif (DALOTIA_WITH_FORTRAN)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(dalotia_cpp PRIVATE -Wall -Wextra)
else()
    message( FATAL_ERROR "Unknown compiler, please add matching compiler flags here." )
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

# have one dalotia library target that can be used in C++ and Fortran
add_library(dalotia INTERFACE)
add_library(dalotia::dalotia ALIAS dalotia)
# cf. https://github.com/3fd768cb-bbfe-4673-82ed-b934eda1b1d4

# export full library interface
install(EXPORT dalotia
  FILE dalotia.cmake
  NAMESPACE dalotia::
  DESTINATION lib/cmake/dalotia
)
