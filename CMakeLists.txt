cmake_minimum_required(VERSION 3.16)

project(DALOTIA_CPP CXX C)

option(DALOTIA_CPP_BUILD_EXAMPLES "Build examples" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MAKE_PROGRAM "make")

set(DALOTIA_CPP_SOURCES dalotia.cpp)
add_library(dalotia_cpp ${DALOTIA_CPP_SOURCES})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)


find_package(safetensors_cpp REQUIRED)
if(SAFETENSORS_CPP_FOUND)
  target_include_directories(dalotia_cpp PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR})
  target_link_libraries(dalotia_cpp ${SAFETENSORS_CPP_LIBRARY})
  target_compile_options(dalotia_cpp PUBLIC "-DDALOTIA_WITH_SAFETENSORS_CPP")
endif (SAFETENSORS_CPP_FOUND)

if (DALOTIA_CPP_BUILD_EXAMPLES)
  add_executable(example example.cpp)
  target_compile_definitions(example PRIVATE "DALOTIA_CPP_NO_IMPLEMENTATION")
  target_link_libraries(example dalotia_cpp)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model.safetensors DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

  if (DALOTIA_CPP_BUILD_C_API)
    add_executable(example-c example-c.c)
    target_compile_definitions(example-c PRIVATE "DALOTIA_C_NO_IMPLEMENTATION")
    target_link_libraries(example-c dalotia_c)
  endif ()
endif ()

enable_testing()
add_subdirectory(test)
