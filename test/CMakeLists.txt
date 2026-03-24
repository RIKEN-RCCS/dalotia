# cf. https://stackoverflow.com/questions/52730994/how-to-pass-arguments-to-memcheck-with-ctest
set(MEMORYCHECK_COMMAND_OPTIONS
    "--gen-suppressions=all --suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tensorflow.supp --leak-check=full"
)
include (CTest)

if(DALOTIA_WITH_SAFETENSORS_CPP)
    add_executable( test_safetensors test_safetensors.cpp )
    target_link_libraries( test_safetensors dalotia_cpp )
    target_include_directories( test_safetensors PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR} ${safetensors-cpp_DIR})
    add_test( safetensors-file test_safetensors )

    add_executable( test_load_c test_load.c )
    target_link_libraries( test_load_c dalotia_cpp )
    target_include_directories( test_load_c PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR} ${safetensors-cpp_DIR})
    add_test( load-file-c test_load_c )

    add_executable( test_mnist test_mnist.cpp )
    target_link_libraries( test_mnist dalotia_cpp )
    target_include_directories( test_mnist PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR} ${safetensors-cpp_DIR})
    add_test( mnist_load test_mnist )

    if (DALOTIA_WITH_FORTRAN)
        add_executable( test_mnist_fortran test_mnist.f90 )
        set_target_properties(test_mnist_fortran PROPERTIES LINKER_LANGUAGE Fortran)
        target_include_directories(test_mnist_fortran PUBLIC $<TARGET_PROPERTY:dalotia_fortran,Fortran_MODULE_DIRECTORY>)
        target_link_libraries( test_mnist_fortran dalotia_fortran )
        add_test( mnist_load_fortran test_mnist_fortran )
    endif (DALOTIA_WITH_FORTRAN)
endif (DALOTIA_WITH_SAFETENSORS_CPP)

# Compile-only test: verify headers don't require C++20
add_executable(test_cxx17_compat test_cxx17_compat.cpp)
target_link_libraries(test_cxx17_compat dalotia_cpp)
set_target_properties(test_cxx17_compat PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME cxx17-compat COMMAND test_cxx17_compat)

if (DALOTIA_WITH_CUDA)
    add_executable( test_cufile test_cufile.cpp )
    target_link_libraries( test_cufile dalotia_cpp CUDA::cudart )
    if (DALOTIA_WITH_CUFILE)
        target_link_libraries( test_cufile CUDA::cuda_driver )
    endif()
    add_test( cufile-pointer-detection test_cufile )

    if (DALOTIA_WITH_CPP_PMR)
        add_executable( test_cuda_memory_resource test_cuda_memory_resource.cpp )
        target_link_libraries( test_cuda_memory_resource dalotia_cpp CUDA::cudart )
        add_test( cuda-memory-resource test_cuda_memory_resource )
    endif (DALOTIA_WITH_CPP_PMR)
endif (DALOTIA_WITH_CUDA)

if (DALOTIA_WITH_TENSORFLOW)
    add_executable( test_tensorflow test_tensorflow.cpp )
    target_link_libraries( test_tensorflow dalotia_cpp tensorflow::tensorflow )
    target_include_directories( test_tensorflow PUBLIC ${tensorflow_INCLUDE_DIRS})
    add_test( tensorflow-file test_tensorflow )
endif (DALOTIA_WITH_TENSORFLOW)
