
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
endif (DALOTIA_WITH_SAFETENSORS_CPP)
