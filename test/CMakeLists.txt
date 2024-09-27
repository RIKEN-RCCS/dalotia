
if(SAFETENSORS_CPP_FOUND)
    add_executable( test_safetensors test_safetensors.cpp )
    target_link_libraries( test_safetensors dalotia_cpp )
    target_include_directories( test_safetensors PUBLIC ${SAFETENSORS_CPP_INCLUDE_DIR} ${safetensors-cpp_DIR})
    add_test( safetensors-file test_safetensors )
endif (SAFETENSORS_CPP_FOUND)
