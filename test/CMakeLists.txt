
add_executable( test_safetensors test_safetensors.cpp )
target_link_libraries( test_safetensors dalotia_cpp )
add_test( safetensors-file test_safetensors )
