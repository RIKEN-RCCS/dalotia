find_path(
  SAFETENSORS_CPP_INCLUDE_DIR
    NAMES "safetensors.hh"
    PATHS "${PROJECT_SOURCE_DIR}/safetensors-cpp/" "${PROJECT_SOURCE_DIR}/../safetensors-cpp/"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/safetensors-cpp-src/" "${safetensors-cpp_DIR}"
)

if( SAFETENSORS_CPP_INCLUDE_DIR )
  set(SAFETENSORS_CPP_FOUND TRUE)
  find_library(
    SAFETENSORS_CPP_LIBRARY
    NAMES safetensors_cpp
    PATHS "${PROJECT_SOURCE_DIR}/safetensors-cpp" "${PROJECT_SOURCE_DIR}/../safetensors-cpp" 
        "${PROJECT_SOURCE_DIR}/safetensors-cpp/build" "${PROJECT_SOURCE_DIR}/../safetensors-cpp/build"
        "${safetensors-cpp_DIR}" "${FETCHCONTENT_BASE_DIR}/safetensors-cpp-build/"
  )
  if ( SAFETENSORS_CPP_LIBRARY )
    message(STATUS "FindSafetensors: found safetensors-cpp")
  else ( SAFETENSORS_CPP_LIBRARY )
    message(STATUS "FindSafetensors: Could not find safetensors-cpp library")
  endif ( SAFETENSORS_CPP_LIBRARY )
else(SAFETENSORS_CPP_INCLUDE_DIR)
  message(STATUS "FindSafetensors: Could not find safetensors.hh")
endif(SAFETENSORS_CPP_INCLUDE_DIR)