find_path(
  SAFETENSORS_CPP_INCLUDE_DIR
    NAMES "safetensors.hh"
    PATHS "${PROJECT_SOURCE_DIR}/safetensors-cpp/" "${PROJECT_SOURCE_DIR}/../safetensors-cpp/"
)

if( SAFETENSORS_CPP_INCLUDE_DIR )
  find_library(
    SAFETENSORS_CPP_LIBRARY
    NAMES safetensors_cpp
    PATHS "${PROJECT_SOURCE_DIR}/safetensors-cpp" "${PROJECT_SOURCE_DIR}/../safetensors-cpp" 
        "${PROJECT_SOURCE_DIR}/safetensors-cpp/build" "${PROJECT_SOURCE_DIR}/../safetensors-cpp/build"
  )
  if ( SAFETENSORS_CPP_LIBRARY )
    message(STATUS "FindSafetensors: found safetensors-cpp")
    set(SAFETENSORS_CPP_FOUND TRUE)
  else ( SAFETENSORS_CPP_LIBRARY )
    message(STATUS "FindSafetensors: Could not find safetensors-cpp library")
  endif ( SAFETENSORS_CPP_LIBRARY )
else(SAFETENSORS_CPP_INCLUDE_DIR)
  message(STATUS "FindSafetensors: Could not find safetensors.hh")
endif(SAFETENSORS_CPP_INCLUDE_DIR)