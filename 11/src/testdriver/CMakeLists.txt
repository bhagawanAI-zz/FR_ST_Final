set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -DNIST_EXTERN_FRVT_STRUCTS_VERSION -DNIST_EXTERN_API_VERSION")
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include)

# Configure to put executable in top level bin directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Get library implementation name
set (FRVT_IMPL_LIB $ENV{FRVT_IMPL_LIB})

# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../otherConfig/python3_9_13/lib/ ${CMAKE_CURRENT_SOURCE_DIR}/../../config/python3_9_13/lib/)

# Build executable link to dependent libraries
add_executable (validate11 ../../../common/src/util/util.cpp validate11.cpp)
target_link_libraries (validate11 ${FRVT_IMPL_LIB})