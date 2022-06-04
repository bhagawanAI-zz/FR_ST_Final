project(firstImpl)
cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
include_directories (${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include ${CMAKE_CURRENT_SOURCE_DIR}/../../config/python3_9_13/include/python3.9)

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)


#############################################
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../config/python3_9_13/lib/)


#############################################


# Build the shared libraries
add_library (frvt_11_first_001 SHARED firstimplfrvt11.cpp)

