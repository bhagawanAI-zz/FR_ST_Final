project(firstImpl)
cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
include_directories (${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include ${CMAKE_CURRENT_SOURCE_DIR}/../../config/python3_9_13/include/python3.9)
# include_directories (${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include /home/ubuntu/testing2/python_june_24/include/python3.9/)

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)


#############################################
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../config/python3_9_13/lib/)
# link_directories(/home/ubuntu/testing2/python_june_24/lib/)

#############################################


set(FINAL_SHARED_LIBRARY_NAME "frvt_11_vida_002")
# Build the shared libraries
add_library (${FINAL_SHARED_LIBRARY_NAME} SHARED firstimplfrvt11.cpp)
target_link_libraries (${FINAL_SHARED_LIBRARY_NAME} python3.9)