# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
cmake_minimum_required(VERSION 2.8.1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})


#set(SOURCES PangoVis.cpp)
#set(HEADERS main.h PangoVis.h)

add_executable(ShowPointCloud main.cpp ${HEADERS} ${SOURCES})
target_link_libraries(ShowPointCloud ${Pangolin_LIBRARIES})


