# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
cmake_minimum_required(VERSION 2.8.1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


set(OpenCV_DIR /home/jacky/code/KintinuousExt/deps/opencv-2.4.9/build)
INCLUDE_DIRECTORIES(/home/jacky/code/KintinuousExt/deps/opencv-2.4.9/build/include)
set(OpenCV_INCLUDE_DIRS /home/jacky/code/KintinuousExt/deps/opencv-2.4.9/build/include)
set(OpenCV_LIBS /home/jacky/code/KintinuousExt/deps/opencv-2.4.9/build/lib)


find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL 1.2 REQUIRED)


include_directories(${Pangolin_INCLUDE_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


#set(SOURCES PangoVis.cpp)
#set(HEADERS main.h PangoVis.h)
set(HEADERS main.h PangoCloud.h)

add_executable(ShowPointCloud main.cpp ${HEADERS} ${SOURCES})
target_link_libraries(ShowPointCloud ${Pangolin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})


