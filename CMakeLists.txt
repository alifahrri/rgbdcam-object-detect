cmake_minimum_required(VERSION 2.8)
project(recorder)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -O")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)


find_package(OpenNI2 REQUIRED)
find_package(OpenCV REQUIRED)

message(WARNING "OpenNI Include Dir : " ${OPENNI2_INCLUDE_DIRS})
link_directories(${OPENNI2_INCLUDE_DIRS})

include_directories(include ${OPENNI2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

file(GLOB SRCFILES src/*.cpp)
add_executable(object-detect ${SRCFILES})

target_link_libraries(object-detect ${OpenCV_LIBS} ${OPENNI2_LIBRARY})
