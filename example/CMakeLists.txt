cmake_minimum_required(VERSION 2.8)
project(rgbd-object-detect-example)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Ofast -pthread")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Network REQUIRED)

include_directories(client)
include_directories(${QT_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_lib client/client.cpp)
target_link_libraries(${PROJECT_NAME}_lib Qt5::Network Qt5::Core)
add_executable(example_client example_client.cpp)
target_link_libraries(example_client ${PROJECT_NAME}_lib)