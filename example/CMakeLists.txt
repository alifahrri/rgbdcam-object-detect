cmake_minimum_required(VERSION 2.8)
project(rgbd-object-detect-example)

message(STATUS "msg_protocol_lib : ${MSG_PROTOCOL_LIB}")
message(STATUS "protocol header dir : ${PROTO_HEADER_DIR}")
message(STATUS "protocol src dir : ${PROTO_SRC_DIR}")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Ofast -pthread")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Network REQUIRED)

include_directories(${PROTO_HEADER_DIR})
include_directories(cpp/include)
include_directories(${QT_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_lib cpp/src/client.cpp cpp/include/client.h)
target_link_libraries(${PROJECT_NAME}_lib Qt5::Network Qt5::Core ${MSG_PROTOCOL_LIB})
add_executable(example_client cpp/example_client.cpp)
target_link_libraries(example_client ${PROJECT_NAME}_lib)