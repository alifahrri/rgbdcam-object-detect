cmake_minimum_required(VERSION 2.8)
project(rgbd-object-detect)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Ofast -pthread")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_definitions(-O4 -g -fstack-protector)
find_package(CUDA)
if(CUDA_FOUND)
	add_definitions(-DGPU)
endif()

find_package(Threads)
find_package(OpenNI2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5Network REQUIRED)

set(DARKNET_DIR "${CMAKE_SOURCE_DIR}/darknet/")
add_definitions(-DOPENCV)
add_definitions(-DDARKNET_DIR="${CMAKE_SOURCE_DIR}/darknet/")

message("darknet : ${DARKNET_DIR}")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no c++11 support")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast -g")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)

find_package(Boost REQUIRED python)
find_package(PythonLibs REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(darknet)
include_directories(darknet/include)
include_directories(darknet/src)

include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
include_directories(wrapper/include)
include_directories(include ${OPENNI2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${QT_INCLUDE_DIRS}
)

message(WARNING "OpenNI Include Dir : " ${OPENNI2_INCLUDE_DIRS})
link_directories(${OPENNI2_INCLUDE_DIRS})

file(GLOB WRAPPER_FILES wrapper/src/*.cpp wrapper/src/*.c)
add_library(darknet_cpp_wraplib ${WRAPPER_FILES})
target_link_libraries(darknet_cpp_wraplib darknetlib)

file(GLOB SRCFILES src/*.cpp)
add_executable(object-detect ${SRCFILES})

target_link_libraries(object-detect darknet_cpp_wraplib ${OpenCV_LIBS} ${OPENNI2_LIBRARY} Qt5::Network)

add_subdirectory(example)