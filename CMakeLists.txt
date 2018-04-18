cmake_minimum_required(VERSION 2.8)
project(recorder)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -O")

set(DARKNET_DIR "${PROJECT_SOURCE_DIR}/darknet")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++0x")
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no c++11 support")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -Ofast -g")
endif()

FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
	message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
	message(STATUS "CUDA Libraries: " ${CUDA_LIBRARIES})
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
	list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35; -Xcompiler; -fPIC;")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	add_definitions(-DGPU)
	cuda_include_directories(${DARKNET_DIR}/src)
else()
	list(APPEND LIBRARIES "m")
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

add_definitions(-DOPENCV)

include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

include_directories(${DARKNET_DIR}/src)
include_directories(${DARKNET_DIR}/include)
include_directories(${DARKNET_DIR}/examples)
include_directories(${DARKNET_DIR}/gui)

set(YOLO_SRC_DIR ${DARKNET_DIR}/src)
set(YOLO_EXAMPLES_DIR ${DARKNET_DIR}/examples)

FILE(GLOB YOLO_SRC_FILES 
${YOLO_SRC_DIR}/*.c
${YOLO_EXAMPLES_DIR}/yolo.c
${YOLO_EXAMPLES_DIR}/super.c
${YOLO_EXAMPLES_DIR}/lsd.c
${YOLO_EXAMPLES_DIR}/detector.c
${YOLO_EXAMPLES_DIR}/cifar.c
${YOLO_EXAMPLES_DIR}/go.c
${YOLO_EXAMPLES_DIR}/coco.c
${YOLO_EXAMPLES_DIR}/char_rnn.c
${YOLO_EXAMPLES_DIR}/classifier.c
${YOLO_EXAMPLES_DIR}/rnn.c
${YOLO_EXAMPLES_DIR}/attention.c
${YOLO_EXAMPLES_DIR}/regressor.c
${YOLO_EXAMPLES_DIR}/segmenter.c
${YOLO_EXAMPLES_DIR}/art.c
${YOLO_EXAMPLES_DIR}/tag.c
${YOLO_EXAMPLES_DIR}/captcha.c
${YOLO_EXAMPLES_DIR}/nightmare.c
)

if(CUDA_FOUND)
	message(STATUS "CUDA FOUND")
	FILE(GLOB YOLO_CU_SRC_FILES ${YOLO_SRC_DIR}/*.cu)
	cuda_include_directories(${YOLO_SRC_DIR})
	message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=[sm_50,compute_50]
        -gencode arch=compute_52,code=[sm_52,compute_52]
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
		)
	CUDA_COMPILE(cuda_objs ${YOLO_CU_SRC_FILES})
endif()

add_library(darknet_lib ${YOLO_SRC_FILES} ${YOLO_CU_SRC_FILES} ${cuda_objs})

find_package(OpenNI2 REQUIRED)
find_package(OpenCV REQUIRED)

message(WARNING "OpenNI Include Dir : " ${OPENNI2_INCLUDE_DIRS})
link_directories(${OPENNI2_INCLUDE_DIRS})

include_directories(include ${OPENNI2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

file(GLOB SRCFILES src/*.cpp)
add_executable(object-detect ${SRCFILES})

target_link_libraries(object-detect ${OpenCV_LIBS} ${OPENNI2_LIBRARY})
