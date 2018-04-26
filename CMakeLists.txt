cmake_minimum_required(VERSION 2.8)
project(recorder)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++11 -Ofast -pthread")

set(DARKNET_DIR ${PROJECT_SOURCE_DIR}/darknet)

message("Darknet Dir : " ${DARKNET_DIR})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

find_package(Threads)
find_package(OpenNI2 REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions(-DOPENCV)
# add_subdirectory(${DARKNET_DIR})

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
	# add_definitions(-DGPU)
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

include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

include_directories(${DARKNET_DIR}/src)
include_directories(${DARKNET_DIR}/include)
include_directories(${DARKNET_DIR}/examples)
include_directories(${DARKNET_DIR}/gui)
include_directories(wrapper/include)
include_directories(include ${OPENNI2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(YOLO_SRC_DIR ${DARKNET_DIR}/src)
set(YOLO_EXAMPLES_DIR ${DARKNET_DIR}/examples)

FILE(GLOB YOLO_SRC_FILES 
${YOLO_SRC_DIR}/activation_layer.c 
${YOLO_SRC_DIR}/activations.c 
${YOLO_SRC_DIR}/avgpool_layer.c 
${YOLO_SRC_DIR}/bathcnorm_layer.c 
${YOLO_SRC_DIR}/blas.c 
${YOLO_SRC_DIR}/box.c 
${YOLO_SRC_DIR}/col2im.c 
${YOLO_SRC_DIR}/compare.c 
${YOLO_SRC_DIR}/connected_layer.c 
${YOLO_SRC_DIR}/convolutional_layer.c 
${YOLO_SRC_DIR}/cost_layer.c 
${YOLO_SRC_DIR}/crnn_layer.c 
${YOLO_SRC_DIR}/crop_layer.c 
${YOLO_SRC_DIR}/cuda.c 
${YOLO_SRC_DIR}/data.c 
${YOLO_SRC_DIR}/deconvolutional_layer.c 
# ${YOLO_SRC_DIR}/demo.c 
${YOLO_SRC_DIR}/dropout_layer.c
${YOLO_SRC_DIR}/gemm.c 
${YOLO_SRC_DIR}/gru_layer.c 
${YOLO_SRC_DIR}/im2col.c 
${YOLO_SRC_DIR}/image.c 
${YOLO_SRC_DIR}/layer.c 
${YOLO_SRC_DIR}/list.c 
${YOLO_SRC_DIR}/local_layer.c 
${YOLO_SRC_DIR}/lstm_layer.c 
${YOLO_SRC_DIR}/matrix.c 
${YOLO_SRC_DIR}/maxpool_layer.c 
${YOLO_SRC_DIR}/network.c 
${YOLO_SRC_DIR}/normalization_layer.c 
${YOLO_SRC_DIR}/option_list.c 
${YOLO_SRC_DIR}/parser.c 
${YOLO_SRC_DIR}/region_layer.c 
${YOLO_SRC_DIR}/reorg_layer.c 
${YOLO_SRC_DIR}/rnn_layer.c 
${YOLO_SRC_DIR}/route_layer.c 
${YOLO_SRC_DIR}/softmax_layer.c 
${YOLO_SRC_DIR}/tree.c 
${YOLO_SRC_DIR}/utils.c 
${YOLO_SRC_DIR}/darknet.c 
# ${YOLO_EXAMPLES_DIR}/yolo.c 
# ${YOLO_EXAMPLES_DIR}/super.c 
# ${YOLO_EXAMPLES_DIR}/lsd.c 
# ${YOLO_EXAMPLES_DIR}/detector.c 
# ${YOLO_EXAMPLES_DIR}/cifar.c 
# ${YOLO_EXAMPLES_DIR}/go.c 
# ${YOLO_EXAMPLES_DIR}/coco.c 
# ${YOLO_EXAMPLES_DIR}/char_rnn.c 
# # ${YOLO_EXAMPLES_DIR}/classifier.c 
# ${YOLO_EXAMPLES_DIR}/rnn.c 
# ${YOLO_EXAMPLES_DIR}/attention.c 
# # ${YOLO_EXAMPLES_DIR}/regressor.c 
# ${YOLO_EXAMPLES_DIR}/segmenter.c 
# ${YOLO_EXAMPLES_DIR}/art.c 
# ${YOLO_EXAMPLES_DIR}/tag.c 
# ${YOLO_EXAMPLES_DIR}/captcha.c 
# ${YOLO_EXAMPLES_DIR}/nightmare.c 
)

# if(CUDA_FOUND)
# 	message(STATUS "CUDA FOUND")
# 	FILE(GLOB YOLO_CU_SRC_FILES ${YOLO_SRC_DIR}/*.cu)
# 	cuda_include_directories(${YOLO_SRC_DIR})
# 	message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
#     message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
#     set(CUDA_NVCC_FLAGS
#         ${CUDA_NVCC_FLAGS};
#         -O3
#         -gencode arch=compute_30,code=sm_30
#         -gencode arch=compute_35,code=sm_35
#         -gencode arch=compute_50,code=[sm_50,compute_50]
#         -gencode arch=compute_52,code=[sm_52,compute_52]
#         -gencode arch=compute_61,code=sm_61
#         -gencode arch=compute_62,code=sm_62
# 		)
# 	cuda_include_directories(${DARKNET_DIR}/src)
# 	CUDA_COMPILE(cuda_objs ${YOLO_CU_SRC_FILES})
# endif()

message("Compile Definitions : " ${COMPILE_DEFINITIONS})

message("YOLO src files : " ${YOLO_SRC_FILES})
message("YOLO cu src files : " ${YOLO_CU_SRC_FILES})

# add_library(darknet_lib 
# ${YOLO_SRC_DIR}/activation_layer.c 
# ${YOLO_SRC_DIR}/activations.c 
# ${YOLO_SRC_DIR}/avgpool_layer.c 
# ${YOLO_SRC_DIR}/batchnorm_layer.c 
# ${YOLO_SRC_DIR}/blas.c 
# ${YOLO_SRC_DIR}/box.c 
# ${YOLO_SRC_DIR}/col2im.c 
# ${YOLO_SRC_DIR}/compare.c 
# ${YOLO_SRC_DIR}/connected_layer.c 
# ${YOLO_SRC_DIR}/convolutional_layer.c 
# ${YOLO_SRC_DIR}/cost_layer.c 
# ${YOLO_SRC_DIR}/crnn_layer.c 
# ${YOLO_SRC_DIR}/crop_layer.c 
# ${YOLO_SRC_DIR}/cuda.c 
# ${YOLO_SRC_DIR}/data.c 
# ${YOLO_SRC_DIR}/deconvolutional_layer.c 
# ${YOLO_SRC_DIR}/demo.c 
# ${YOLO_SRC_DIR}/dropout_layer.c
# ${YOLO_SRC_DIR}/gemm.c 
# ${YOLO_SRC_DIR}/gru_layer.c 
# ${YOLO_SRC_DIR}/im2col.c 
# ${YOLO_SRC_DIR}/image.c 
# ${YOLO_SRC_DIR}/layer.c 
# ${YOLO_SRC_DIR}/list.c 
# ${YOLO_SRC_DIR}/local_layer.c 
# ${YOLO_SRC_DIR}/lstm_layer.c 
# ${YOLO_SRC_DIR}/matrix.c 
# ${YOLO_SRC_DIR}/maxpool_layer.c 
# ${YOLO_SRC_DIR}/network.c 
# ${YOLO_SRC_DIR}/normalization_layer.c 
# ${YOLO_SRC_DIR}/option_list.c 
# ${YOLO_SRC_DIR}/parser.c 
# ${YOLO_SRC_DIR}/region_layer.c 
# ${YOLO_SRC_DIR}/reorg_layer.c 
# ${YOLO_SRC_DIR}/rnn_layer.c 
# ${YOLO_SRC_DIR}/route_layer.c 
# ${YOLO_SRC_DIR}/softmax_layer.c 
# ${YOLO_SRC_DIR}/tree.c 
# ${YOLO_SRC_DIR}/utils.c 
# ${YOLO_SRC_DIR}/darknet.c 
# ${YOLO_EXAMPLES_DIR}/yolo.c 
# ${YOLO_EXAMPLES_DIR}/super.c 
# ${YOLO_EXAMPLES_DIR}/lsd.c 
# ${YOLO_EXAMPLES_DIR}/detector.c 
# ${YOLO_EXAMPLES_DIR}/cifar.c 
# ${YOLO_EXAMPLES_DIR}/go.c 
# ${YOLO_EXAMPLES_DIR}/coco.c 
# ${YOLO_EXAMPLES_DIR}/classifier.c 
# ${YOLO_EXAMPLES_DIR}/rnn.c 
# ${YOLO_EXAMPLES_DIR}/attention.c 
# ${YOLO_EXAMPLES_DIR}/regressor.c 
# ${YOLO_EXAMPLES_DIR}/segmenter.c 
# ${YOLO_EXAMPLES_DIR}/art.c 
# ${YOLO_EXAMPLES_DIR}/tag.c 
# ${YOLO_EXAMPLES_DIR}/yolo.c 
# ${YOLO_EXAMPLES_DIR}/captcha.c 
# ${YOLO_EXAMPLES_DIR}/nightmare.c 
# ${YOLO_CU_SRC_FILES} 
# ${cuda_objs}
# )

add_library(darknet_lib ${YOLO_SRC_FILES} ${YOLO_CU_SRC_FILES} ${cuda_objs})
target_link_libraries(darknet_lib ${LIBRARIES} ${OpenCV_LIBS})

message(WARNING "OpenNI Include Dir : " ${OPENNI2_INCLUDE_DIRS})
link_directories(${OPENNI2_INCLUDE_DIRS})

# cuda_add_library(yolo_cuda_lib ${YOLO_CU_SRC_FILES})
add_library(darknet_cpp_wraplib wrapper/src/yolowrapper.cpp ${YOLO_CU_SRC_FILES})
target_link_libraries(darknet_cpp_wraplib )
# target_link_libraries(darknet_cpp_wraplib darknet_lib yolo_cuda_lib)

file(GLOB SRCFILES src/*.cpp)
add_executable(object-detect ${SRCFILES})

target_link_libraries(object-detect darknet_cpp_wraplib ${OpenCV_LIBS} ${OPENNI2_LIBRARY})
