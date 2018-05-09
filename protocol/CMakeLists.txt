cmake_minimum_required(VERSION 3.0.2)
project(rgbdcam-object-detect-protocol)

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR})
set(PROTO_HEADER_DIR ${PROJECT_SOURCE_DIR})
set(PROTO_PY_DIR ${PROJECT_SOURCE_DIR})
set(MSG_PROTOCOL_LIB protolib)

message("project source dir : ${PROJECT_SOURCE_DIR}")
message("proto src dir : ${PROTO_SRC_DIR}")
message("proto header dir : ${PROTO_HEADER_DIR}")
message("proto python dir : ${PROTO_PY_DIR}")
# protobuf_generate_cpp(${PROTO_SRC_DIR} ${PROTO_HEADER_DIR} message.proto)
# protobuf_generate_python(${PROTO_PY_DIR} message.proto) 
set(PROTO_MSG_CMD "bash")
set(PROTO_MSG_ARG "${PROJECT_SOURCE_DIR}/compile_message.sh")
set(PROTO_MSG_WORK_DIR "${PROJECT_SOURCE_DIR}")
message(STATUS "${PROTO_MSG_CMD} ${PROTO_MSG_ARG}")

execute_process(COMMAND ${PROTO_MSG_CMD} ${PROTO_MSG_ARG}
                WORKING_DIRECTORY ${PROTO_MSG_WORK_DIR}
                RESULT_VARIABLE PROTO_MSG_SEND_RES
                OUTPUT_VARIABLE PROTO_MSG_SEND_OUT)

file(GLOB PROTO_HEADER_FILES ${PROTO_HEADER_DIR}/message*.h)
file(GLOB PROTO_SRC_FILES ${PROTO_SRC_DIR}/message*.cc)

message(STATUS "proto src files : ${PROTO_SRC_FILES}")
add_library(${MSG_PROTOCOL_LIB} SHARED ${PROTO_HEADER_FILES} ${PROTO_SRC_FILES})
target_link_libraries(${MSG_PROTOCOL_LIB} ${PROTOBUF_LIBRARIES})