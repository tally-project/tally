cmake_minimum_required(VERSION 3.10)
project(preload)

set(CMAKE_CXX_STANDARD 17)

set (LIBIPC_PROJECT_DIR ${PROJECT_SOURCE_DIR}/third_party/cpp-ipc)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/sync SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/platform SRC_FILES)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_include_directories(${PROJECT_NAME}
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>)

target_link_libraries(${PROJECT_NAME} PUBLIC
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>:pthread>
    $<$<NOT:$<OR:$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>>:rt>)