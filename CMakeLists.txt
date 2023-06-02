cmake_minimum_required(VERSION 3.10)
project(preload LANGUAGES C CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_STANDARD 17)

set (LIBIPC_PROJECT_DIR ${PROJECT_SOURCE_DIR}/third_party/cpp-ipc)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/sync LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/platform LIBIPC_SRC_FILES)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES} ${LIBIPC_SRC_FILES})

target_include_directories(${PROJECT_NAME}
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>)

target_link_libraries(${PROJECT_NAME} PUBLIC
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>:pthread>
    $<$<NOT:$<OR:$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>>:rt>)

add_library(ipc SHARED ${LIBIPC_SRC_FILES})

target_include_directories(ipc
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>
)

target_link_libraries(ipc PUBLIC
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>:pthread>
    $<$<NOT:$<OR:$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>>:rt>
)

add_executable(client ${PROJECT_SOURCE_DIR}/client.cu)

set_target_properties(client PROPERTIES
    CUDA_ARCHITECTURES "86"
)

add_executable(server ${PROJECT_SOURCE_DIR}/server.cu)

set_target_properties(server PROPERTIES
    CUDA_ARCHITECTURES "86"
)

target_include_directories(server
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
)

target_link_libraries(server PUBLIC
    ipc
)