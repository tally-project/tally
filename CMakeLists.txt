cmake_minimum_required(VERSION 3.10)
project(tally LANGUAGES C CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

FIND_PACKAGE(Boost COMPONENTS serialization regex REQUIRED)

set (LIBIPC_PROJECT_DIR ${PROJECT_SOURCE_DIR}/third_party/cpp-ipc)

set(TALLY_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/tally/daemon.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/cache.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/cuda_launch.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/cuda_util.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/util.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/signal.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/env.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/transform.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/generated/cuda_api.cpp
)

set(TALLY_SERVER_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/tally/server.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/generated/server.cpp
)

set(TALLY_CLIENT_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/tally/client.cpp
)

aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/sync LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/platform LIBIPC_SRC_FILES)

add_library(tally STATIC ${TALLY_SRC_FILES})
target_include_directories(tally
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>
)

add_library(tally_client SHARED
    ${PROJECT_SOURCE_DIR}/src/tally/preload/tally_client.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/preload/generated/tally_client.cpp
    ${LIBIPC_SRC_FILES}
    ${TALLY_SRC_FILES}
    ${TALLY_CLIENT_SRC_FILES}
)

target_include_directories(tally_client
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>)

target_link_libraries(tally_client PUBLIC
    PUBLIC Boost::serialization 
    PUBLIC Boost::regex
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>:pthread>
    $<$<NOT:$<OR:$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>>:rt>)

add_library(tally_transform SHARED ${PROJECT_SOURCE_DIR}/src/tally/preload/tally_transform.cpp ${TALLY_SRC_FILES})

target_include_directories(tally_transform
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(tally_transform
    PUBLIC Boost::serialization
    PUBLIC Boost::regex
)

add_library(tally_profile_cpu SHARED ${PROJECT_SOURCE_DIR}/src/tally/preload/tally_profile_cpu.cpp)

target_include_directories(tally_profile_cpu
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
)

add_library(tally_profile_gpu SHARED ${PROJECT_SOURCE_DIR}/src/tally/preload/tally_profile_gpu.cpp)

target_include_directories(tally_profile_gpu
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
)

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

if (LIBIPC_BUILD_DEMOS)
    add_subdirectory(${LIBIPC_PROJECT_DIR}/demo/chat)
    add_subdirectory(${LIBIPC_PROJECT_DIR}/demo/msg_que)
    add_subdirectory(${LIBIPC_PROJECT_DIR}/demo/send_recv)
endif()

add_executable(server ${PROJECT_SOURCE_DIR}/src/tally/server.cu ${TALLY_SERVER_SRC_FILES})

set_target_properties(server PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(server
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
)

target_link_libraries(server PUBLIC
    cuda
    ipc
    dl
    tally
    PUBLIC Boost::serialization
    PUBLIC Boost::regex
)

add_subdirectory(tests)