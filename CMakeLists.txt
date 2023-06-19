cmake_minimum_required(VERSION 3.10)
project(preload_sched LANGUAGES C CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_STANDARD 17)

FIND_PACKAGE(Boost COMPONENTS serialization REQUIRED)

set (LIBIPC_PROJECT_DIR ${PROJECT_SOURCE_DIR}/third_party/cpp-ipc)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/tally TALLY_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/sync LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/platform LIBIPC_SRC_FILES)

add_library(preload_sched SHARED ${PROJECT_SOURCE_DIR}/src/preload_sched.cpp ${LIBIPC_SRC_FILES})

target_include_directories(preload_sched
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>)

target_link_libraries(preload_sched PUBLIC
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>:pthread>
    $<$<NOT:$<OR:$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>>:rt>)

add_library(preload_slice SHARED ${PROJECT_SOURCE_DIR}/src/preload_slice.cpp ${TALLY_SRC_FILES})

target_include_directories(preload_slice
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PUBLIC ${Boost_INCLUDE_DIR}
)

target_link_libraries(preload_slice
    PUBLIC Boost::serialization
)

add_library(preload_profile_cpu SHARED ${PROJECT_SOURCE_DIR}/src/preload_profile_cpu.cpp)

target_include_directories(preload_profile_cpu
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
)

add_library(preload_profile_gpu SHARED ${PROJECT_SOURCE_DIR}/src/preload_profile_gpu.cpp)

target_include_directories(preload_profile_gpu
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

add_executable(server ${PROJECT_SOURCE_DIR}/server.cu)

set_target_properties(server PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(server
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
)

target_link_libraries(server PUBLIC
    ipc
    dl
)

add_subdirectory(tests)