cmake_minimum_required(VERSION 3.10)
project(preload LANGUAGES C CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

llvm_map_components_to_libnames(llvm_libs support core irreader)

set (LIBIPC_PROJECT_DIR ${PROJECT_SOURCE_DIR}/third_party/cpp-ipc)

aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/sync LIBIPC_SRC_FILES)
aux_source_directory(${LIBIPC_PROJECT_DIR}/src/libipc/platform LIBIPC_SRC_FILES)

add_library(preload SHARED ${PROJECT_SOURCE_DIR}/src/preload.cpp ${LIBIPC_SRC_FILES})

target_include_directories(preload
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PRIVATE ${LIBIPC_PROJECT_DIR}/src
        $<$<BOOL:UNIX>:${LIBIPC_PROJECT_DIR}/src/libipc/platform/linux>)

target_link_libraries(preload PUBLIC
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>:pthread>
    $<$<NOT:$<OR:$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_SYSTEM_NAME},QNX>>>:rt>)

add_library(preload_cubin SHARED ${PROJECT_SOURCE_DIR}/src/preload-cubin.cpp)

target_include_directories(preload_cubin
    PUBLIC  /usr/local/cuda/include
)

target_link_libraries(preload_cubin ${llvm_libs})

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
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(server ${PROJECT_SOURCE_DIR}/server.cu)

set_target_properties(server PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(server
    PUBLIC  ${LIBIPC_PROJECT_DIR}/include
)

target_link_libraries(server PUBLIC
    ipc
    dl
)

add_subdirectory(slice)