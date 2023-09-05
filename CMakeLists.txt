cmake_minimum_required(VERSION 3.24)
project(tally LANGUAGES C CXX CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-comment -Wno-deprecated -Wno-maybe-uninitialized")

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    # set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()

option(ENABLE_LOGGING "Enable logging" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)
option(RUN_LOCALLY "Run client process locally" OFF)

link_directories(/usr/local/cuda/lib64)

FIND_PACKAGE(Boost COMPONENTS serialization regex REQUIRED)

add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/iceoryx/iceoryx_meta)
add_subdirectory(third_party/json)
add_subdirectory(third_party/folly)
add_subdirectory(third_party/readerwriterqueue)

include(GNUInstallDirs)

find_package(iceoryx_platform REQUIRED)
find_package(iceoryx_posh REQUIRED)
find_package(iceoryx_hoofs REQUIRED)

include(IceoryxPackageHelper)
include(IceoryxPlatform)
include(IceoryxPlatformSettings)

set(TALLY_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/tally/cache.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/cuda_util.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/util.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/signal.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/env.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/transform.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/generated/cuda_api.cpp
)

set(TALLY_SERVER_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/tally/measurements.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/server.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/generated/server.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/partial.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/cuda_launch.cpp
)

set(TALLY_CLIENT_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/tally/generated/client.cpp
)

add_library(tally STATIC ${TALLY_SRC_FILES})
target_include_directories(tally
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PUBLIC ${PROJECT_SOURCE_DIR}/third_party/json/single_include
)

target_link_libraries(tally PUBLIC
    iceoryx_posh::iceoryx_posh
    dl
)

add_library(tally_client SHARED
    ${PROJECT_SOURCE_DIR}/src/tally/preload/tally_client.cpp
    ${PROJECT_SOURCE_DIR}/src/tally/preload/generated/tally_client.cpp
    ${TALLY_SRC_FILES}
    ${TALLY_CLIENT_SRC_FILES}
)

if(ENABLE_LOGGING)
    target_compile_definitions(tally_client PRIVATE ENABLE_LOGGING=${ENABLE_LOGGING})
endif()

if(ENABLE_PROFILING)
    target_compile_definitions(tally_client PRIVATE ENABLE_PROFILING=${ENABLE_PROFILING})
endif()

if(RUN_LOCALLY)
    target_compile_definitions(tally_client PRIVATE RUN_LOCALLY=${RUN_LOCALLY})
endif()

target_include_directories(tally_client
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC  /usr/local/cuda/include
    PUBLIC ${PROJECT_SOURCE_DIR}/third_party/json/single_include
)

target_link_libraries(tally_client PUBLIC
    PUBLIC Boost::serialization 
    PUBLIC Boost::regex
    iceoryx_posh::iceoryx_posh
)

target_compile_definitions(tally_client PRIVATE IS_CLIENT="TRUE")

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

add_executable(tally_server ${PROJECT_SOURCE_DIR}/src/tally/tally_server.cpp ${TALLY_SERVER_SRC_FILES})

if(ENABLE_LOGGING)
    target_compile_definitions(tally_server PRIVATE ENABLE_LOGGING=${ENABLE_LOGGING})
endif()

target_include_directories(tally_server
    PUBLIC  /usr/local/cuda/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/third_party/json/single_include
)

target_link_libraries(tally_server PUBLIC
    cuda
    cudart
    cublas
    dl
    tally
    cublasLt
    cudnn
    PUBLIC Boost::serialization
    PUBLIC Boost::regex
    spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>
    iceoryx_posh::iceoryx_posh
    folly
    readerwriterqueue
)

add_subdirectory(scripts)
add_subdirectory(tests)
add_subdirectory(third_party/cudnn-frontend)