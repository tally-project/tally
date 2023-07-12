cmake_minimum_required(VERSION 3.18)

Include(FetchContent)

# Find the cuda compiler
find_package(CUDAToolkit)

# Fetch and build catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(Catch2)

# Find cudnn
# include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/cuDNN.cmake)

add_executable(
    samples
    conv_sample.cpp
    resnet_test_list.cpp
    resnet_sample.cpp
    test_list.cpp 
    fp16_emu.cpp 
    helpers.cpp 
    fusion_sample.cpp 
    fp8_sample.cpp 
    mha_sample.cpp
    norm_samples.cpp
    fused_mha_sample.cpp
    f16_flash_mha_sample.cpp
    fp8_flash_mha_sample.cpp
)

if(DEFINED ENV{NO_DEFAULT_IN_SWITCH})
    message("Default case in the switch is disabled")
    add_compile_definitions(NO_DEFAULT_IN_SWITCH)
endif()

target_compile_options(
    samples PRIVATE
    -Wall
    -Wextra
    -Werror
    -Wno-unused-function
)

target_include_directories(samples
    PRIVATE "resnet_block/include"
    PUBLIC /usr/local/cuda/cudnn-frontend/include
    PUBLIC /usr/local/cuda/include
)

target_link_libraries(samples PUBLIC
    Catch2::Catch2WithMain
    cudart
    cudnn
)
