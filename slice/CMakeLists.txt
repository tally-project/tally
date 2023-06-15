
add_executable(elementwise elementwise.cu)

set_target_properties(elementwise PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(elementwise
    cudart
    cuda
)

add_executable(matmul matmul.cu)

set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)