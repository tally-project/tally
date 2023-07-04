
add_executable(elementwise elementwise.cu)

set_target_properties(elementwise PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(matmul matmul.cu)

set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(max_pool max_pool.cu)

set_target_properties(max_pool PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)