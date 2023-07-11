
add_executable(elementwise elementwise.cu)

set_target_properties(elementwise PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(elementwise
    cuda
    cudart
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

add_executable(cublas_test cublas.cu)

target_link_libraries(cublas_test
    cuda
    cudart
    cublas
)

set_target_properties(cublas_test PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cublasLt_test cublasLt.cu)

target_link_libraries(cublasLt_test
    cuda
    cudart
    cublas
    cublasLt
)

set_target_properties(cublasLt_test PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)