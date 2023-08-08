
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
message(CUDA_ARCH_LIST="${CUDA_ARCH_LIST}")

add_executable(elementwise elementwise.cu)

set_target_properties(elementwise PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(elementwise
    cuda
    cudart
)

add_executable(matmul matmul.cu)

set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(max_pool max_pool.cu)

set_target_properties(max_pool PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cublas_test cublas.cu)

target_link_libraries(cublas_test
    cuda
    cudart
    cublas
)

set_target_properties(cublas_test PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
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
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cudnn_test cudnn_test.cu)

target_link_libraries(cudnn_test
    cuda
    cudart
    cudnn
)

set_target_properties(cudnn_test PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(test-sync test-sync.cu)

target_link_libraries(test-sync
    cuda
    cudart
    cudnn
)

set_target_properties(test-sync PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(test-jit test-jit.cpp)

target_link_libraries(test-jit
    cuda
    cudart
)

target_include_directories(test-jit
    PUBLIC  /usr/local/cuda/include
)

add_executable(basic_gemm basic_gemm.cu)

target_link_libraries(basic_gemm
    cuda
    cudart
)

set_target_properties(basic_gemm PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(basic_gemm
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
)

add_executable(turing_tensorop_gemm turing_tensorop_gemm.cu)

target_link_libraries(turing_tensorop_gemm
    cuda
    cudart
)

set_target_properties(turing_tensorop_gemm PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(turing_tensorop_gemm
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
)