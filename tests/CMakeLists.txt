include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
message(INSTALLED_GPU_CCS_1="${INSTALLED_GPU_CCS_1}")
string(REPLACE " " "" INSTALLED_GPU_CCS_1 ${INSTALLED_GPU_CCS_1})
string(REPLACE "." "" INSTALLED_GPU_CCS_1 ${INSTALLED_GPU_CCS_1})
string(SUBSTRING ${INSTALLED_GPU_CCS_1} 0 2 CUDA_ARCH_LIST)
message(CUDA_ARCH_LIST="${CUDA_ARCH_LIST}")

add_executable(elementwise cuda_samples_basic/elementwise.cu)

set_target_properties(elementwise PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(elementwise
    cuda
    cudart
)

add_executable(elementwise_no_ptx cuda_samples_basic/elementwise_no_ptx.cu)

set_target_properties(elementwise_no_ptx PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}-real
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(elementwise_no_ptx
    cuda
    cudart
)

add_executable(elementwise_with_cond cuda_samples_basic/elementwise_with_cond.cu)

set_target_properties(elementwise_with_cond PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(elementwise_with_cond
    cuda
    cudart
)

add_executable(add_driver_api cuda_samples_basic/add_driver_api.cu)

set_target_properties(add_driver_api PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(add_driver_api
    cuda
)

add_executable(matmul cuda_samples_basic/matmul.cu)

set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(max_pool cuda_samples_basic/max_pool.cu)

set_target_properties(max_pool PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cublas_test cuda_samples_basic/cublas.cu)

target_link_libraries(cublas_test
    cuda
    cudart
    cublas
)

set_target_properties(cublas_test PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cublasLt_test cuda_samples_basic/cublasLt.cu)

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

add_executable(cudnn_test cuda_samples_basic/cudnn_test.cu)

target_link_libraries(cudnn_test
    cuda
    cudart
    cudnn
)

set_target_properties(cudnn_test PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(test-sync cuda_samples_basic/test-sync.cu)

target_link_libraries(test-sync
    cuda
    cudart
    cudnn
)

set_target_properties(test-sync PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cublas_to_cutlass cuda_samples_basic/cublas_to_cutlass.cu ${PROJECT_SOURCE_DIR}/src/tally/cutlass/cutlass.cu)

target_link_libraries(cublas_to_cutlass
    cuda
    cudart
    cublas
    cublasLt
)

set_target_properties(cublas_to_cutlass PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(cublas_to_cutlass
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
)

add_executable(matmul_fp16 cuda_samples_basic/matmul_fp16.cu)

target_link_libraries(matmul_fp16
    cuda
    cudart
    cublas
)

set_target_properties(matmul_fp16 PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(matmul_fp16
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
)

add_executable(batched_gemm cuda_samples_basic/batched_gemm.cu)

target_link_libraries(batched_gemm
    cuda
    cudart
    cublas
)

set_target_properties(batched_gemm PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(batched_gemm
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
)

add_executable(batched_gemm_cutlass cuda_samples_basic/batched_gemm_cutlass.cu)

set_target_properties(batched_gemm_cutlass PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(batched_gemm_cutlass
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
)

add_executable(cutlass_bias_epilogue cuda_samples_basic/cutlass_bias_epilogue.cu)

target_link_libraries(cutlass_bias_epilogue
    cuda
    cudart
    cublas
    cublasLt
)

set_target_properties(cutlass_bias_epilogue PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(cutlass_bias_epilogue
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
)

add_executable(basic_gemm cuda_samples_basic/basic_gemm.cu)

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

add_executable(turing_tensorop_gemm cuda_samples_basic/turing_tensorop_gemm.cu)

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

add_executable(volta_tensorop_gemm cuda_samples_basic/volta_tensorop_gemm.cu)

target_link_libraries(volta_tensorop_gemm
    cuda
    cudart
)

set_target_properties(volta_tensorop_gemm PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(volta_tensorop_gemm
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
)

add_executable(ampere_tf32_tensorop_gemm cuda_samples_basic/ampere_tf32_tensorop_gemm.cu)

target_link_libraries(ampere_tf32_tensorop_gemm
    cuda
    cudart
)

set_target_properties(ampere_tf32_tensorop_gemm PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(ampere_tf32_tensorop_gemm
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    PUBLIC  ${PROJECT_SOURCE_DIR}/third_party/cutlass/tools/util/include/
    PUBLIC  ${PROJECT_SOURCE_DIR}/tests
)

add_executable(cuda-memcpy-test cuda_samples_basic/cuda-memcpy-test.cu)

target_link_libraries(cuda-memcpy-test
    cuda
    cudart
)

set_target_properties(cuda-memcpy-test PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(test_nccl cuda_samples_basic/test_nccl.cu)

target_include_directories(test_nccl
    PUBLIC ${PROJECT_SOURCE_DIR}/third_party/nccl/build/include
)

target_link_libraries(test_nccl
    cuda
    cudart
    ${PROJECT_SOURCE_DIR}/third_party/nccl/build/lib/libnccl.so
)

set_target_properties(test_nccl PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(cudnnReduceTensor cuda_samples_basic/cudnnReduceTensor.cu)

target_link_libraries(cudnnReduceTensor
    cuda
    cudart
    cudnn
)

set_target_properties(cudnnReduceTensor PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}
    CUDA_RUNTIME_LIBRARY Shared
)