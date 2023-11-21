#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "cutlass/gemm/device/gemm.h"

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator>;

cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    cutlass::half_t const *A,
    int lda,
    cutlass::half_t const *B,
    int ldb,
    float beta,
    cutlass::half_t *C,
    int ldc,
    cutlass::half_t *D,
    int ldd
) {

    using RowMajor = cutlass::layout::ColumnMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    cutlass::Status status;

    using Gemm = cutlass::gemm::device::Gemm<cutlass::half_t,
                                         ColumnMajor,
                                         cutlass::half_t,
                                         ColumnMajor,
                                         cutlass::half_t,
                                         ColumnMajor,
                                         float>;
    Gemm gemm_operator;

    Gemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                         {A, lda},    // Tensor-ref for source matrix A
                         {B, ldb},    // Tensor-ref for source matrix B
                         {C, ldc},    // Tensor-ref for source matrix C
                         {D, ldd},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                         {alpha, beta}); // Scalars used in the Epilogue


    status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "): " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << "(" << __LINE__ << "): " \
                  << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    srand (1);
    
    // Define the size of the matrices
    int m = 256, n = 256, k = 256;

    // Allocate host memory for matrices A, B, and C
    half *h_A = new half[m * k];
    half *h_B = new half[k * n];
    half *h_C = new half[m * n];
    half *h_D_cublas = new half[m * n];
    half *h_D_cutlass = new half[m * n];

    // Initialize matrices A and B with some values (for example purposes)
    for (int i = 0; i < m * k; i++) {
        h_A[i] = __float2half(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
    }

    for (int i = 0; i < k * n; i++) {
        h_B[i] = __float2half(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
    }

    for (int i = 0; i < m * n; i++) {
        h_C[i] = __float2half(1.0f);
    }

    // Allocate device memory
    half *d_A, *d_B, *d_cublas, *d_cutlass;
    CHECK_CUDA(cudaMalloc((void **)&d_A, m * k * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, k * n * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&d_cublas, m * n * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void **)&d_cutlass, m * n * sizeof(half)));

    // Copy matrices from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cublas, h_C, m * n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cutlass, h_C, m * n * sizeof(half), cudaMemcpyHostToDevice));

    // Create a handle for cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform matrix multiplication: C = A * B
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, d_A, CUDA_R_16F, m,
                              d_B, CUDA_R_16F, k, &beta,
                              d_cublas, CUDA_R_16F, m, CUBLAS_COMPUTE_32F, 
                              CUBLAS_GEMM_DEFAULT));

    // Copy the result matrix back to host
    CHECK_CUDA(cudaMemcpy(h_D_cublas, d_cublas, m * n * sizeof(half), cudaMemcpyDeviceToHost));

    CHECK_CUDA(CutlassSgemmNN(m, n, k, alpha, (cutlass::half_t *)d_A, m, (cutlass::half_t *)d_B, k, beta, (cutlass::half_t *)d_cutlass, m, (cutlass::half_t *)d_cutlass, m));

    CHECK_CUDA(cudaMemcpy(h_D_cutlass, d_cutlass, m * n * sizeof(half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            auto cublas_val = __half2float(h_D_cublas[i * n + j]);
            auto cutlass_val = __half2float(h_D_cutlass[i * n + j]);

            if (abs(cublas_val - cutlass_val) > 0.1) {
                std::cout << "Results do not match." << std::endl;
                std::cout << "cublas_val: " << cublas_val << std::endl;
                std::cout << "cutlass_val: " << cutlass_val << std::endl;
                exit(1);
            }
        }
    }

    return 0;
}
