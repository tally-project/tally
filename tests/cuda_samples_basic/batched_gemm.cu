#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/gemm/device/gemm_batched.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include <cuda_runtime.h>
#include <iostream>

__global__ void batched_gemm_kernel(int m, int n, int k,
                                    const float *alpha,
                                    const float *A, int lda, long long int strideA,
                                    const float *B, int ldb, long long int strideB,
                                    const float *beta,
                                    float *C, int ldc, long long int strideC,
                                    int batchCount) {
    int batchIndex = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        const float *currentA = A + batchIndex * strideA;
        const float *currentB = B + batchIndex * strideB;
        float *currentC = C + batchIndex * strideC;

        float sum = 0.0f;
        for (int e = 0; e < k; ++e) {
            sum += currentA[row + e * lda] * currentB[e + col * ldb];
        }

        currentC[row + col * ldc] = *alpha * sum + *beta * currentC[row + col * ldc];
    }
}

cudaError_t cutlass_strided_batched_sgemm(
  int m, 
  int n,
  int k,
  float alpha,
  float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float beta,
  int batch_count) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

int main() {
    srand (1);

    int m = 2;
    int n = 3;
    int k = 4;
    int batch_count = 2;

    int stride_A = m * k;
    int stride_B = k * n;
    int stride_C = m * n;

    const float alpha = 1.0f;
    const float beta = 0.1f;

    float *h_A, *h_B, *h_C, *h_ref, *h_cublas, *h_cutlass;

    h_A = (float *) malloc(sizeof(float) * m * k * batch_count);
    h_B = (float *) malloc(sizeof(float) * k * n * batch_count);
    h_C = (float *) malloc(sizeof(float) * m * n * batch_count);
    h_cublas = (float *) malloc(sizeof(float) * m * n * batch_count);
    h_cutlass = (float *) malloc(sizeof(float) * m * n * batch_count);
    h_ref = (float *) malloc(sizeof(float) * m * n * batch_count);

    // Set values in h_A
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                (h_A + batch_idx * stride_A)[IDX2C(i, j, m)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
    }

    std::cout << "h_A:" << std::endl;
    
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                std::cout << (h_A + batch_idx * stride_A)[IDX2C(i, j, m)] << " ";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;
    }

    // Set values in h_B
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                (h_B + batch_idx * stride_B)[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
    }

    std::cout << "h_B:" << std::endl;

    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << (h_B + batch_idx * stride_B)[IDX2C(i, j, k)] << " ";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;
    }

    // Set values in h_C
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                (h_C + batch_idx * stride_C)[IDX2C(i, j, m)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
    }

    std::cout << "h_C:" << std::endl;

    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << (h_C + batch_idx * stride_C)[IDX2C(i, j, k)] << " ";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;
    }

    // Compute h_ref
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.f;
                for( int p = 0; p < k; p++ ) {
                    sum += alpha * ((h_A + batch_idx * stride_A)[IDX2C(i, p, m)] * (h_B + batch_idx * stride_B)[IDX2C(p, j, k)]);
                }
                (h_ref + batch_idx * stride_C)[IDX2C(i, j, m)] = alpha * sum + beta * (h_C + batch_idx * stride_C)[IDX2C(i, j, m)];
            }
        }
    }

    float* d_A, *d_B, *d_cublas, *d_cutlass;
    cudaMalloc((void**)&d_A, sizeof(float) * m * k * batch_count);
    cudaMalloc((void**)&d_B, sizeof(float) * k * n * batch_count);
    cudaMalloc((void**)&d_cublas, sizeof(float) * m * n * batch_count);
    cudaMalloc((void**)&d_cutlass, sizeof(float) * m * n * batch_count);

    cudaMemcpy(d_A, h_A, sizeof(float) * m * k * batch_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n * batch_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cublas, h_C, sizeof(float) * m * n * batch_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cutlass, h_C, sizeof(float) * m * n * batch_count, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemmStridedBatched(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m, n, k,
                              &alpha,
                              d_A, m, stride_A,
                              d_B, k, stride_B,
                              &beta,
                              d_cublas, m, stride_C,
                              batch_count);

    cudaMemcpy(h_cublas, d_cublas, sizeof(float) * m * n * batch_count, cudaMemcpyDeviceToHost);

    cutlass_strided_batched_sgemm(m, n, k, alpha, d_A, m, stride_A, d_B, k, stride_B, d_cutlass, m, stride_C, beta, batch_count);

    cudaMemcpy(h_cutlass, d_cutlass, sizeof(float) * m * n * batch_count, cudaMemcpyDeviceToHost);

    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                auto ref_val = (h_ref + batch_idx * stride_C)[IDX2C(i, j, m)];
                auto cublas_val = (h_cublas + batch_idx * stride_C)[IDX2C(i, j, m)];
                auto cutlass_val = (h_cutlass + batch_idx * stride_C)[IDX2C(i, j, m)];

                if (abs(cutlass_val - cublas_val) > 0.001 || abs(ref_val - cublas_val) > 0.001) {
                    std::cout << "Results do not match." << std::endl;
                    std::cout << "ref_val: " << ref_val << std::endl;
                    std::cout << "cublas_val: " << cublas_val << std::endl;
                    std::cout << "cutlass_val: " << cutlass_val << std::endl;
                    // exit(1);
                }
            }
        }
    }

    return 0;
}

// #include <cuda_runtime.h>
// #include <iostream>
// #include <cmath>
// #include <cstdlib>

// // CUDA Kernel definition
// // ... [Insert the batched_gemm_kernel definition from previous response here]
// __global__ void batched_gemm_kernel(int m, int n, int k,
//                                     const float alpha,
//                                     const float *A, int lda, long long int strideA,
//                                     const float *B, int ldb, long long int strideB,
//                                     const float beta,
//                                     float *C, int ldc, long long int strideC,
//                                     int batchCount) {
//     int batchIndex = blockIdx.z;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < m && col < n) {

//         const float *currentA = A + batchIndex * strideA;
//         const float *currentB = B + batchIndex * strideB;
//         float *currentC = C + batchIndex * strideC;

//         float sum = 0.0f;
//         for (int e = 0; e < k; ++e) {
//             sum += currentA[row * k + e] * currentB[e * n + col];
//         }

//         currentC[row * n + col] = alpha * sum + beta * currentC[row * n + col];
//     }
// }

// // CPU-based matrix multiplication for reference
// void cpu_matrix_multiply(float *A, float *B, float *C, float *D, int m, int n, int k, float alpha, float beta) {
//     for (int row = 0; row < m; ++row) {
//         for (int col = 0; col < n; ++col) {
//             float sum = 0.0f;
//             for (int e = 0; e < k; ++e) {
//                 sum += A[row * k + e] * B[e * n + col];
//             }
//             D[row * n + col] = alpha * sum + beta * C[row * n + col];
//         }
//     }
// }

// int main() {
//     srand (1);

//     // Matrix dimensions and batch count
//     int m = 128, n = 128, k = 128, batchCount = 128;
    
//     // Allocate and initialize matrices A, B, and C for both GPU and CPU
//     float *A, *B, *C, *C_cpu;
//     float *dev_A, *dev_B, *dev_C;

//     A = (float *) malloc(m * k * batchCount * sizeof(float));
//     B = (float *) malloc(k * n * batchCount * sizeof(float));
//     C = (float *) malloc(m * n * batchCount * sizeof(float));
//     C_cpu = (float*)malloc(m * n * batchCount * sizeof(float));

//     // Initialize A, B, and C with some values (for simplicity, using random values)
//     for (int i = 0; i < m * k * batchCount; ++i) {
//         A[i] = static_cast<float>(rand()) / RAND_MAX;
//     }
//     for (int i = 0; i < k * n * batchCount; ++i) {
//         B[i] = static_cast<float>(rand()) / RAND_MAX;
//     }
//     for (int i = 0; i < m * n * batchCount; ++i) {
//         C[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     cudaMalloc(&dev_A, m * k * batchCount * sizeof(float));
//     cudaMalloc(&dev_B, k * n * batchCount * sizeof(float));
//     cudaMalloc(&dev_C, m * n * batchCount * sizeof(float));

//     cudaMemcpy(dev_A, A, m * k * batchCount * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dev_B, B, k * n * batchCount * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dev_C, C, m * n * batchCount * sizeof(float), cudaMemcpyHostToDevice);

//     // Scalars for the GEMM operation
//     float alpha = 1.0f, beta = 0.1f;

//     // Leading dimensions and strides
//     int lda = m, ldb = k, ldc = m;
//     long long int strideA = m * k, strideB = k * n, strideC = m * n;

//     // Kernel launch parameters
//     dim3 threadsPerBlock(16, 16); 
//     dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                    (m + threadsPerBlock.y - 1) / threadsPerBlock.y,
//                    batchCount);

//     // Launch the kernel
//     batched_gemm_kernel<<<numBlocks, threadsPerBlock>>>(m, n, k, alpha, dev_A, lda, strideA, dev_B, ldb, strideB, beta, dev_C, ldc, strideC, batchCount);
//     cudaDeviceSynchronize();

//     // CPU-based reference calculation
//     for (int b = 0; b < batchCount; ++b) {
//         cpu_matrix_multiply(A + b * strideA, B + b * strideB, C + b * strideC, C_cpu + b * strideC, m, n, k, alpha, beta);
//     }

//     cudaMemcpy(C, dev_C, m * n * batchCount * sizeof(float), cudaMemcpyDeviceToHost);

//     // Compare the results
//     bool match = true;
//     for (int i = 0; i < m * n * batchCount; ++i) {
//         if (std::abs(C[i] - C_cpu[i]) > 1e-2) {

//             std::cout << "C[i]: " << C[i] << std::endl;
//             std::cout << "C_cpu[i]: " << C_cpu[i] << std::endl;

//             match = false;
//             break;
//         }
//     }

//     if (match) {
//         std::cout << "Test Passed: GPU and CPU results match." << std::endl;
//     } else {
//         std::cout << "Test Failed: GPU and CPU results do not match." << std::endl;
//     }

//     // Free memory
//     cudaFree(A);
//     cudaFree(B);
//     cudaFree(C);
//     free(C_cpu);

//     return 0;
// }
