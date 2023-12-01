#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/gemm/device/gemm_batched.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

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
    const float beta = 0.5f;

    float *h_A, *h_B, *h_ref, *h_cublas, *h_cutlass;

    h_A = (float *) malloc(sizeof(float) * m * k * batch_count);
    h_B = (float *) malloc(sizeof(float) * k * n * batch_count);
    h_cublas = (float *) malloc(sizeof(float) * m * n * batch_count);
    h_cutlass = (float *) malloc(sizeof(float) * m * n * batch_count);
    h_ref = (float *) malloc(sizeof(float) * m * n * batch_count);
    memset(h_ref, 0, sizeof(float) * m * n * batch_count);

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

    // Compute h_ref
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for( int p = 0; p < k; p++ ) {
                    (h_ref + batch_idx * stride_C)[IDX2C(i, j, m)] += alpha * ((h_A + batch_idx * stride_A)[IDX2C(i, p, m)] * (h_B + batch_idx * stride_B)[IDX2C(p, j, k)]);
                }
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

                if (abs(cutlass_val - cublas_val) > 0.001) {
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