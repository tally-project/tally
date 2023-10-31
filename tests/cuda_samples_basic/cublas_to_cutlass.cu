#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>

#include <cublas_v2.h>
#include "cuda.h"

#include "cutlass/gemm/device/gemm.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = 1024;
    int n = 1024;
    int k = 1024;

    float *h_A, *h_B, *h_cublas, *h_cutlass;
    h_A = (float *) malloc(sizeof(float) * m * k);
    h_B = (float *) malloc(sizeof(float) * k * n);
    h_cublas = (float *) malloc(sizeof(float) * m * n);
    h_cutlass = (float *) malloc(sizeof(float) * m * n);

    // Set values in h_A
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            h_A[IDX2C(i, j, m)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // Set values in h_B
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            h_B[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // Allocate memory on the device
    float* d_A, *d_B, *d_cublas, *d_cutlass;
    cudaMalloc((void**)&d_A, sizeof(float) * m * k); // m x k matrix
    cudaMalloc((void**)&d_B, sizeof(float) * k * n); // k x n matrix
    cudaMalloc((void**)&d_cublas, sizeof(float) * m * n); // m x n matrix
    cudaMalloc((void**)&d_cutlass, sizeof(float) * m * n); // m x n matrix

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.5f;

    cublasOperation_t transa = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t transb = CUBLAS_OP_N; // No transpose for B
    int lda = m; // Leading dimension of A (A is a m x k matrix)
    int ldb = k; // Leading dimension of B (B is a k x n matrix)
    int ldc = m; // Leading dimension of C (C is a m x n matrix)

    cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_cublas, ldc);

    cudaMemcpy(h_cublas, d_cublas, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    // Run cutlass impl
    CutlassSgemmNN(m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_cutlass, ldc);

    //
    cudaMemcpy(h_cutlass, d_cutlass, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            std::cout << h_cutlass[IDX2C(i, j, m)] << std::endl;
            std::cout << h_cublas[IDX2C(i, j, m)] << std::endl;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_cublas);
    cudaFree(d_cutlass);

    return 0;
}