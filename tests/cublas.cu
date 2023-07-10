#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>

#include <cublas_v2.h>
#include "cuda.h"

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = 1024;
    int n = 1024;
    int k = 1024;

    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(sizeof(float) * m * k);
    h_B = (float *) malloc(sizeof(float) * k * n);
    h_C = (float *) malloc(sizeof(float) * m * n);

    // Allocate memory on the device
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * m * k); // m x k matrix
    cudaMalloc((void**)&d_B, sizeof(float) * k * n); // k x n matrix
    cudaMalloc((void**)&d_C, sizeof(float) * m * n); // m x n matrix

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasOperation_t transa = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t transb = CUBLAS_OP_N; // No transpose for B
    int lda = m; // Leading dimension of A (A is a m x k matrix)
    int ldb = k; // Leading dimension of B (B is a k x n matrix)
    int ldc = m; // Leading dimension of C (C is a m x n matrix)

    cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

    cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}