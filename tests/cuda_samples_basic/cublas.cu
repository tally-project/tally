#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>

#include <cublas_v2.h>
#include "cuda.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = 128;
    int n = 128;
    int k = 128;

    float *h_A, *h_B, *h_C, *h_ref;
    h_A = (float *) malloc(sizeof(float) * m * k);
    h_B = (float *) malloc(sizeof(float) * k * n);
    h_C = (float *) malloc(sizeof(float) * m * n);
    h_ref = (float *) malloc(sizeof(float) * m * n);

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            h_A[IDX2C(i, j, m)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            h_B[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < n; p++) {
                h_ref[IDX2C(i, j, m)] += h_A[IDX2C(i, p, m)] * h_B[IDX2C(p, j, k)];
            }
        }
    }

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

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            if (abs(h_C[IDX2C(i, j, m)] - h_ref[IDX2C(i, j, m)]) > 0.001) {
                std::cout << "Results do not match." << std::endl;
                std::cout << "h_C[IDX2C(i, j, m)]: " << h_C[IDX2C(i, j, m)] << std::endl;
                std::cout << "h_ref[IDX2C(i, j, m)]: " << h_ref[IDX2C(i, j, m)] << std::endl;
                exit(1);
            }
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}