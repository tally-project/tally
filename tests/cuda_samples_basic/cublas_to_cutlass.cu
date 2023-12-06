#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>
#include <chrono>

#include <cublas_v2.h>

#include <tally/cutlass/cutlass.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main()
{
    srand (1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = 1024;
    int n = 60;
    int k = 1024;

    float *h_A, *h_B, *h_cublas, *h_cutlass;
    float *h_AT, *h_ref;

    // A will be k * m
    h_A = (float *) malloc(sizeof(float) * k * m);

    // AT will be m * k
    h_AT = (float *) malloc(sizeof(float) * m * k);

    // B will be k * n
    h_B = (float *) malloc(sizeof(float) * k * n);
    h_cublas = (float *) malloc(sizeof(float) * m * n);
    h_cutlass = (float *) malloc(sizeof(float) * m * n);

    h_ref = (float *) malloc(sizeof(float) * m * n);
    memset(h_ref, 0, sizeof(float) * m * n);

    // Set values in h_A
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < k; i++) {
            h_A[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // for (int i = 0; i < k; i++) {
    //     for (int j = 0; j < m; j++) {
    //         std::cout << h_A[IDX2C(i, j, k)] << " ";
    //     }

    //     std::cout << std::endl;
    // }

    //  std::cout << std::endl;

    // Set values in h_AT
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            h_AT[IDX2C(i, j, m)] = h_A[IDX2C(j, i, k)];
        }
    }

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < k; j++) {
    //         std::cout << h_AT[IDX2C(i, j, m)] << " ";
    //     }

    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;

    // Set values in h_B
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            h_B[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // for (int i = 0; i < k; i++) {
    //     for (int j = 0; j < n; j++) {
    //         std::cout << h_B[IDX2C(i, j, k)] << " ";
    //     }

    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for( int p = 0; p < k; p++ ) {
                h_ref[IDX2C(i, j, m)] += h_AT[IDX2C(i, p, m)] * h_B[IDX2C(p, j, k)];
            }
        }
    }

    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         std::cout << h_ref[IDX2C(i, j, m)] << " ";
    //     }

    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;

    // Allocate memory on the device
    float* d_A, *d_AT, *d_B, *d_cublas, *d_cutlass;
    cudaMalloc((void**)&d_A, sizeof(float) * k * m); // k x m matrix
    cudaMalloc((void**)&d_AT, sizeof(float) * m * k); // k x m matrix
    cudaMalloc((void**)&d_B, sizeof(float) * k * n); // k x n matrix
    cudaMalloc((void**)&d_cublas, sizeof(float) * m * n); // m x n matrix
    cudaMalloc((void**)&d_cutlass, sizeof(float) * m * n); // m x n matrix

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * k * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_AT, h_AT, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.f;

    cublasOperation_t transa = CUBLAS_OP_T; // No transpose for A
    cublasOperation_t transb = CUBLAS_OP_N; // No transpose for B

    cutlassOperation_t transa_cutlass = CUTLASS_OP_T;
    cutlassOperation_t transb_cutlass = CUTLASS_OP_N;

    // warmup
    cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, d_A, k /*lda*/, d_B, k /*ldb*/, &beta, d_cublas, m /*ldc*/);
    cutlassGemm_f32(transa_cutlass, transb_cutlass, m, n, k, alpha, d_A, k /*lda*/, d_B, k /*ldb*/, beta, d_cutlass, m /*ldc*/, d_cutlass, m /*ldd*/, NULL, NULL, NULL);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, d_A, k /*lda*/, d_B, k /*ldb*/, &beta, d_cublas, m /*ldc*/);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    auto cublas_ms = duration.count();

    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();

    // Run cutlass impl
    cutlassGemm_f32(transa_cutlass, transb_cutlass, m, n, k, alpha, d_A, k /*lda*/, d_B, k /*ldb*/, beta, d_cutlass, m /*ldc*/, d_cutlass, m /*ldd*/, NULL, NULL, NULL);

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto cutlass_ms = duration.count();

    std::cout << "cutlassGemm_f32: " << std::to_string(cutlass_ms) << "ms" << std::endl;
    std::cout << "cublasSgemm_v2: " << std::to_string(cublas_ms) << "ms" << std::endl;

    cudaMemcpy(h_cutlass, d_cutlass, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cublas, d_cublas, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {

            if (abs(h_cutlass[IDX2C(i, j, m)] - h_cublas[IDX2C(i, j, m)]) > 0.01) {
                std::cout << "Results do not match." << std::endl;
                std::cout << "(i, j): " << i << " " << j << std::endl;
                std::cout << "h_cutlass[IDX2C(i, j, m)]: " << h_cutlass[IDX2C(i, j, m)] << std::endl;
                std::cout << "h_cublas[IDX2C(i, j, m)]: " << h_cublas[IDX2C(i, j, m)] << std::endl;
                std::cout << "h_ref[IDX2C(i, j, m)]: " << h_ref[IDX2C(i, j, m)] << std::endl;
                exit(1);
            }

        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_cublas);
    cudaFree(d_cutlass);
    cublasDestroy(handle);

    return 0;
}