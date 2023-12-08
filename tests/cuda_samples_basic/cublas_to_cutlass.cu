#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>
#include <chrono>

#include <cublas_v2.h>
#include <cublasLt.h>

#include <tally/cutlass/cutlass.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main()
{
    srand (1);

    int m = 768;
    int n = 51865;
    int k = 1376;

    int lda = 768;
    int ldb = 51865;
    int ldc = 768;

    bool use_fp16 = true;

    std::cout << "M: " << m << " N: " << n << " K: " << k << std::endl;  
    std::cout << "use_fp16: " << use_fp16 << std::endl;  

    void *h_A, *h_B;
    void *h_cublas, *h_cutlass;

    size_t A_size_bytes;
    size_t B_size_bytes;
    size_t C_size_bytes;

    if (use_fp16) {
        A_size_bytes = sizeof(half) * m * k;
        B_size_bytes = sizeof(half) * k * n;
        C_size_bytes = sizeof(half) * m * n;
    } else {
        A_size_bytes = sizeof(float) * m * k;
        B_size_bytes = sizeof(float) * k * n;
        C_size_bytes = sizeof(float) * m * n;
    }

    // A will be k * m
    h_A = malloc(A_size_bytes);
    // B will be k * n
    h_B = malloc(B_size_bytes);

    h_cublas = malloc(C_size_bytes);
    h_cutlass = malloc(C_size_bytes);

    // Set values in h_A
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < k; i++) {
            float val = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

            if (use_fp16) {
                ((half *)h_A)[IDX2C(i, j, k)] = __float2half(val);
            } else {
                ((float *)h_A)[IDX2C(i, j, k)] = val;
            }
        }
    }

    // Set values in h_B
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            float val = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

            if (use_fp16) {
                ((half *)h_B)[IDX2C(i, j, k)] = __float2half(val);
            } else {
                ((float *)h_B)[IDX2C(i, j, k)] = val;
            }
        }
    }

    // Allocate memory on the device
    void* d_A, *d_B;
    float *d_cublas, *d_cutlass;
    cudaMalloc(&d_A, A_size_bytes);
    cudaMalloc(&d_B, B_size_bytes);
    cudaMalloc(&d_cublas, C_size_bytes);
    cudaMalloc(&d_cutlass, C_size_bytes);
    cudaMemset(d_cublas, 0, C_size_bytes);
    cudaMemset(d_cutlass, 0, C_size_bytes);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, A_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size_bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.f;

    cublasOperation_t transa = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t transb = CUBLAS_OP_N; // No transpose for B

    cutlassOperation_t transa_cutlass = CUTLASS_OP_N;
    cutlassOperation_t transb_cutlass = CUTLASS_OP_N;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasLtHandle_t lightHandle;
    cublasLtCreate(&lightHandle);

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_layout;
    cublasLtMatrixLayout_t B_layout;
    cublasLtMatrixLayout_t C_layout;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;

    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_16F, m, k, m);
    cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_16F, k, n, k);
    cublasLtMatrixLayoutCreate(&C_layout, CUDA_R_16F, m, n, m);

    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulPreferenceCreate(&preference);

    cublasLtMatmulAlgoGetHeuristic(lightHandle, matmul_desc, A_layout, B_layout, C_layout, C_layout, preference, 1, &heuristicResult, &returnedResults);

    // warmup
    if (use_fp16) {
        cublasLtMatmul(lightHandle, matmul_desc, &alpha, d_A, A_layout, d_B, B_layout, &beta, d_cublas, C_layout, d_cublas, C_layout, &heuristicResult.algo, NULL, 0, 0);
        cutlassGemm_f16(transa_cutlass, transb_cutlass, m, n, k, alpha, (half *)d_A, lda /*lda*/, (half *)d_B, ldb /*ldb*/, beta, (half *)d_cutlass, ldc /*ldc*/, (half *)d_cutlass, ldc /*ldd*/, NULL, NULL);
    } else {
        cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, (float *)d_A, lda /*lda*/, (float *)d_B, ldb /*ldb*/, &beta, d_cublas, ldc /*ldc*/);
        cutlassGemm_f32(transa_cutlass, transb_cutlass, m, n, k, alpha, (float *)d_A, lda /*lda*/, (float *)d_B, ldb /*ldb*/, beta, (float *)d_cutlass, ldc /*ldc*/, (float *)d_cutlass, ldc /*ldd*/, NULL, NULL, NULL);
    }

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (use_fp16) {
        cublasLtMatmul(lightHandle, matmul_desc, &alpha, d_A, A_layout, d_B, B_layout, &beta, d_cublas, C_layout, d_cublas, C_layout, &heuristicResult.algo, NULL, 0, 0);
    } else {
        cublasSgemm_v2(handle, transa, transb, m, n, k, &alpha, (float *)d_A, lda /*lda*/, (float *)d_B, ldb /*ldb*/, &beta, d_cublas, ldc /*ldc*/);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    auto cublas_ms = duration.count();

    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();

    // Run cutlass impl
    if (use_fp16) {
        cutlassGemm_f16(transa_cutlass, transb_cutlass, m, n, k, alpha, (half *)d_A, lda /*lda*/, (half *)d_B, ldb /*ldb*/, beta, (half *)d_cutlass, ldc /*ldc*/, (half *)d_cutlass, ldc /*ldd*/, NULL, NULL);
    } else {
       cutlassGemm_f32(transa_cutlass, transb_cutlass, m, n, k, alpha, (float *)d_A, lda /*lda*/, (float *)d_B, ldb /*ldb*/, beta, (float *)d_cutlass, ldc /*ldc*/, (float *)d_cutlass, ldc /*ldd*/, NULL, NULL, NULL);
    }

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto cutlass_ms = duration.count();

    std::cout << "cutlass: " << std::to_string(cutlass_ms) << "ms" << std::endl;
    std::cout << "cublas: " << std::to_string(cublas_ms) << "ms" << std::endl;

    cudaMemcpy(h_cutlass, d_cutlass, C_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cublas, d_cublas, C_size_bytes, cudaMemcpyDeviceToHost);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {

            float cublas_val;
            float cutlass_val;

            if (use_fp16) {
                cublas_val = __half2float(((half *)h_cublas)[i * n + j]);
                cutlass_val = __half2float(((half *)h_cutlass)[i * n + j]);
            } else {
                cublas_val = ((float *)h_cublas)[i * n + j];
                cutlass_val = ((float *)h_cutlass)[i * n + j];
            }

            if (abs(cublas_val - cutlass_val) > 1) {
                std::cout << "Results do not match." << std::endl;
                std::cout << "idx: " << (i * n + j) << std::endl;
                std::cout << "cublas_val: " << cublas_val << std::endl;
                std::cout << "cutlass_val: " << cutlass_val << std::endl;
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