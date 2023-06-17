#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <sys/time.h>

__global__ void matrixMultiply(float *A, float *B, float *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

__host__ void runmatrixMultiply(float *h_A, float *h_B, float *h_C, int width)
{
    int size = width * width * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void runmatrixMultiplyCpu(float *h_A, float *h_B, float *h_C, int width)
{
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += h_A[i * width + k] * h_B[k * width + j];
            }
            h_C[i * width + j] = sum;
        }
    }
}

int main()
{
    int width = 64;
    float h_A[64][64];
    float h_B[64][64];
    float h_C_cpu[64][64];
    float h_C_gpu[64][64];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            h_A[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
            h_B[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    runmatrixMultiply((float *)h_A, (float *)h_B, (float *)h_C_gpu, width);
    runmatrixMultiplyCpu((float *)h_A, (float *)h_B, (float *)h_C_cpu, width);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            assert((h_C_gpu[i][j] - h_C_cpu[i][j]) < 0.0001);
        }
    }

    return 0;
}
