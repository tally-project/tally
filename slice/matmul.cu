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

__global__ void matrixMultiplySliced(float *A, float *B, float *C, int width, dim3 blockOffset)
{
    int real_blockIdx_x = blockIdx.x + blockOffset.x;
    int real_blockIdx_y = blockIdx.y + blockOffset.y;

    int row = real_blockIdx_y * blockDim.y + threadIdx.y;
    int col = real_blockIdx_x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

__host__ void run(float *h_A, float *h_B, float *h_C, int width)
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

__host__ void run_sliced(float *h_A, float *h_B, float *h_C, int width)
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
    
    dim3 newGridSize(4, 4);
    dim3 blockOffset(0, 0);

    while (blockOffset.x < gridSize.x && blockOffset.y < gridSize.y) {

        matrixMultiplySliced<<<newGridSize, blockSize>>>(d_A, d_B, d_C, width, blockOffset);

        blockOffset.x += newGridSize.x;

        if (blockOffset.x >= gridSize.x) {
            blockOffset.x = 0;
            blockOffset.y += newGridSize.y;
        }
    }

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int width = 64;
    float h_A[64][64];
    float h_B[64][64];
    float h_C[64][64];
    float h_C_sliced[64][64];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            h_A[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
            h_B[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    run((float *)h_A, (float *)h_B, (float *)h_C, width);
    run_sliced((float *)h_A, (float *)h_B, (float *)h_C_sliced, width);

    // Print the result matrix
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            assert(h_C[i][j] == h_C_sliced[i][j]);
        }
    }

    return 0;
}