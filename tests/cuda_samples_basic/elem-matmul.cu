#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>
#include <chrono>
#include <thread>
#include <unistd.h>

#include "cuda.h"

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

__global__ void elementwiseAddition(float* a, float* b, float* c, int size) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

__host__ void run_elem_matmul(float* arr_a, float* arr_b, float* arr_c, int size, bool ptb)
{
    int width = 4096;
    int matmul_size = width * width * sizeof(float);
    float *d_A, *d_B, *d_C;
    float* h_A = new float[width * width];
    float* h_B = new float[width * width];
    float* res_gpu = new float[width * width];
    float* res_cpu = new float[width * width];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < width * width; ++i) {
        arr_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        arr_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, matmul_size);
    cudaMalloc((void **)&d_B, matmul_size);
    cudaMalloc((void **)&d_C, matmul_size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, matmul_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matmul_size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);
    std::cout << "gridSize: " << gridSize.x << " " << gridSize.y << " " << gridSize.z << std::endl;


    // Allocate memory on the device (GPU)
    float* deviceA, * deviceB, * deviceC;
    cudaMalloc((void**)&deviceA, size * sizeof(float));
    cudaMalloc((void**)&deviceB, size * sizeof(float));
    cudaMalloc((void**)&deviceC, size * sizeof(float));

    // Copy input arrays from host to device
    cudaMemcpy(deviceA, arr_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, arr_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define execution configuration
    dim3 block_dim(256);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);

    // same as before
    dim3 PTB_block_dim(256);

    // Depend on number of PTBs/SM
    dim3 PTB_grid_dim(82 * 4);

    bool *retreat;
    cudaMalloc((void**)&retreat, sizeof(bool));

    uint32_t *global_idx;
    cudaMalloc((void**)&global_idx, sizeof(uint32_t));

    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream);

    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i = 0; i < 150; i++) {
        elementwiseAddition<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, size);
        matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %.2f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(arr_c, deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

void run_elem_matmulCpu(float* arr_a, float* arr_b, float* arr_c, int size)
{
    for (size_t i = 0; i < size; i++) {
        arr_c[i] = arr_a[i] + arr_b[i];
    }
}

int main()
{
    int size = 134217728;
    bool ptb = false;
    
    // Allocate memory on the host (CPU)
    float* arr_a = new float[size];
    float* arr_b = new float[size];
    float* res_gpu = new float[size];
    float* res_cpu = new float[size];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        arr_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        arr_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    run_elem_matmul(arr_a, arr_b, res_gpu, size, ptb);
    
    // Cleanup
    delete[] arr_a;
    delete[] arr_b;
    delete[] res_cpu;
    delete[] res_gpu;

    return 0;
}