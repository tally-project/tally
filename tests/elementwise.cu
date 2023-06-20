#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>

#include "cuda.h"

__global__ void elementwiseAddition(float* a, float* b, float* c, int size) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

__host__ void runElementwiseAddition(float* arr_a, float* arr_b, float* arr_c, int size)
{
    // Allocate memory on the device (GPU)
    float* deviceA, * deviceB, * deviceC;
    cudaMalloc((void**)&deviceA, size * sizeof(float));
    cudaMalloc((void**)&deviceB, size * sizeof(float));
    cudaMalloc((void**)&deviceC, size * sizeof(float));
    
    // Copy input arrays from host to device
    cudaMemcpy(deviceA, arr_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, arr_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define execution configuration
    dim3 block_dim(512);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);

    // Launch the kernel
    for (size_t i = 0; i < 100; i++) { 
        elementwiseAddition<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, size);
    }

    cudaMemcpy(arr_c, deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

void runElementwiseAdditionCpu(float* arr_a, float* arr_b, float* arr_c, int size)
{
    for (size_t i = 0; i < size; i++) {
        arr_c[i] = arr_a[i] + arr_b[i];
    }
}

int main()
{
    int size = 144384;
    
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
    
    runElementwiseAddition(arr_a, arr_b, res_gpu, size);
    runElementwiseAdditionCpu(arr_a, arr_b, res_cpu, size);

    for (int i = 0; i < size; i++) {
        assert(res_gpu[i] == res_cpu[i]);
    }
    
    // Cleanup
    delete[] arr_a;
    delete[] arr_b;
    delete[] res_cpu;
    delete[] res_gpu;

    return 0;
}