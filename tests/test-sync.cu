#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>
#include <chrono>

#include "cuda.h"
#include "cuda_runtime.h"

__global__ void elementwiseAddition(float* a, float* b, float* c, int size) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

__host__ void runElementwiseAddition(float* arr_a, float* arr_b, float* arr_c, int size, bool ptb)
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
    dim3 block_dim(256);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);

    auto d_stream = nullptr;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; i++) {
        elementwiseAddition<<<grid_dim, block_dim, 0, d_stream>>>(deviceA, deviceB, deviceC, size);
        cudaDeviceSynchronize();
    }

    cudaStreamSynchronize(d_stream);

     auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Elapsed time: " << elapsed.count() << " milliseconds." << std::endl;

    cudaMemcpy(arr_c, deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

int main()
{
    int size = 16777216;
    bool ptb = false;
    
    // Allocate memory on the host (CPU)
    float* arr_a = new float[size];
    float* arr_b = new float[size];
    float* res_gpu = new float[size];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        arr_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        arr_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    runElementwiseAddition(arr_a, arr_b, res_gpu, size, ptb);
    
    // Cleanup
    delete[] arr_a;
    delete[] arr_b;
    delete[] res_gpu;

    return 0;
}