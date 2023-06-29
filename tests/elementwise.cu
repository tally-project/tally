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

__global__ void elementwiseAdditionPTB(float* a, float* b, float* c, int size, dim3 original_gridSize) {

    uint32_t num_thread_blocks = original_gridSize.x * original_gridSize.y * original_gridSize.z;
    uint32_t xy_tbs = original_gridSize.x * original_gridSize.y;
    dim3 newBlockIdx(0, 0, 0);

    for (int tb_idx = blockIdx.x; tb_idx < num_thread_blocks; tb_idx += gridDim.x) {

        newBlockIdx.z = tb_idx / xy_tbs;
        newBlockIdx.y = (tb_idx - newBlockIdx.z * xy_tbs) / original_gridSize.x;
        newBlockIdx.x = (tb_idx - newBlockIdx.z * xy_tbs) - newBlockIdx.y * original_gridSize.x;

        int tid = threadIdx.x + newBlockIdx.x * blockDim.x;
    
        if (tid < size) {
            c[tid] = a[tid] + b[tid];
        }
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

    // same as before
    dim3 PTB_block_dim(256);

    // Depend on number of PTBs/SM
    dim3 PTB_grid_dim(82 * 4);

    // Warm up
    if (ptb) {
        elementwiseAdditionPTB<<<PTB_grid_dim, PTB_block_dim>>>(deviceA, deviceB, deviceC, size, grid_dim);
    } else {
        elementwiseAddition<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, size);
    }

    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Launch the kernel
    if (ptb) {
        elementwiseAdditionPTB<<<PTB_grid_dim, PTB_block_dim>>>(deviceA, deviceB, deviceC, size, grid_dim);
    } else {
        elementwiseAddition<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, size);
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

void runElementwiseAdditionCpu(float* arr_a, float* arr_b, float* arr_c, int size)
{
    for (size_t i = 0; i < size; i++) {
        arr_c[i] = arr_a[i] + arr_b[i];
    }
}

int main()
{
    int size = 16777216;
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
    
    runElementwiseAddition(arr_a, arr_b, res_gpu, size, ptb);
    runElementwiseAdditionCpu(arr_a, arr_b, res_cpu, size);

    for (int i = 0; i < size; i++) {
        if (abs(res_gpu[i] - res_cpu[i]) > 0.0001) {
            std::cerr << "result mismatch: res_gpu[i]: " << res_gpu[i] << " " << "res_cpu[i]: " << res_gpu[i] << std::endl;
            exit(1);
        }
    }
    
    // Cleanup
    delete[] arr_a;
    delete[] arr_b;
    delete[] res_cpu;
    delete[] res_gpu;

    return 0;
}