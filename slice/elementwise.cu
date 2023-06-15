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

__host__ void run(float* arr_a, float* arr_b, float* hostC, int size)
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

    // Launch the kernel
    elementwiseAddition<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, size);

    cudaMemcpy(hostC, deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

__host__ void run_sliced(float* arr_a, float* arr_b, float* hostC, int size)
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

    dim3 new_grid_dim(8);
    dim3 blockOffset = (0);

    std::ifstream t("elementwise_sliced.ptx");
    if (!t.is_open()) {
        std::cerr << "elementwise_sliced.ptx not found\n";
    }
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    CUmodule cudaModule;
    CUfunction function;
    cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0);
    cuModuleGetFunction(&function, cudaModule, "_Z19elementwiseAdditionPfS_S_i");

    while (blockOffset.x < grid_dim.x) {

        void *KernelParams[] = { &deviceA, &deviceB, &deviceC, &size, &blockOffset };
        cuLaunchKernel(function, new_grid_dim.x, new_grid_dim.y, new_grid_dim.z,
                        block_dim.x, block_dim.y, block_dim.z, 0, NULL, KernelParams, NULL);

        blockOffset.x += new_grid_dim.x;
    }

    cudaMemcpy(hostC, deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

int main()
{
    int size = 8060928;
    
    // Allocate memory on the host (CPU)
    float* arr_a = new float[size];
    float* arr_b = new float[size];
    float* res = new float[size];
    float* res_sliced = new float[size];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        arr_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        arr_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    
    run(arr_a, arr_b, res, size);
    run_sliced(arr_a, arr_b, res_sliced, size);

    // Print the result
    for (int i = 0; i < size; i++) {
        assert(res[i] == res_sliced[i]);
    }
    
    // Cleanup
    delete[] arr_a;
    delete[] arr_b;
    delete[] res;
    delete[] res_sliced;

    return 0;
}