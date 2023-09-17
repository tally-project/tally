#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

CUmodule    cudaModule;
CUfunction  function;

uint32_t *global_idx[1000];

static __global__ void __launch_bounds__(512) hidet_mul_compute_z(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z) {
  z[((((((int)blockIdx.x * 512) + (int)threadIdx.x) / 256) * 256) + ((((int)blockIdx.x * 512) + (int)threadIdx.x) % 256))] = (x[((((((int)blockIdx.x * 512) + (int)threadIdx.x) / 256) * 256) + ((((int)blockIdx.x * 512) + (int)threadIdx.x) % 256))] * y[((((int)blockIdx.x * 512) + (int)threadIdx.x) / 256)]);
}

static __global__ void __launch_bounds__(512) hidet_mul_compute_z_ptb(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z, dim3 original_gridSize) {
  
    uint32_t num_thread_blocks = original_gridSize.x * original_gridSize.y * original_gridSize.z;
    uint32_t xy_tbs = original_gridSize.x * original_gridSize.y;
    dim3 newBlockIdx(0, 0, 0);

    for (int tb_idx = blockIdx.x; tb_idx < num_thread_blocks; tb_idx += gridDim.x) {

        // newBlockIdx.z = tb_idx / xy_tbs;
        // newBlockIdx.y = (tb_idx - newBlockIdx.z * xy_tbs) / original_gridSize.x;
        newBlockIdx.x = (tb_idx - newBlockIdx.z * xy_tbs) - newBlockIdx.y * original_gridSize.x;

        z[((((((int)newBlockIdx.x * 512) + (int)threadIdx.x) / 256) * 256) + ((((int)newBlockIdx.x * 512) + (int)threadIdx.x) % 256))] = (x[((((((int)newBlockIdx.x * 512) + (int)threadIdx.x) / 256) * 256) + ((((int)newBlockIdx.x * 512) + (int)threadIdx.x) % 256))] * y[((((int)newBlockIdx.x * 512) + (int)threadIdx.x) / 256)]);
    }
}

static __global__ void __launch_bounds__(512) hidet_mul_compute_z_ptb_dynamic(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z, dim3 original_gridSize, uint32_t *__global_idx) {
  
    const bool leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
    const uint32_t num_thread_blocks = original_gridSize.x * original_gridSize.y * original_gridSize.z;
    const uint32_t xy_tbs = original_gridSize.x * original_gridSize.y;

    __shared__ volatile uint32_t volatile_curr_idx;
    uint32_t curr_idx;

    while (true) {

        if (leader) {
            volatile_curr_idx = atomicAdd(__global_idx, 1);
        }

        __syncthreads();

        curr_idx = volatile_curr_idx;

        if (curr_idx >= num_thread_blocks) {
            break;
        }

        dim3 newBlockIdx(0, 0, 0);

        newBlockIdx.z = curr_idx / xy_tbs;
        newBlockIdx.y = (curr_idx - newBlockIdx.z * xy_tbs) / original_gridSize.x;
        newBlockIdx.x = (curr_idx - newBlockIdx.z * xy_tbs) - newBlockIdx.y * original_gridSize.x;

        // newBlockIdx.x = curr_idx;

        z[((((((int)newBlockIdx.x * 512) + (int)threadIdx.x) / 256) * 256) + ((((int)newBlockIdx.x * 512) + (int)threadIdx.x) % 256))] = (x[((((((int)newBlockIdx.x * 512) + (int)threadIdx.x) / 256) * 256) + ((((int)newBlockIdx.x * 512) + (int)threadIdx.x) % 256))] * y[((((int)newBlockIdx.x * 512) + (int)threadIdx.x) / 256)]);
    }
}

void hidet_get_input_shape(int32_t idx, int32_t * __restrict__ dims) {
  if (idx == 0) {
    dims[0] = 1024;
    dims[1] = 256;
    dims[2] = 1;
    dims[3] = 1;
  } 
  if (idx == 1) {
    dims[0] = 1024;
    dims[1] = 1;
    dims[2] = 1;
    dims[3] = 1;
  } 
}

void hidet_get_output_shape(int32_t idx, int32_t * __restrict__ dims) {
  if (idx == 0) {
    dims[0] = 1024;
    dims[1] = 256;
    dims[2] = 1;
    dims[3] = 1;
  } 
}

void hidet_launch_0(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z) {
    for (int i = 0; i < 1000; i++) {
        hidet_mul_compute_z<<<dim3(512, 1, 1), dim3(512, 1, 1), 0, NULL>>>(x, y, z);
    }
}

void hidet_launch_0_ptb(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z) {

    dim3 original_gridSize(512, 1, 1);

    void *KernelParams[] = { (void *)&x, (void *)&y, (void *)&z, (void *)&original_gridSize };

    for (int i = 0; i < 1000; i++) {
        // cuLaunchKernel(function, 82 * 1, 1, 1,
        //                             512, 1, 1,
        //                             0, NULL, KernelParams, NULL);

        hidet_mul_compute_z_ptb<<<dim3(82 * 4, 1, 1), dim3(512, 1, 1), 0, NULL>>>(x, y, z, dim3(512, 1, 1));
    }
}

void hidet_launch_0_ptb_dynamic(float * __restrict__ x, float * __restrict__ y, float * __restrict__ z) {

    for (int i = 0; i < 1000; i++) {
        hidet_mul_compute_z_ptb_dynamic<<<dim3(82 * 4, 1, 1), dim3(512, 1, 1), 0, NULL>>>(x, y, z, dim3(512, 1, 1), global_idx[i]);
    }
}


int main() {

    int width = 5120;
    float* arr_a = new float[width * width];
    float* arr_b = new float[width * width];
    float* res_gpu = new float[width * width];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < width * width; ++i) {
        arr_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        arr_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    int size = width * width * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, arr_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, arr_b, size, cudaMemcpyHostToDevice);

    for (int i = 0; i < 1000; i++) {
        cudaMalloc((void**)&(global_idx[i]), sizeof(uint32_t));
        cudaMemset(global_idx[i], 0, sizeof(uint32_t));
    }

    std::ifstream t("kernel.ptx");
    if (!t.is_open()) {
        std::cerr << "kernel.ptx not found\n";
    }
    std::string str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());

    cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0);
    cuModuleGetFunction(&function, cudaModule, "_Z19hidet_mul_compute_zPfS_S_");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // hidet_launch_0(d_A, d_B, d_C);
    // hidet_launch_0_ptb(d_A, d_B, d_C);
    hidet_launch_0_ptb_dynamic(d_A, d_B, d_C);
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %.2f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}