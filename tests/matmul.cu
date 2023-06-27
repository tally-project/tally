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

__global__ void matrixMultiplyPTB(float *A, float *B, float *C, int width, dim3 original_gridSize) {

    uint32_t num_thread_blocks = original_gridSize.x * original_gridSize.y * original_gridSize.z;
    uint32_t part_size = (num_thread_blocks + gridDim.x - 1) / gridDim.x;
    uint32_t start_idx = blockIdx.x * part_size;
    uint32_t end_idx = start_idx + part_size;
    if (blockIdx.x == gridDim.x - 1) {
        end_idx = num_thread_blocks;
    }

    for (int tb_idx = start_idx; tb_idx < end_idx; tb_idx++) {
        int currBlockIdx_x = tb_idx / original_gridSize.x;
        int currBlockIdx_y = (tb_idx - currBlockIdx_x * original_gridSize.x) / original_gridSize.y;

        int row = currBlockIdx_y * blockDim.y + threadIdx.y;
        int col = currBlockIdx_x * blockDim.x + threadIdx.x;

        if (row < width && col < width) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }

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

    // same as before
    dim3 PTB_block_dim(16, 16);

    // Depend on number of PTBs/SM
    dim3 PTB_grid_dim(82 * 4);
    std::cout << "gridSize: " << gridSize.x << " " << gridSize.y << " " << gridSize.z << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // // Launch the kernel
    matrixMultiplyPTB<<<PTB_grid_dim, PTB_block_dim>>>(d_A, d_B, d_C, width, gridSize);
    
    // Launch kernel
    // matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %.2f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);

        printf("Device %d: %s\n", deviceIndex, deviceProperties.name);
        printf("Number of SMs: %d\n", deviceProperties.multiProcessorCount);
    }

    int width = 4096;
    float* arr_a = new float[width * width];
    float* arr_b = new float[width * width];
    float* res_gpu = new float[width * width];
    float* res_cpu = new float[width * width];

    std::srand(std::time(nullptr));
    
    // Initialize input arrays
    for (int i = 0; i < width * width; ++i) {
        arr_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        arr_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    runmatrixMultiply(arr_a, arr_b, res_gpu, width);
    runmatrixMultiplyCpu(arr_a, arr_b, res_cpu, width);

    for (int i = 0; i < width * width; ++i) {
        assert((res_gpu[i] - res_cpu[i]) < 0.0001);
    }

    delete[] arr_a;
    delete[] arr_b;
    delete[] res_cpu;
    delete[] res_gpu;

    return 0;
}
