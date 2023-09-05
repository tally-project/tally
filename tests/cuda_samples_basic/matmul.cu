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
    uint32_t xy_tbs = original_gridSize.x * original_gridSize.y;
    dim3 newBlockIdx(0, 0, 0);

    for (int tb_idx = blockIdx.x; tb_idx < num_thread_blocks; tb_idx += gridDim.x) {

        newBlockIdx.z = tb_idx / xy_tbs;
        newBlockIdx.y = (tb_idx - newBlockIdx.z * xy_tbs) / original_gridSize.x;
        newBlockIdx.x = (tb_idx - newBlockIdx.z * xy_tbs) - newBlockIdx.y * original_gridSize.x;

        int row = newBlockIdx.y * blockDim.y + threadIdx.y;
        int col = newBlockIdx.x * blockDim.x + threadIdx.x;

        if (row < width && col < width) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

__global__ void matrixMultiplyPTB_dynamic(float *A, float *B, float *C, int width, dim3 original_gridSize, int *global_idx, volatile bool *retreat) {

    const bool leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
    const uint32_t num_thread_blocks = original_gridSize.x * original_gridSize.y * original_gridSize.z;
    const uint32_t xy_tbs = original_gridSize.x * original_gridSize.y;

    __shared__ volatile unsigned int volatile_curr_idx;
    uint32_t curr_idx;

    while (true) {

        if (leader) {
            if (*retreat) {
                curr_idx = num_thread_blocks + 1;
            } else {
                curr_idx = atomicAdd(global_idx, 1);
            }
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

        int row = newBlockIdx.y * blockDim.y + threadIdx.y;
        int col = newBlockIdx.x * blockDim.x + threadIdx.x;

        if (row < width && col < width) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

__host__ void runmatrixMultiply(float *h_A, float *h_B, float *h_C, int width, bool ptb)
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
    std::cout << "gridSize: " << gridSize.x << " " << gridSize.y << " " << gridSize.z << std::endl;

    // same as before
    dim3 PTB_block_dim(16, 16);

    // Depend on number of PTBs/SM
    dim3 PTB_grid_dim(82 * 4);

    bool *retreat;
    cudaMalloc((void**)&retreat, sizeof(bool));

    int *global_idx;
    cudaMalloc((void**)&global_idx, sizeof(int));

    cudaStream_t kernel_stream;
    cudaStreamCreate(&kernel_stream);

    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);
 
    // warmup
    if (ptb) {
        matrixMultiplyPTB<<<PTB_grid_dim, PTB_block_dim>>>(d_A, d_B, d_C, width, gridSize);
    } else {
        matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    }
    
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Launch the kernel
    if (ptb) {
        // matrixMultiplyPTB<<<PTB_grid_dim, PTB_block_dim>>>(d_A, d_B, d_C, width, gridSize);

        cudaMemset(retreat, 0, sizeof(bool));
        cudaMemset(global_idx, 0, sizeof(int));

        matrixMultiplyPTB_dynamic<<<PTB_grid_dim, PTB_block_dim, 0, kernel_stream>>>(d_A, d_B, d_C, width, gridSize, global_idx, retreat);

        cudaMemsetAsync(retreat, 1, sizeof(bool), new_stream);

        int progress = 0;
        cudaMemcpy(&progress, global_idx, sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << "progress: " << progress << std::endl; 

        cudaMemsetAsync(retreat, 0, sizeof(bool), new_stream);

        matrixMultiplyPTB_dynamic<<<PTB_grid_dim, PTB_block_dim, 0, kernel_stream>>>(d_A, d_B, d_C, width, gridSize, global_idx, retreat);

        // cudaMemcpy(&progress, global_idx, sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << "progress: " << progress << std::endl; 

    } else {
        matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    }

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
    bool ptb = false;
    int width = 1024;
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

    runmatrixMultiply(arr_a, arr_b, res_gpu, width, ptb);
    runmatrixMultiplyCpu(arr_a, arr_b, res_cpu, width);

    for (int i = 0; i < width * width; ++i) {
        // std::cout << "res_gpu[i]: " << res_gpu[i] << " " << "res_cpu[i]: " << res_cpu[i] << std::endl;
        if (abs(res_gpu[i] - res_cpu[i]) > 0.001) {
            std::cerr << "result mismatch: res_gpu[i]: " << res_gpu[i] << " " << "res_cpu[i]: " << res_cpu[i] << std::endl;
            exit(1);
        }
    }

    delete[] arr_a;
    delete[] arr_b;
    delete[] res_cpu;
    delete[] res_gpu;

    return 0;
}

// #include <iostream>
// #include <cuda_runtime.h>

// const int TILE_SIZE = 16;

// // Kernel to perform matrix multiplication with tiling
// __global__ void matrixMulTiled(float* A, float* B, float* C, int N) {
//     int row = blockIdx.y * TILE_SIZE + threadIdx.y;
//     int col = blockIdx.x * TILE_SIZE + threadIdx.x;

//     __shared__ float tileA[TILE_SIZE][TILE_SIZE];
//     __shared__ float tileB[TILE_SIZE][TILE_SIZE];

//     float result = 0.0f;

//     for (int i = 0; i < N / TILE_SIZE; ++i) {
//         tileA[threadIdx.y][threadIdx.x] = A[row * N + (i * TILE_SIZE + threadIdx.x)];
//         tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];

//         __syncthreads();

//         for (int k = 0; k < TILE_SIZE; ++k) {
//             result += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
//         }

//         __syncthreads();
//     }

//     C[row * N + col] = result;
// }

// int main() {
//     int N = 1024; // Matrix size NxN

//     // Allocate memory for matrices A, B, and C on the host
//     float* h_A = new float[N * N];
//     float* h_B = new float[N * N];
//     float* h_C = new float[N * N];
//     float* res_cpu = new float[N * N];

//     // Initialize matrices h_A and h_B (fill with your data)
//     for (int i = 0; i < N * N; ++i) {
//         h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
//         h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
//     }

//     // Allocate memory for matrices A, B, and C on the device
//     float* d_A, *d_B, *d_C;
//     cudaMalloc((void**)&d_A, N * N * sizeof(float));
//     cudaMalloc((void**)&d_B, N * N * sizeof(float));
//     cudaMalloc((void**)&d_C, N * N * sizeof(float));

//     // Copy matrices A and B from host to device
//     cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

//     // Define grid and block dimensions
//     dim3 blockDim(TILE_SIZE, TILE_SIZE);
//     dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

//     // Launch the CUDA kernel
//     matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

//     // Copy the result matrix C from device to host
//     cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     runmatrixMultiplyCpu(h_A, h_B, res_cpu, N);

//     for (int i = 0; i < N * N; ++i) {
//         if (abs(h_C[i] - res_cpu[i]) > 0.001) {
//             std::cerr << "result mismatch: h_C[i]: " << h_C[i] << " " << "res_cpu[i]: " << res_cpu[i] << std::endl;
//             exit(1);
//         }
//     }

//     // Free host memory
//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C;

//     return 0;
// }
