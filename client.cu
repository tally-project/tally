#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// nvcc client.cu -o client -cudart shared

__global__ void addArrays(int* a, int* b, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

int main() {
    int size = 100;
    int a[size], b[size], result[size];
    int *devA, *devB, *devResult;

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i + 1;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&devA, size * sizeof(int));
    cudaMalloc((void**)&devB, size * sizeof(int));
    cudaMalloc((void**)&devResult, size * sizeof(int));

    // Copy input arrays from host to device
    cudaMemcpy(devA, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addArrays<<<numBlocks, blockSize>>>(devA, devB, devResult, size);

    // Copy result from device to host
    cudaMemcpy(result, devResult, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devResult);

    return 0;
}
