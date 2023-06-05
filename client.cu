#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

// nvcc -O0 -Xcicc -O0 -Xptxas -O0 client.cu -o client -cudart shared

__global__ void addOneKernel(int* array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
    if (tid < size) {
        array[tid] += 1;
    }
}

int main()
{
    const int arraySize = 10;
    int array[arraySize] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int ret_array[arraySize] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int *devArray;

    cudaMalloc((void**)&devArray, arraySize * sizeof(int));
    cudaMemcpy(devArray, array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Define execution configuration
    dim3 blockDim(256);
    dim3 gridDim((arraySize + blockDim.x - 1) / blockDim.x);

    // Launch the kernel
    addOneKernel<<<gridDim, blockDim>>>(devArray, arraySize);

    cudaMemcpy(ret_array, devArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the modified array
    for (int i = 0; i < arraySize; ++i) {
        std::cout << ret_array[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(devArray);

    return 0;
}