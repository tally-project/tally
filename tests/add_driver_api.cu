#include <cuda.h>
#include <cstdio>

__global__ void addKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int numElements = 1024;
    const int blockSize = 256;
    const int gridSize = (numElements + blockSize - 1) / blockSize;

    // Initialize CUDA Driver API
    cuInit(0);

    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Allocate device memory
    CUdeviceptr d_a, d_b, d_c;
    cuMemAllocAsync(&d_a, numElements * sizeof(float), NULL);
    cuMemAllocAsync(&d_b, numElements * sizeof(float), NULL);
    cuMemAllocAsync(&d_c, numElements * sizeof(float), NULL);

    // Initialize host arrays
    float *h_a = new float[numElements];
    float *h_b = new float[numElements];
    float *h_c = new float[numElements];
    for (int i = 0; i < numElements; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Copy host data to device
    cuMemcpy(d_a, (CUdeviceptr)h_a, numElements * sizeof(float));
    cuMemcpy(d_b, (CUdeviceptr)h_b, numElements * sizeof(float));

    // Launch kernel
    addKernel<<<gridSize, blockSize>>>((float *)d_a, (float *)d_b, (float *)d_c, numElements);

    // Copy result back to host
    cuMemcpy((CUdeviceptr)h_c, d_c, numElements * sizeof(float));

    // Print some results
    for (int i = 0; i < 10; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);

    cuCtxDestroy(cuContext);

    return 0;
}
