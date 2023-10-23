
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 10

__global__ void kernel(int *d_data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_data[tid] = tid;
    }
}

int main() {
    int h_data[N];  // Host array
    int *d_data;    // Device array
    CUcontext cuContext;
    CUdeviceptr d_data_ptr;
    CUdevice cuDevice;

    // Initialize CUDA
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Allocate device memory
    cuMemAlloc(&d_data_ptr, N * sizeof(int));
    d_data = (int*)d_data_ptr;

    // Launch kernel to fill device array
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    kernel<<<grid, block>>>(d_data, N);
    cuCtxSynchronize(); // Wait for the kernel to finish

    // Copy data from device to host asynchronously
    CUstream stream;
    cuStreamCreate(&stream, 0);
    cuMemcpyDtoHAsync(h_data, d_data_ptr, N * sizeof(int), stream);

    // Synchronize the stream to make sure the copy is complete
    cuStreamSynchronize(stream);

    // Verify the data
    for (int i = 0; i < N; i++) {
        if (h_data[i] != i) {
            fprintf(stderr, "Verification failed at element %d!\n", i);
            return 1;
        }
    }

    // Clean up
    // cuStreamDestroy(stream);
    // cuMemFree(d_data_ptr);
    // cuCtxDestroy(cuContext);

    printf("Test completed successfully!\n");
    return 0;
}

