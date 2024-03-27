#include <cuda_runtime.h>
#include <iostream>

// Callback function to be executed on the host
void CUDART_CB myHostCallback(void* userData) {
    // Cast userData back to its original type if needed
    int* hostData = static_cast<int*>(userData);
    std::cout << "Callback executed on host with data: " << *hostData << std::endl;
}

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Example GPU operation (e.g., kernel launch)
    // Assume we have a kernel called myKernel that's been defined elsewhere
    // myKernel<<<1, 1, 0, stream>>>();

    // Data to be passed to the callback function
    int hostData = 42;

    // Enqueue the host callback. The callback will be executed after
    // all preceding operations in the stream have completed.
    cudaLaunchHostFunc(stream, myHostCallback, &hostData);

    // Wait for the stream to complete to ensure the callback has been called
    cudaStreamSynchronize(stream);

    // Clean up
    cudaStreamDestroy(stream);

    return 0;
}
