#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize NCCL
    ncclComm_t comm;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclCommInitRank(&comm, 1, id, 0);

    // Allocate host and device buffers
    int N = 1024;
    float *sendbuff, *recvbuff;
    cudaMalloc(&sendbuff, N * sizeof(float));
    cudaMalloc(&recvbuff, N * sizeof(float));

    // Initialize send buffer with some values
    std::vector<float> host_data(N, 1.0f); // Fill with 1.0f for example
    cudaMemcpy(sendbuff, host_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform all-reduce operation
    ncclAllReduce(sendbuff, recvbuff, N, ncclFloat, ncclSum, comm, stream);

    // Wait for the operation to complete
    cudaStreamSynchronize(stream);

    // Copy result back to host and print it
    std::vector<float> result(N);
    cudaMemcpy(result.data(), recvbuff, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Finalize
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);

    return 0;
}
