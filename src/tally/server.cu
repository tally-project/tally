#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

#include "tally/server.h"

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char ** argv) {

    TallyServer::server->start(1000);

    return 0;
}