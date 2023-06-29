#include <signal.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdlib.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <cassert>
#include <functional>
#include <unordered_map>
#include <cxxabi.h>
#include <map>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

#include "tally/util.h"
#include "tally/msg_struct.h"
#include "tally/cuda_api.h"
#include "tally/kernel_slice.h"
#include "tally/server.h"

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char ** argv) {

    tally_server.start(1000);

    return 0;
}