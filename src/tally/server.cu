#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda.h>

#include "libipc/ipc.h"

#include "tally/generated/server.h"

// To trigger CUDA register functions
__global__ void PlaceholderKernel(){}

int main(int argc, char ** argv) {

    TallyServer::server->start(1000);

    return 0;
}