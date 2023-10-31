
#ifndef TALLY_CUTLASS_H
#define TALLY_CUTLASS_H

#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

extern "C" {

void tally_register_cutlass();

cudaError_t CutlassSgemmNN(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc,
    cudaStream_t stream
);

}

#endif // TALLY_CUTLASS_H