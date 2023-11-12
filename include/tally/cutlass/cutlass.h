
#ifndef TALLY_CUTLASS_H
#define TALLY_CUTLASS_H

#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <tally/cutlass/cutlass_struct.h>

extern "C" {

void tally_register_cutlass();

cudaError_t CutlassSgemmNN(
    cutlassOperation_t transA,
    cutlassOperation_t transB,
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
    float *D,
    int ldd,
    float *bias=nullptr,
    void *workSpace=nullptr,
    cudaStream_t stream=nullptr
);

}

#endif // TALLY_CUTLASS_H