
#ifndef TALLY_CUTLASS_H
#define TALLY_CUTLASS_H

#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <tally/cutlass/cutlass_struct.h>

template <typename T>
struct AddVecBiasFunctor {
    int rows;
    const T* vec;

    AddVecBiasFunctor(int _rows, const T* _vec) : rows(_rows), vec(_vec) {}

    __host__ __device__
    T operator()(const T& matrix_val, const int& idx) const {
        // Compute the corresponding index in the vector
        int vec_idx = idx % rows;
        return matrix_val + vec[vec_idx];
    }
};

extern "C" {

void tally_register_cutlass();

cudaError_t cutlassGemm_f32(
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

cudaError_t cutlassGemm_f16(
    cutlassOperation_t transA,
    cutlassOperation_t transB,
    int M,
    int N,
    int K,
    float alpha,
    half const *A,
    int lda,
    half const *B,
    int ldb,
    float beta,
    half *C,
    int ldc,
    half *D,
    int ldd,
    half *bias=nullptr,
    cudaStream_t stream=nullptr
);

cudaError_t cutlassStridedBatchedGemm_f32(
    cutlassOperation_t transA,
    cutlassOperation_t transB,
    int m, 
    int n,
    int k,
    float alpha,
    float const *A,
    int lda,
    long long int batch_stride_A,
    float const *B,
    int ldb,
    long long int batch_stride_B,
    float *C,
    int ldc,
    long long int batch_stride_C,
    float beta,
    int batch_count,
    cudaStream_t stream=nullptr
);

}

#endif // TALLY_CUTLASS_H