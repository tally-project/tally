
#include "cutlass/gemm/device/gemm.h"

#include <tally/cutlass/cutlass.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "cutlass/gemm/device/gemm.h"

#define CUTLASS_GEMM_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                      \
                                             LAYOUT_A,                                          \
                                             ELEMENT_TYPE,                                      \
                                             LAYOUT_B,                                          \
                                             ELEMENT_TYPE,                                      \
                                             LAYOUT_C,                                          \
                                             ELEMENT_ACCUMULATOR>;                              \
    Gemm gemm_operator;                                                                         \
    Gemm::Arguments args({M, N, K},                                                             \
                        {(ELEMENT_TYPE *) A, lda},                                              \
                        {(ELEMENT_TYPE *) B, ldb},                                              \
                        {(ELEMENT_TYPE *) C, ldc},                                              \
                        {(ELEMENT_TYPE *) D, ldd},                                              \
                        {alpha, beta});                                                         \
    status = gemm_operator(args, stream);

struct AddVecBiasFunctor {
    int rows;
    const float* vec;

    AddVecBiasFunctor(int _rows, const float* _vec) : rows(_rows), vec(_vec) {}

    __host__ __device__
    float operator()(const float& matrix_val, const int& idx) const {
        // Compute the corresponding index in the vector
        int vec_idx = idx % rows;
        return matrix_val + vec[vec_idx];
    }
};

extern "C" {

void tally_register_cutlass()
{
    std::cout << "tally register cutlass ..." << std::endl;
}

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
    float *bias,
    void *workSpace,
    cudaStream_t stream
) {

    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    cutlass::Status status;

    if (transA == cutlassOperation_t::CUTLASS_OP_N &&
        transB == cutlassOperation_t::CUTLASS_OP_N) {

        CUTLASS_GEMM_TEMPLATE(float, ColumnMajor, ColumnMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_N){

        CUTLASS_GEMM_TEMPLATE(float, RowMajor, ColumnMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_N &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_TEMPLATE(float, ColumnMajor, RowMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_TEMPLATE(float, RowMajor, RowMajor, ColumnMajor, float);

    } else {
        throw std::runtime_error("Not implemented.");
    }

    if (bias) {
        thrust::device_ptr<float> D_thrust(D);

        thrust::transform(D_thrust, D_thrust + M * N, 
                          thrust::make_counting_iterator(0), 
                          D_thrust, 
                          AddVecBiasFunctor(M, thrust::raw_pointer_cast(bias)));
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

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
    cudaStream_t stream
) {
    using ElementAccumulator = float;
    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    cutlass::Status status;

    if (transA == cutlassOperation_t::CUTLASS_OP_N &&
        transB == cutlassOperation_t::CUTLASS_OP_N) {

        CUTLASS_GEMM_TEMPLATE(cutlass::half_t, ColumnMajor, ColumnMajor, ColumnMajor, ElementAccumulator);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_N){

        CUTLASS_GEMM_TEMPLATE(cutlass::half_t, RowMajor, ColumnMajor, ColumnMajor, ElementAccumulator);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_N &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_TEMPLATE(cutlass::half_t, ColumnMajor, RowMajor, ColumnMajor, ElementAccumulator);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_TEMPLATE(cutlass::half_t, RowMajor, RowMajor, ColumnMajor, ElementAccumulator);

    } else {
        throw std::runtime_error("Not implemented.");
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

}