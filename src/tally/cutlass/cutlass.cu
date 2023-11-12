
#include "cutlass/gemm/device/gemm.h"

#include <tally/cutlass/cutlass.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "cutlass/gemm/device/gemm.h"

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
    float *bias,
    void *workSpace,
    cudaStream_t stream
) {

    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    cutlass::Status status;

    if (transA == cutlassOperation_t::CUTLASS_OP_N && transB == cutlassOperation_t::CUTLASS_OP_N) {

        using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                        ColumnMajor,  // Layout of A matrix
                                                        float,        // Data-type of B matrix
                                                        ColumnMajor,  // Layout of B matrix
                                                        float,        // Data-type of C matrix
                                                        ColumnMajor>; // Layout of C matrix

        CutlassGemm gemm_operator;

        CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                    {A, lda},    // Tensor-ref for source matrix A
                                    {B, ldb},    // Tensor-ref for source matrix B
                                    {C, ldc},    // Tensor-ref for source matrix C
                                    {D, ldd},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                    {alpha, beta}); // Scalars used in the Epilogue


        status = gemm_operator(args, stream);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T && transB == cutlassOperation_t::CUTLASS_OP_N){
        using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                        RowMajor,  // Layout of A matrix
                                                        float,        // Data-type of B matrix
                                                        ColumnMajor,  // Layout of B matrix
                                                        float,        // Data-type of C matrix
                                                        ColumnMajor>; // Layout of C matrix

        CutlassGemm gemm_operator;

        CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                    {A, lda},    // Tensor-ref for source matrix A
                                    {B, ldb},    // Tensor-ref for source matrix B
                                    {C, ldc},    // Tensor-ref for source matrix C
                                    {D, ldd},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                    {alpha, beta}); // Scalars used in the Epilogue


        status = gemm_operator(args, stream);
    } else if (transA == cutlassOperation_t::CUTLASS_OP_N && transB == cutlassOperation_t::CUTLASS_OP_T){
        using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                        ColumnMajor,  // Layout of A matrix
                                                        float,        // Data-type of B matrix
                                                        RowMajor,  // Layout of B matrix
                                                        float,        // Data-type of C matrix
                                                        ColumnMajor>; // Layout of C matrix

        CutlassGemm gemm_operator;

        CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                    {A, lda},    // Tensor-ref for source matrix A
                                    {B, ldb},    // Tensor-ref for source matrix B
                                    {C, ldc},    // Tensor-ref for source matrix C
                                    {D, ldd},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                    {alpha, beta}); // Scalars used in the Epilogue


        status = gemm_operator(args, stream);
    } else if (transA == cutlassOperation_t::CUTLASS_OP_T && transB == cutlassOperation_t::CUTLASS_OP_T){
        using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                        RowMajor,  // Layout of A matrix
                                                        float,        // Data-type of B matrix
                                                        RowMajor,  // Layout of B matrix
                                                        float,        // Data-type of C matrix
                                                        ColumnMajor>; // Layout of C matrix

        CutlassGemm gemm_operator;

        CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                    {A, lda},    // Tensor-ref for source matrix A
                                    {B, ldb},    // Tensor-ref for source matrix B
                                    {C, ldc},    // Tensor-ref for source matrix C
                                    {D, ldd},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                    {alpha, beta}); // Scalars used in the Epilogue


        status = gemm_operator(args, stream);
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

}