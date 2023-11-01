
#include "cutlass/gemm/device/gemm.h"

#include <tally/cutlass/cutlass.h>

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
                                    {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
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
                                    {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
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
                                    {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
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
                                    {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                    {alpha, beta}); // Scalars used in the Epilogue


        status = gemm_operator(args, stream);
    } else {
        throw std::runtime_error("Not implemented.");
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

}