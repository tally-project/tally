#include <map>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"

#include <tally/cutlass/cutlass.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// Get the workspace with given size
// Will keep one workspace for each stream
void *get_workspace(size_t size, cudaStream_t stream) {
    static std::map<cudaStream_t, std::pair<size_t, void *>> workspace_map;

    if (workspace_map.find(stream) == workspace_map.end()) {
        void *workspace;
        cudaMalloc(&workspace, size);
        workspace_map[stream].first = size;
        workspace_map[stream].second = workspace;
    }

    return workspace_map[stream].second;
}

#define CUTLASS_GEMM_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                      \
                                             LAYOUT_A,                                          \
                                             ELEMENT_TYPE,                                      \
                                             LAYOUT_B,                                          \
                                             ELEMENT_TYPE,                                      \
                                             LAYOUT_C,                                          \
                                             ELEMENT_ACCUMULATOR>;                              \
    Gemm gemm_op;                                                                               \
    Gemm::Arguments args({M, N, K},                                                             \
                        {(ELEMENT_TYPE *) A, lda},                                              \
                        {(ELEMENT_TYPE *) B, ldb},                                              \
                        {(ELEMENT_TYPE *) C, ldc},                                              \
                        {(ELEMENT_TYPE *) D, ldd},                                              \
                        {alpha, beta});                                                         \
    status = gemm_op(args, nullptr, stream);

#define CUTLASS_GEMM_SPLIT_K_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)          \
    using Gemm = cutlass::gemm::device::GemmSplitKParallel<ELEMENT_TYPE,                                        \
                                                            LAYOUT_A,                                          \
                                                            ELEMENT_TYPE,                                      \
                                                            LAYOUT_B,                                          \
                                                            ELEMENT_TYPE,                                      \
                                                            LAYOUT_C,                                          \
                                                            ELEMENT_ACCUMULATOR>;                              \
    Gemm gemm_op;                                                                               \
    Gemm::Arguments args({M, N, K},                                                             \
                        {(ELEMENT_TYPE *) A, lda},                                              \
                        {(ELEMENT_TYPE *) B, ldb},                                              \
                        {(ELEMENT_TYPE *) C, ldc},                                              \
                        {(ELEMENT_TYPE *) D, ldd},                                              \
                        {alpha, beta},                                                          \
                        split_k_slices);                                                        \
    size_t workspace_size = Gemm::get_workspace_size(args);                                     \
    void *workspace = get_workspace(workspace_size, stream);                                    \
    gemm_op.initialize(args, workspace);                                                        \
    status = gemm_op(stream);

#define CUTLASS_GEMM_BATCHED_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    using Gemm = cutlass::gemm::device::GemmBatched<ELEMENT_TYPE,                                       \
                                                    LAYOUT_A,                                           \
                                                    ELEMENT_TYPE,                                       \
                                                    LAYOUT_B,                                           \
                                                    ELEMENT_TYPE,                                       \
                                                    LAYOUT_C,                                           \
                                                    ELEMENT_ACCUMULATOR>;                               \
    Gemm gemm_op;                                                                                       \
    Gemm::Arguments args({M, N, K},                                                                     \
                        {(ELEMENT_TYPE *) A, lda},                                                      \
                        batch_stride_A,                                                                 \
                        {(ELEMENT_TYPE *) B, ldb},                                                      \
                        batch_stride_B,                                                                 \
                        {(ELEMENT_TYPE *) C, ldc},                                                      \
                        batch_stride_C,                                                                 \
                        {(ELEMENT_TYPE *) C, ldc},                                                      \
                        batch_stride_C,                                                                 \
                        {alpha, beta},                                                                  \
                        batch_count);                                                                   \
    status = gemm_op(args, nullptr, stream);

#define INVOKE_CUTLASS_GEMM_TEMPLATE(TEMPLATE_NAME, ELEMENT_TYPE, ElementAccumulator)                           \
    if (transA == cutlassOperation_t::CUTLASS_OP_N && transB == cutlassOperation_t::CUTLASS_OP_N) {             \
        TEMPLATE_NAME(ELEMENT_TYPE, ColumnMajor, ColumnMajor, ColumnMajor, ElementAccumulator);                 \
    } else if (transA == cutlassOperation_t::CUTLASS_OP_T && transB == cutlassOperation_t::CUTLASS_OP_N) {      \
        TEMPLATE_NAME(ELEMENT_TYPE, RowMajor, ColumnMajor, ColumnMajor, ElementAccumulator);                    \
    } else if (transA == cutlassOperation_t::CUTLASS_OP_N && transB == cutlassOperation_t::CUTLASS_OP_T) {      \
        TEMPLATE_NAME(ELEMENT_TYPE, ColumnMajor, RowMajor, ColumnMajor, ElementAccumulator);                    \
    } else if (transA == cutlassOperation_t::CUTLASS_OP_T && transB == cutlassOperation_t::CUTLASS_OP_T)  {     \
        TEMPLATE_NAME(ELEMENT_TYPE, RowMajor, RowMajor, ColumnMajor, ElementAccumulator);                       \
    } else {                                                                                                    \
        throw std::runtime_error("Not implemented.");                                                           \
    }

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

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

    cutlass::Status status;

    if (M == 1024 && N == 60 && K == 1024) {
        int split_k_slices = 32;
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_SPLIT_K_TEMPLATE, float, float);
    } else {
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_TEMPLATE, float, float);
    }

    if (bias) {
        thrust::device_ptr<float> D_thrust(D);

        thrust::transform(thrust::cuda::par.on(stream),
                          D_thrust,
                          D_thrust + M * N, 
                          thrust::make_counting_iterator(0), 
                          D_thrust, 
                          AddVecBiasFunctor<float>(M, thrust::raw_pointer_cast(bias)));
    }

    if (status != cutlass::Status::kSuccess) {
        std::cout << "status != cutlass::Status::kSuccess" << std::endl;
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
    half *bias,
    cudaStream_t stream
) {
    using ElementAccumulator = float;
    
    cutlass::Status status;

    INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_TEMPLATE, cutlass::half_t, ElementAccumulator);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    if (bias) {
        thrust::device_ptr<half> D_thrust(D);

        thrust::transform(thrust::cuda::par.on(stream),
                          D_thrust,
                          D_thrust + M * N, 
                          thrust::make_counting_iterator(0), 
                          D_thrust, 
                          AddVecBiasFunctor<half>(M, thrust::raw_pointer_cast(bias)));
    }

    return cudaSuccess;
}

cudaError_t cutlassStridedBatchedGemm_f32(
    cutlassOperation_t transA,
    cutlassOperation_t transB,
    int M, 
    int N,
    int K,
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
    cudaStream_t stream
) {

    cutlass::Status status;

    if (transA == cutlassOperation_t::CUTLASS_OP_N &&
        transB == cutlassOperation_t::CUTLASS_OP_N) {

        CUTLASS_GEMM_BATCHED_TEMPLATE(float, ColumnMajor, ColumnMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_N){

        CUTLASS_GEMM_BATCHED_TEMPLATE(float, RowMajor, ColumnMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_N &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_BATCHED_TEMPLATE(float, ColumnMajor, RowMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_BATCHED_TEMPLATE(float, RowMajor, RowMajor, ColumnMajor, float);

    } else {
        throw std::runtime_error("Not implemented.");
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;

}

cudaError_t cutlassStridedBatchedGemm_f16(
    cutlassOperation_t transA,
    cutlassOperation_t transB,
    int M, 
    int N,
    int K,
    float alpha,
    half const *A,
    int lda,
    long long int batch_stride_A,
    half const *B,
    int ldb,
    long long int batch_stride_B,
    half *C,
    int ldc,
    long long int batch_stride_C,
    float beta,
    int batch_count,
    cudaStream_t stream
) {

    cutlass::Status status;

    if (transA == cutlassOperation_t::CUTLASS_OP_N &&
        transB == cutlassOperation_t::CUTLASS_OP_N) {

        CUTLASS_GEMM_BATCHED_TEMPLATE(cutlass::half_t, ColumnMajor, ColumnMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_N){

        CUTLASS_GEMM_BATCHED_TEMPLATE(cutlass::half_t, RowMajor, ColumnMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_N &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_BATCHED_TEMPLATE(cutlass::half_t, ColumnMajor, RowMajor, ColumnMajor, float);

    } else if (transA == cutlassOperation_t::CUTLASS_OP_T &&
               transB == cutlassOperation_t::CUTLASS_OP_T){

        CUTLASS_GEMM_BATCHED_TEMPLATE(cutlass::half_t, RowMajor, RowMajor, ColumnMajor, float);

    } else {
        throw std::runtime_error("Not implemented.");
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;

}

}