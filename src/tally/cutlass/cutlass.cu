#include <map>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"

#include <tally/cutlass/cutlass.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define LAUNCH_GEMM                                                                             \
    status = gemm_op.can_implement(args);                                                       \
    if (status == cutlass::Status::kSuccess) {                                                  \
        size_t workspace_size = Gemm::get_workspace_size(args);                                 \
        void *workspace = get_workspace(workspace_size, stream);                                \
        gemm_op.initialize(args, workspace);                                                    \
        status = gemm_op(stream);                                                               \
    }

// Basic template of using cutlass::gemm::device::Gemm. The default alignment is 1
// https://github.com/NVIDIA/cutlass/blob/f4a021660162510572f90ea715b018cff9c0f12f/include/cutlass/gemm/device/default_gemm_configuration.h#L73
#define CUTLASS_GEMM_SIMT_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                      \
                                             LAYOUT_A,                                          \
                                             ELEMENT_TYPE,                                      \
                                             LAYOUT_B,                                          \
                                             ELEMENT_TYPE,                                      \
                                             LAYOUT_C,                                          \
                                             ELEMENT_ACCUMULATOR,                               \
                                             cutlass::arch::OpClassSimt,                        \
                                             cutlass::arch::Sm80,                               \
                                             cutlass::gemm::GemmShape<128, 128, 8>,             \
                                             cutlass::gemm::GemmShape<32, 64, 8>,               \
                                             cutlass::gemm::GemmShape<1, 1, 1>>;                \
    Gemm gemm_op;                                                                               \
    Gemm::Arguments args({M, N, K},                                                             \
                        {(ELEMENT_TYPE *) A, lda},                                              \
                        {(ELEMENT_TYPE *) B, ldb},                                              \
                        {(ELEMENT_TYPE *) C, ldc},                                              \
                        {(ELEMENT_TYPE *) D, ldd},                                              \
                        {alpha, beta});                                                         \
    LAUNCH_GEMM;

// Tensor op gemm template
#define CUTLASS_GEMM_TENSOR_OP_TEMPLATE_BASE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, ALIGNMENT, EPILOGUE_ALIGNMENT)     \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                              \
                                             LAYOUT_A,                                                  \
                                             ELEMENT_TYPE,                                              \
                                             LAYOUT_B,                                                  \
                                             ELEMENT_TYPE,                                              \
                                             LAYOUT_C,                                                  \
                                             ELEMENT_ACCUMULATOR,                                       \
                                             cutlass::arch::OpClassTensorOp,                            \
                                             cutlass::arch::Sm80,                                       \
                                             cutlass::gemm::GemmShape<128, 128, 16>,                    \
                                             cutlass::gemm::GemmShape<64, 64, 16>,                      \
                                             cutlass::gemm::GemmShape<16, 8, 8>,                        \
                                             cutlass::epilogue::thread::LinearCombination<              \
                                                ELEMENT_TYPE,                                           \
                                                EPILOGUE_ALIGNMENT,                 \
                                                ELEMENT_ACCUMULATOR,                                            \
                                                ELEMENT_ACCUMULATOR>,                                           \
                                             cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,      \
                                             4 /* NumStages */,                                                 \
                                             ALIGNMENT,                    \
                                             ALIGNMENT>;                   \
    Gemm gemm_op;                                                                                       \
    Gemm::Arguments args({M, N, K},                                                                     \
                        {(ELEMENT_TYPE *) A, lda},                                                      \
                        {(ELEMENT_TYPE *) B, ldb},                                                      \
                        {(ELEMENT_TYPE *) C, ldc},                                                      \
                        {(ELEMENT_TYPE *) D, ldd},                                                      \
                        {alpha, beta});                                                                 \
    LAUNCH_GEMM;

#define CUTLASS_GEMM_TENSOR_OP_TEMPLATE_DEFAULT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    CUTLASS_GEMM_TENSOR_OP_TEMPLATE_BASE(   \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,    \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,    \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value);

#define CUTLASS_GEMM_TENSOR_OP_TEMPLATE_SMALL_ALIGNMENT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    CUTLASS_GEMM_TENSOR_OP_TEMPLATE_BASE(   \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,    \
        2,    \
        1);

#define CUTLASS_GEMM_FP16_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)     \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                              \
                                             LAYOUT_A,                                                  \
                                             ELEMENT_TYPE,                                              \
                                             LAYOUT_B,                                                  \
                                             ELEMENT_TYPE,                                              \
                                             LAYOUT_C,                                                  \
                                             ELEMENT_ACCUMULATOR,                                       \
                                             cutlass::arch::OpClassTensorOp,                            \
                                             cutlass::arch::Sm80,                                       \
                                             cutlass::gemm::GemmShape<128, 128, 16>,                    \
                                             cutlass::gemm::GemmShape<64, 64, 16>,                      \
                                             cutlass::gemm::GemmShape<16, 8, 8>,                        \
                                             cutlass::epilogue::thread::LinearCombination<              \
                                                ELEMENT_TYPE,                                           \
                                                128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,                 \
                                                ELEMENT_ACCUMULATOR,                                            \
                                                ELEMENT_ACCUMULATOR>,                                           \
                                             cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,      \
                                             4 /* NumStages */,                                                 \
                                             128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,                    \
                                             128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value>;                   \
    Gemm gemm_op;                                                                                       \
    Gemm::Arguments args({M, N, K},                                                                     \
                        {(ELEMENT_TYPE *) A, lda},                                                      \
                        {(ELEMENT_TYPE *) B, ldb},                                                      \
                        {(ELEMENT_TYPE *) C, ldc},                                                      \
                        {(ELEMENT_TYPE *) D, ldd},                                                      \
                        {alpha, beta});                                                                 \
    status = gemm_op.can_implement(args);                                                               \
    if (status == cutlass::Status::kSuccess) {                                                  \
        size_t workspace_size = Gemm::get_workspace_size(args);                                             \
        void *workspace = get_workspace(workspace_size, stream);                                            \
        gemm_op.initialize(args, workspace);                                                                \
        status = gemm_op(stream);                                                                           \
    }

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
    status = gemm_op.can_implement(args);                                                       \
    if (status == cutlass::Status::kSuccess) { \
        size_t workspace_size = Gemm::get_workspace_size(args);                                     \
        void *workspace = get_workspace(workspace_size, stream);                                    \
        gemm_op.initialize(args, workspace);                                                        \
        status = gemm_op(stream);                                                                   \
    }

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
    status = gemm_op.can_implement(args);                                                               \
    if (status == cutlass::Status::kSuccess) { \
        size_t workspace_size = Gemm::get_workspace_size(args);                                             \
        void *workspace = get_workspace(workspace_size, stream);                                            \
        gemm_op.initialize(args, workspace);                                                                \
        status = gemm_op(stream);                                                                           \
    }

#define CUTLASS_GEMM_BATCHED_FP16_TEMPLATE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    using Gemm = cutlass::gemm::device::GemmBatched<ELEMENT_TYPE,                                       \
                                                    LAYOUT_A,                                           \
                                                    ELEMENT_TYPE,                                       \
                                                    LAYOUT_B,                                           \
                                                    ELEMENT_TYPE,                                       \
                                                    LAYOUT_C,                                           \
                                                    ELEMENT_ACCUMULATOR,    \
                                                    cutlass::arch::OpClassTensorOp,                            \
                                                    cutlass::arch::Sm80,                                       \
                                                    cutlass::gemm::GemmShape<128, 128, 16>,                    \
                                                    cutlass::gemm::GemmShape<64, 64, 16>,                      \
                                                    cutlass::gemm::GemmShape<16, 8, 8>>;                               \
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
    status = gemm_op.can_implement(args);                                                               \
    if (status == cutlass::Status::kSuccess) {  \
        size_t workspace_size = Gemm::get_workspace_size(args);                                             \
        void *workspace = get_workspace(workspace_size, stream);                                            \
        gemm_op.initialize(args, workspace);                                                                \
        status = gemm_op(stream);                                                                           \
    }

#define INVOKE_CUTLASS_GEMM_TEMPLATE(TEMPLATE_NAME, ELEMENT_TYPE, ElementAccumulator)                    \
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

#define SET_SPLIT_K_SLICES(MM, NN, KK, SPLICES)         \
    if (M == MM && N == NN && K == KK) {                \
        split_k_slices = SPLICES;                       \
        use_k_split = true;                             \
    }

// Get the workspace with given size
// Will keep one workspace for each stream
void *get_workspace(size_t size, cudaStream_t stream) {

    static std::map<cudaStream_t, std::pair<size_t, void *>> workspace_map;

    if (workspace_map.find(stream) == workspace_map.end()) {
        workspace_map[stream].first = 0;
        workspace_map[stream].second = NULL;
    }

    if (size > workspace_map[stream].first) {
        cudaStreamSynchronize(stream);
        auto old_workspace = workspace_map[stream].second;

        if (old_workspace) {
            cudaFree(old_workspace);
        }

        void *workspace;
        cudaMalloc(&workspace, size);
        workspace_map[stream].first = size;
        workspace_map[stream].second = workspace;
    }

    return workspace_map[stream].second;
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

    bool use_k_split = false;
    int split_k_slices = 0;

    // Hardcode the split-k slices for specific input dimensions
    SET_SPLIT_K_SLICES(1024, 60, 1024, 16);
    SET_SPLIT_K_SLICES(4096, 60, 1024, 4);
    SET_SPLIT_K_SLICES(1024, 60, 4096, 16);
    SET_SPLIT_K_SLICES(1024, 60, 96103, 128);

    if (use_k_split) {
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_SPLIT_K_TEMPLATE, float, float);
    } else {
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_SIMT_TEMPLATE, float, float);
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

    CUTLASS_CHECK(status);
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
    half *bias,
    cudaStream_t stream
) {

    cutlass::Status status;

    // Try to use default tensorop gemm
    INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_TENSOR_OP_TEMPLATE_DEFAULT, cutlass::half_t, float);

    // If fails with kErrorMisalignedOperand, try with a smaller alignment
    if (status == cutlass::Status::kErrorMisalignedOperand) {
        std::cout << "Failing back to small alignment because of kErrorMisalignedOperand" << std::endl;
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_TENSOR_OP_TEMPLATE_SMALL_ALIGNMENT, cutlass::half_t, float);
    }

    // If still fails, try with simt gemm 
    if (status == cutlass::Status::kErrorMisalignedOperand) {
        std::cout << "Failing back to simt op because of kErrorMisalignedOperand" << std::endl;
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_SIMT_TEMPLATE, cutlass::half_t, float);
    }

    CUTLASS_CHECK(status);
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

    INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_BATCHED_TEMPLATE, float, float);

    CUTLASS_CHECK(status);
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

    INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_BATCHED_FP16_TEMPLATE, cutlass::half_t, float);

    // Some sizes do not work in tensor core mode because of misaligned error
    // won't fix at this point
    if (status == cutlass::Status::kErrorMisalignedOperand) {
        INVOKE_CUTLASS_GEMM_TEMPLATE(CUTLASS_GEMM_BATCHED_TEMPLATE, cutlass::half_t, float);
    }

    CUTLASS_CHECK(status);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;

}

}