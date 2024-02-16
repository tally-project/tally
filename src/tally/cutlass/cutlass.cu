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
    if (status == cutlass::Status::kSuccess) {                                                  \
        size_t workspace_size = Gemm::get_workspace_size(args);                                 \
        void *workspace = get_workspace(workspace_size, stream);                                \
        gemm_op.initialize(args, workspace);                                                    \
        status = gemm_op(stream);                                                               \
    }

// Basic template of using cutlass::gemm::device::Gemm. The default alignment is 1
// https://github.com/NVIDIA/cutlass/blob/f4a021660162510572f90ea715b018cff9c0f12f/include/cutlass/gemm/device/default_gemm_configuration.h#L73
#define CUTLASS_GEMM_SIMT_BASE(GEMM_TYPE, ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, SPLIT_K_SLICES)  \
    using Gemm = cutlass::gemm::device::GEMM_TYPE<ELEMENT_TYPE,                                                             \
                                             LAYOUT_A,                                                                      \
                                             ELEMENT_TYPE,                                                                  \
                                             LAYOUT_B,                                                                      \
                                             ELEMENT_TYPE,                                                                  \
                                             LAYOUT_C,                                                                      \
                                             ELEMENT_ACCUMULATOR,                                                           \
                                             cutlass::arch::OpClassSimt,                                                    \
                                             cutlass::arch::Sm80,                                                           \
                                             cutlass::gemm::GemmShape<128, 128, 8>,                                         \
                                             cutlass::gemm::GemmShape<32, 64, 8>,                                           \
                                             cutlass::gemm::GemmShape<1, 1, 1>>;                                            \
    Gemm gemm_op;                                                                                                           \
    Gemm::Arguments args({M, N, K},                                                                                         \
                        {(ELEMENT_TYPE *) A, lda},                                                                          \
                        {(ELEMENT_TYPE *) B, ldb},                                                                          \
                        {(ELEMENT_TYPE *) C, ldc},                                                                          \
                        {(ELEMENT_TYPE *) D, ldd},                                                                          \
                        {alpha, beta},                                                                                      \
                        SPLIT_K_SLICES);                                                                                    \
    status = gemm_op.can_implement(args);

#define CUTLASS_GEMM_SIMT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    CUTLASS_GEMM_SIMT_BASE(Gemm, ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, 1)

#define CUTLASS_GEMM_SIMT_SPLIT_K(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    CUTLASS_GEMM_SIMT_BASE(GemmSplitKParallel, ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, split_k_slices)

// Tensor op gemm template
#define CUTLASS_GEMM_TENSOR_OP_BASE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, ALIGNMENT, EPILOGUE_ALIGNMENT)     \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                                                              \
                                             LAYOUT_A,                                                                                  \
                                             ELEMENT_TYPE,                                                                              \
                                             LAYOUT_B,                                                                                  \
                                             ELEMENT_TYPE,                                                                              \
                                             LAYOUT_C,                                                                                  \
                                             ELEMENT_ACCUMULATOR,                                                                       \
                                             cutlass::arch::OpClassTensorOp,                                                            \
                                             cutlass::arch::Sm80,                                                                       \
                                             cutlass::gemm::GemmShape<128, 128, 16>,                                                    \
                                             cutlass::gemm::GemmShape<64, 64, 16>,                                                      \
                                             cutlass::gemm::GemmShape<16, 8, 8>,                                                        \
                                             cutlass::epilogue::thread::LinearCombination<                                              \
                                                ELEMENT_TYPE,                                                                           \
                                                EPILOGUE_ALIGNMENT,                                                                     \
                                                ELEMENT_ACCUMULATOR,                                                                    \
                                                ELEMENT_ACCUMULATOR>,                                                                   \
                                             cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                              \
                                             4 /* NumStages */,                                                                         \
                                             ALIGNMENT,                                                                                 \
                                             ALIGNMENT>;                                                                                \
    Gemm gemm_op;                                                                                                                       \
    Gemm::Arguments args({M, N, K},                                                                                                     \
                        {(ELEMENT_TYPE *) A, lda},                                                                                      \
                        {(ELEMENT_TYPE *) B, ldb},                                                                                      \
                        {(ELEMENT_TYPE *) C, ldc},                                                                                      \
                        {(ELEMENT_TYPE *) D, ldd},                                                                                      \
                        {alpha, beta});                                                                                                 \
    status = gemm_op.can_implement(args);

#define CUTLASS_GEMM_TENSOR_OP(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    CUTLASS_GEMM_TENSOR_OP_BASE(                                                                \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                        \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,                                        \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value);

#define CUTLASS_GEMM_TENSOR_OP_SMALL_ALIGNMENT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    CUTLASS_GEMM_TENSOR_OP_BASE(                                                                                \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                        \
        2,                                                                                                      \
        1);

// Tensor op gemm split k template
#define CUTLASS_GEMM_TENSOR_OP_SPLIT_K_BASE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, ALIGNMENT, EPILOGUE_ALIGNMENT) \
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ELEMENT_TYPE,                                                     \
                                                                          EPILOGUE_ALIGNMENT,                                               \
                                                                          ELEMENT_ACCUMULATOR,                                              \
                                                                          ELEMENT_ACCUMULATOR>;                                             \
    using Gemm = cutlass::gemm::device::GemmSplitKParallel<ELEMENT_TYPE,                                                                    \
                                                           LAYOUT_A,                                                                        \
                                                           ELEMENT_TYPE,                                                                    \
                                                           LAYOUT_B,                                                                        \
                                                           ELEMENT_TYPE,                                                                    \
                                                           LAYOUT_C,                                                                        \
                                                           ELEMENT_ACCUMULATOR,                                                             \
                                                           cutlass::arch::OpClassTensorOp,                                                  \
                                                           cutlass::arch::Sm80,                                                             \
                                                           cutlass::gemm::GemmShape<128, 128, 16>,                                          \
                                                           cutlass::gemm::GemmShape<64, 64, 16>,                                            \
                                                           cutlass::gemm::GemmShape<16, 8, 8>,                                              \
                                                           EpilogueOutputOp,                                                                \
                                                           cutlass::epilogue::thread::Convert<                                              \
                                                               ELEMENT_ACCUMULATOR,                                                         \
                                                               EpilogueOutputOp::kCount,                                                    \
                                                               ELEMENT_ACCUMULATOR>,                                                        \
                                                           cutlass::reduction::thread::ReduceAdd<                                           \
                                                               ELEMENT_ACCUMULATOR,                                                         \
                                                               EpilogueOutputOp::ElementAccumulator,                                        \
                                                               EpilogueOutputOp::kCount>,                                                   \
                                                           cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                    \
                                                           4 /* NumStages */,                                                               \
                                                           ALIGNMENT,                                                                       \
                                                           ALIGNMENT                                                                        \
                                                    >;                                                                                      \
    Gemm gemm_op;                                                                                                                           \
    Gemm::Arguments args({M, N, K},                                                                                                         \
                        {(ELEMENT_TYPE *) A, lda},                                                                                          \
                        {(ELEMENT_TYPE *) B, ldb},                                                                                          \
                        {(ELEMENT_TYPE *) C, ldc},                                                                                          \
                        {(ELEMENT_TYPE *) D, ldd},                                                                                          \
                        {alpha, beta},                                                                                                      \
                        split_k_slices);                                                                                                    \
    status = gemm_op.can_implement(args);

#define CUTLASS_GEMM_TENSOR_OP_SPLIT_K(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    CUTLASS_GEMM_TENSOR_OP_SPLIT_K_BASE(                                                                \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,                                                \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value);

#define CUTLASS_GEMM_TENSOR_OP_SPLIT_K_SMALL_ALIGNMENT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    CUTLASS_GEMM_TENSOR_OP_SPLIT_K_BASE(                                                                                \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                                \
        2,    \
        1);

// Tensor op gemm split k template
#define CUTLASS_GEMM_TENSOR_OP_TURING_SPLIT_K_BASE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, ALIGNMENT, EPILOGUE_ALIGNMENT)  \
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ELEMENT_TYPE,                                                             \
                                                                          EPILOGUE_ALIGNMENT,                                                       \
                                                                          ELEMENT_ACCUMULATOR,                                                      \
                                                                          ELEMENT_ACCUMULATOR>;                                                     \
    using Gemm = cutlass::gemm::device::GemmSplitKParallel<ELEMENT_TYPE,                                                                            \
                                                           LAYOUT_A,                                                                                \
                                                           ELEMENT_TYPE,                                                                            \
                                                           LAYOUT_B,                                                                                \
                                                           ELEMENT_TYPE,                                                                            \
                                                           LAYOUT_C,                                                                                \
                                                           ELEMENT_ACCUMULATOR,                                                                     \
                                                           cutlass::arch::OpClassTensorOp,                                                          \
                                                           cutlass::arch::Sm75,                                                                     \
                                                           cutlass::gemm::GemmShape<128, 256, 32>,                                                  \
                                                           cutlass::gemm::GemmShape<64, 64, 32>,                                                    \
                                                           cutlass::gemm::GemmShape<16, 8, 8>,                                                      \
                                                           EpilogueOutputOp,                                                                        \
                                                           cutlass::epilogue::thread::Convert<                                                      \
                                                               ELEMENT_ACCUMULATOR,                                                                 \
                                                               EpilogueOutputOp::kCount,                                                            \
                                                               ELEMENT_ACCUMULATOR>,                                                                \
                                                           cutlass::reduction::thread::ReduceAdd<                                                   \
                                                               ELEMENT_ACCUMULATOR,                                                                 \
                                                               EpilogueOutputOp::ElementAccumulator,                                                \
                                                               EpilogueOutputOp::kCount>,                                                           \
                                                           cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,                            \
                                                           2 /* NumStages */,                                                                       \
                                                           ALIGNMENT,                                                                               \
                                                           ALIGNMENT                                                                                \
                                                    >;                                                                                              \
    Gemm gemm_op;                                                                                                                                   \
    Gemm::Arguments args({M, N, K},                                                                                                                 \
                        {(ELEMENT_TYPE *) A, lda},                                                                                                  \
                        {(ELEMENT_TYPE *) B, ldb},                                                                                                  \
                        {(ELEMENT_TYPE *) C, ldc},                                                                                                  \
                        {(ELEMENT_TYPE *) D, ldd},                                                                                                  \
                        {alpha, beta},                                                                                                              \
                        split_k_slices);                                                                                                            \
    status = gemm_op.can_implement(args);

#define CUTLASS_GEMM_TENSOR_OP_TURING_SPLIT_K(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    CUTLASS_GEMM_TENSOR_OP_TURING_SPLIT_K_BASE(                                                                 \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                        \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,                                                        \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value);

#define CUTLASS_GEMM_TENSOR_OP_TURING_SPLIT_K_SMALL_ALIGNMENT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    CUTLASS_GEMM_TENSOR_OP_TURING_SPLIT_K_BASE(                                                                                 \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                                        \
        1,    \
        1);

// Tensor op gemm turing template
#define CUTLASS_GEMM_TENSOR_OP_TURING_BASE(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR, ALIGNMENT, EPILOGUE_ALIGNMENT)     \
    using Gemm = cutlass::gemm::device::Gemm<ELEMENT_TYPE,                                              \
                                             LAYOUT_A,                                                  \
                                             ELEMENT_TYPE,                                              \
                                             LAYOUT_B,                                                  \
                                             ELEMENT_TYPE,                                              \
                                             LAYOUT_C,                                                  \
                                             ELEMENT_ACCUMULATOR,                                       \
                                             cutlass::arch::OpClassTensorOp,                            \
                                             cutlass::arch::Sm75,                                       \
                                             cutlass::gemm::GemmShape<128, 256, 32>,                    \
                                             cutlass::gemm::GemmShape<64, 64, 32>,                      \
                                             cutlass::gemm::GemmShape<16, 8, 8>,                        \
                                             cutlass::epilogue::thread::LinearCombination<              \
                                                ELEMENT_TYPE,                                           \
                                                EPILOGUE_ALIGNMENT,                                             \
                                                ELEMENT_ACCUMULATOR,                                            \
                                                ELEMENT_ACCUMULATOR>,                                           \
                                             cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,      \
                                             2 /* NumStages */,                                                 \
                                             ALIGNMENT,                                                         \
                                             ALIGNMENT>;                                                        \
    Gemm gemm_op;                                                                                       \
    Gemm::Arguments args({M, N, K},                                                                     \
                        {(ELEMENT_TYPE *) A, lda},                                                      \
                        {(ELEMENT_TYPE *) B, ldb},                                                      \
                        {(ELEMENT_TYPE *) C, ldc},                                                      \
                        {(ELEMENT_TYPE *) D, ldd},                                                      \
                        {alpha, beta});                                                                 \
    status = gemm_op.can_implement(args);

#define CUTLASS_GEMM_TENSOR_OP_TURING(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    CUTLASS_GEMM_TENSOR_OP_TURING_BASE(                                                                 \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value,                                                \
        128 / cutlass::sizeof_bits<ELEMENT_TYPE>::value);

#define CUTLASS_GEMM_TENSOR_OP_TURING_SMALL_ALIGNMENT(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)  \
    CUTLASS_GEMM_TENSOR_OP_TURING_BASE(                                                                                 \
        ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR,                                                \
        1,                                                                                                              \
        1);

#define CUTLASS_GEMM_SIMT_BATCHED(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR)      \
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
    status = gemm_op.can_implement(args);

#define CUTLASS_GEMM_TENSOR_OP_BATCHED(ELEMENT_TYPE, LAYOUT_A, LAYOUT_B, LAYOUT_C, ELEMENT_ACCUMULATOR) \
    using Gemm = cutlass::gemm::device::GemmBatched<ELEMENT_TYPE,                                       \
                                                    LAYOUT_A,                                           \
                                                    ELEMENT_TYPE,                                       \
                                                    LAYOUT_B,                                           \
                                                    ELEMENT_TYPE,                                       \
                                                    LAYOUT_C,                                           \
                                                    ELEMENT_ACCUMULATOR,                                \
                                                    cutlass::arch::OpClassTensorOp,                     \
                                                    cutlass::arch::Sm80,                                \
                                                    cutlass::gemm::GemmShape<128, 128, 16>,             \
                                                    cutlass::gemm::GemmShape<64, 64, 16>,               \
                                                    cutlass::gemm::GemmShape<16, 8, 8>>;                \
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
    status = gemm_op.can_implement(args);

#define INVOKE_CUTLASS_GEMM(TEMPLATE_NAME, ELEMENT_TYPE, ElementAccumulator)                                    \
    if (transA == cutlassOperation_t::CUTLASS_OP_N && transB == cutlassOperation_t::CUTLASS_OP_N) {             \
        TEMPLATE_NAME(ELEMENT_TYPE, ColumnMajor, ColumnMajor, ColumnMajor, ElementAccumulator);                 \
        LAUNCH_GEMM;                                                                                            \
    } else if (transA == cutlassOperation_t::CUTLASS_OP_T && transB == cutlassOperation_t::CUTLASS_OP_N) {      \
        TEMPLATE_NAME(ELEMENT_TYPE, RowMajor, ColumnMajor, ColumnMajor, ElementAccumulator);                    \
        LAUNCH_GEMM;                                                                                            \
    } else if (transA == cutlassOperation_t::CUTLASS_OP_N && transB == cutlassOperation_t::CUTLASS_OP_T) {      \
        TEMPLATE_NAME(ELEMENT_TYPE, ColumnMajor, RowMajor, ColumnMajor, ElementAccumulator);                    \
        LAUNCH_GEMM;                                                                                            \
    } else if (transA == cutlassOperation_t::CUTLASS_OP_T && transB == cutlassOperation_t::CUTLASS_OP_T)  {     \
        TEMPLATE_NAME(ELEMENT_TYPE, RowMajor, RowMajor, ColumnMajor, ElementAccumulator);                       \
        LAUNCH_GEMM;                                                                                            \
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
        auto err = cudaMalloc(&workspace, size);
        if (err) {
            throw std::runtime_error("Fail to allocate memory for workspace");
        }
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

    // Let's set up some heuristic to decide if to use split k
    int m_tiles = (M + 128 - 1) / 128;
    int n_tiles = (N + 128 - 1) / 128;

    if (m_tiles * n_tiles < 60 && K >= 128) {
        use_k_split = true;
        split_k_slices = (K + 128 - 1) / 128;
    }

    // Hardcode the split-k slices for specific input dimensions
    // SET_SPLIT_K_SLICES(1024, 60, 1024, 16);
    // SET_SPLIT_K_SLICES(4096, 60, 1024, 4);
    // SET_SPLIT_K_SLICES(1024, 60, 4096, 16);
    // SET_SPLIT_K_SLICES(1024, 60, 96103, 128);

    if (use_k_split) {
        INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_SIMT_SPLIT_K, float, float);
    } else {
        INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_SIMT, float, float);
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

    bool use_k_split = false;
    int split_k_slices = 0;

    // Let's set up some heuristic to decide if to use split k
    int m_tiles = (M + 128 - 1) / 128;
    int n_tiles = (N + 128 - 1) / 128;

    if (m_tiles * n_tiles < 60 && K >= 128) {
        use_k_split = true;
        split_k_slices = (K + 128 - 1) / 128;
    }

    // Hardcode split-k slices for specific input dimensions
    // SET_SPLIT_K_SLICES(2704, 64, 576, 4);
    // SET_SPLIT_K_SLICES(676, 128, 576, 2);
    // SET_SPLIT_K_SLICES(676, 128, 1152, 4);
    // SET_SPLIT_K_SLICES(169, 256, 1152, 8);
    // SET_SPLIT_K_SLICES(169, 256, 128, 2);
    // SET_SPLIT_K_SLICES(169, 256, 2304, 16);

    if (use_k_split) {

        // Use this to detect whether tensorop split k can be used
        CUTLASS_GEMM_TENSOR_OP(cutlass::half_t, ColumnMajor, ColumnMajor, ColumnMajor, float);

        // Try to use tensorop split k if possible
        if (status == cutlass::Status::kSuccess) {
            INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_TENSOR_OP_SPLIT_K, cutlass::half_t, float);
        } else {

            // Use this to detect whether tensorop small alignment split k can be used
            CUTLASS_GEMM_TENSOR_OP_SMALL_ALIGNMENT(cutlass::half_t, ColumnMajor, ColumnMajor, ColumnMajor, float);

            if (status == cutlass::Status::kSuccess) {
                INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_TENSOR_OP_SPLIT_K_SMALL_ALIGNMENT, cutlass::half_t, float);

            // lastly, fall back to simt split k
            } else {
                INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_SIMT_SPLIT_K, cutlass::half_t, float);
            }
        }
        
    } else {

        // Try to use default tensorop gemm
        INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_TENSOR_OP, cutlass::half_t, float);

        // If fails with kErrorMisalignedOperand, try with a smaller alignment
        if (status == cutlass::Status::kErrorMisalignedOperand) {
            // std::cout << "fails with kErrorMisalignedOperand, try with a smaller alignment" << std::endl;
            INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_TENSOR_OP_SMALL_ALIGNMENT, cutlass::half_t, float);
        }

        // If still fails with kErrorMisalignedOperand, try turing tensorop smaller alignment
        if (status == cutlass::Status::kErrorMisalignedOperand) {
            // std::cout << "still fails with kErrorMisalignedOperand, try with turing" << std::endl;
            INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_TENSOR_OP_TURING_SMALL_ALIGNMENT, cutlass::half_t, float);
        }

        // If still fails, try with simt gemm 
        if (status == cutlass::Status::kErrorMisalignedOperand) {
            // std::cout << "still fails, try with simt gemm " << std::endl;
            INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_SIMT, cutlass::half_t, float);
        }
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

    INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_SIMT_BATCHED, float, float);

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

    INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_TENSOR_OP_BATCHED, cutlass::half_t, float);

    if (status == cutlass::Status::kErrorMisalignedOperand) {
        INVOKE_CUTLASS_GEMM(CUTLASS_GEMM_SIMT_BATCHED, cutlass::half_t, float);
    }

    CUTLASS_CHECK(status);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;

}

}