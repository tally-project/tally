#include <stdio.h>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sys/time.h>

#include <cublas_v2.h>
#include <cublasLt.h>
#include "cuda.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "cutlass/gemm/device/gemm.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

struct RowAddFunctor {
    int rows;
    const float* vec;

    RowAddFunctor(int _rows, const float* _vec) : rows(_rows), vec(_vec) {}

    __host__ __device__
    float operator()(const float& matrix_val, const int& idx) const {
        // Compute the corresponding index in the vector
        int vec_idx = idx % rows;
        return matrix_val + vec[vec_idx];
    }
};

cudaError_t CutlassSgemmNN(
    cublasOperation_t transA,
    cublasOperation_t transB,
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
    float *Bias,
    int ldbias,
    void *workSpace,
    cudaStream_t stream
) {

    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    cutlass::Status status;

    if (transA == CUBLAS_OP_N && transB == CUBLAS_OP_N) {

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

    } else if (transA == CUBLAS_OP_T && transB == CUBLAS_OP_N){
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
    } else if (transA == CUBLAS_OP_N && transB == CUBLAS_OP_T){
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
    } else if (transA == CUBLAS_OP_T && transB == CUBLAS_OP_T){
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

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

void print_tensor(std::string name, float *tensor, int rows, int cols) {
    std::cout << "Tensor " << name << ": (" << rows << ", " << cols << ")" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << tensor[IDX2C(i, j, rows)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    srand (1);

    int m = 3;
    int n = 4;
    int k = 5;

    const float alpha = 1.0f;
    const float beta = 0.5f;

    float *h_A, *h_B, *h_C, *h_d_cublas, *h_d_cutlass;
    float *h_AT, *h_d_ref;
    float *h_bias;

    h_A = (float *) malloc(sizeof(float) * k * m);
    h_AT = (float *) malloc(sizeof(float) * m * k);
    h_B = (float *) malloc(sizeof(float) * k * n);
    h_C = (float *) malloc(sizeof(float) * m * n);
    h_bias = (float *) malloc(sizeof(float) * m);

    h_d_cublas = (float *) malloc(sizeof(float) * m * n);
    h_d_cutlass = (float *) malloc(sizeof(float) * m * n);
    h_d_ref = (float *) malloc(sizeof(float) * m * n);

    // Set values in h_A
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < k; i++) {
            h_A[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // Set values in h_AT
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            h_AT[IDX2C(i, j, m)] = h_A[IDX2C(j, i, k)];
        }
    }

    // Set values in h_B
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            h_B[IDX2C(i, j, k)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // Set values in h_C
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            h_C[IDX2C(i, j, m)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    // Set values in h_bias
    for (int j = 0; j < m; j++) {
        h_bias[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    print_tensor("A", h_A, k, m);
    print_tensor("AT", h_AT, m, k);
    print_tensor("B", h_B, k, n);
    print_tensor("C", h_C, m, n);

    // Compute Matmul
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_d_ref[IDX2C(i, j, m)] = beta * h_C[IDX2C(i, j, m)];
            for( int p = 0; p < k; p++ ) {
                h_d_ref[IDX2C(i, j, m)] += alpha * (h_AT[IDX2C(i, p, m)] * h_B[IDX2C(p, j, k)]);
            }
            h_d_ref[IDX2C(i, j, m)] += h_bias[i];
        }
    }

    // Allocate memory on the device
    float* d_A, *d_B, *d_C, *d_cublas, *d_cutlass, *d_bias;
    cudaMalloc((void**)&d_A, sizeof(float) * k * m);
    cudaMalloc((void**)&d_B, sizeof(float) * k * n);
    cudaMalloc((void**)&d_C, sizeof(float) * m * n);
    cudaMalloc((void**)&d_bias, sizeof(float) * m);
    cudaMalloc((void**)&d_cublas, sizeof(float) * m * n);
    cudaMalloc((void**)&d_cutlass, sizeof(float) * m * n);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * k * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, sizeof(float) * m, cudaMemcpyHostToDevice);

    cublasOperation_t transa = CUBLAS_OP_T; // Transpose for A
    cublasOperation_t transb = CUBLAS_OP_N; // No transpose for B

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;

    // Run cublas impl
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));

    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(cublasLtEpilogue_t));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void *));

    cublasLtMatrixLayout_t  A_layout;
    cublasLtMatrixLayout_t  B_layout;
    cublasLtMatrixLayout_t  C_layout;
    cublasLtMatrixLayout_t  D_layout;

    cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_32F, k, m, k);
    cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_32F, k, n, k);
    cublasLtMatrixLayoutCreate(&C_layout, CUDA_R_32F, m, n, m);
    cublasLtMatrixLayoutCreate(&D_layout, CUDA_R_32F, m, n, m);

    cublasLtMatmul(handle, matmul_desc, &alpha, d_A, A_layout, d_B, B_layout, &beta, d_C, C_layout, d_cublas, D_layout, NULL, NULL, 0, NULL);

    // Run cutlass impl
    CutlassSgemmNN(transa, transb, m, n, k, alpha, d_A, k, d_B, k, beta, d_C, m, d_cutlass, m, d_bias, 0, NULL, NULL);

    thrust::device_ptr<float> d_D_thrust(d_cutlass);

    thrust::transform(thrust::cuda::par.on(NULL),
                      d_D_thrust,
                      d_D_thrust + m * n, 
                      thrust::make_counting_iterator(0), 
                      d_D_thrust, 
                      RowAddFunctor(m, thrust::raw_pointer_cast(d_bias)));

    cudaMemcpy(h_d_cublas, d_cublas, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_cutlass, d_cutlass, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {

            if (abs(h_d_cutlass[IDX2C(i, j, m)] - h_d_cublas[IDX2C(i, j, m)]) > 0.001 ||
                abs(h_d_cutlass[IDX2C(i, j, m)] - h_d_ref[IDX2C(i, j, m)]) > 0.001) {
                std::cout << "Results do not match." << std::endl;
                std::cout << "h_d_cutlass[IDX2C(i, j, m)]: " << h_d_cutlass[IDX2C(i, j, m)] << std::endl;
                std::cout << "h_d_cublas[IDX2C(i, j, m)]: " << h_d_cublas[IDX2C(i, j, m)] << std::endl;
                std::cout << "h_d_ref[IDX2C(i, j, m)]: " << h_d_ref[IDX2C(i, j, m)] << std::endl;
                exit(1);
            }

        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_cublas);
    cudaFree(d_cutlass);

    return 0;
}