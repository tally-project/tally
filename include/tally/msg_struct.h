#ifndef TALLY_DEF_H
#define TALLY_DEF_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cublasLt.h>

#include <tally/generated/cuda_api_enum.h>

struct __align__(8) fatBinaryHeader
{
    unsigned int           magic;
    unsigned short         version;
    unsigned short         headerSize;
    unsigned long long int fatSize;
};

struct __cudaRegisterFatBinaryArg {
    bool cached;
    int magic;
    int version;
    char data[];
};

struct registerKernelArg {
    void *host_func;
    uint32_t kernel_func_len; 
    char data[]; // kernel_func_name
};

typedef struct {
    int magic;
    int version;
    const unsigned long long* data;
    void *filename_or_fatbins;  /* version 1: offline filename,
                                * version 2: array of prelinked fatbins */
} my__fatBinC_Wrapper_t;

typedef struct MessageHeader {
    CUDA_API_ENUM api_id;
} MessageHeader_t;

struct cudaMallocArg {
    void ** devPtr;
    size_t  size;
};

struct cudaMallocResponse {
    void *ptr;
    cudaError_t err;
};

struct cudaMemcpyArg {
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    char data[];
};

struct cudaMemcpyAsyncArg {
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    char data[];
};

struct cudaLaunchKernelArg {
    const void *host_func;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    char params[];
};

struct cudaMemcpyResponse {
    cudaError_t err;
    char data[];
};

struct cudaMemcpyAsyncResponse {
    cudaError_t err;
    char data[];
};

struct cublasSgemm_v2Arg {
	cublasHandle_t handle;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m;
	int n;
	int k;
	float alpha;
	const float *A;
	int lda;
	const float *B;
	int ldb;
	float beta;
	float *C;
	int ldc;
};

struct cublasLtMatmulDescSetAttributeArg {
    cublasLtMatmulDesc_t  matmulDesc;
    cublasLtMatmulDescAttributes_t  attr;
    size_t  sizeInBytes;
    char  buf[];
};

struct cublasLtMatrixLayoutSetAttributeArg {
    cublasLtMatrixLayout_t  matLayout;
    cublasLtMatrixLayoutAttribute_t  attr;
    size_t  sizeInBytes;
    char buf[];
};

struct cublasLtMatmulPreferenceSetAttributeArg {
    cublasLtMatmulPreference_t  pref;
    cublasLtMatmulPreferenceAttributes_t  attr;
    size_t  sizeInBytes;
    char buf[];
};

struct cublasLtMatmulAlgoGetHeuristicArg {
    cublasLtHandle_t  lightHandle;
    cublasLtMatmulDesc_t  operationDesc;
    cublasLtMatrixLayout_t  Adesc;
    cublasLtMatrixLayout_t  Bdesc;
    cublasLtMatrixLayout_t  Cdesc;
    cublasLtMatrixLayout_t  Ddesc;
    cublasLtMatmulPreference_t  preference;
    int  requestedAlgoCount;
    // head of the result array
    // server need to keep track of the addresses
    cublasLtMatmulHeuristicResult_t *heuristicResultsArray;
};

struct cublasLtMatmulAlgoGetHeuristicResponse {
    int returnAlgoCount;
    cublasStatus_t err;
    cublasLtMatmulHeuristicResult_t  heuristicResultsArray[];
};

struct cublasLtMatmulArg {
    cublasLtHandle_t  lightHandle;
    cublasLtMatmulDesc_t  computeDesc;
    uint64_t  alpha; // Don't know what type it is, so copy 64 bits
    const void*  A;
    cublasLtMatrixLayout_t  Adesc;
    const void*  B;
    cublasLtMatrixLayout_t  Bdesc;
    uint64_t  beta; // Don't know what type it is, so copy 64 bits
    void*  C;
    cublasLtMatrixLayout_t  Cdesc;
    void*  D;
    cublasLtMatrixLayout_t  Ddesc;
    cublasLtMatmulAlgo_t algo;
    void*  workspace;
    size_t  workspaceSizeInBytes;
    cudaStream_t  stream;
};

struct cudaGetErrorStringArg {
    cudaError_t  error;
};

struct cudaGetErrorStringResponse {
    uint32_t str_len;
    char data[];
};

#endif // TALLY_DEF_H