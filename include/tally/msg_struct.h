#ifndef TALLY_DEF_H
#define TALLY_DEF_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cublasLt.h>
#include <cudnn.h>

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
	void * devPtr;
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

struct cudnnBackendSetAttributeArg {
    cudnnBackendDescriptor_t  descriptor;
    cudnnBackendAttributeName_t  attributeName;
    cudnnBackendAttributeType_t  attributeType;
    int64_t  elementCount;
    char arrayOfElements[];
};

struct cudnnBackendSetAttributeResponse {
    cudnnStatus_t err;
};

struct cudnnBackendGetAttributeArg {
    cudnnBackendDescriptor_t descriptor;
    cudnnBackendAttributeName_t  attributeName;
    cudnnBackendAttributeType_t  attributeType;
    int64_t  requestedElementCount;
    int64_t * elementCount;
    void * arrayOfElements;

    // I found out that the data is needed too
    // So I'll copy the value
    char arrayOfElementsData[];
};

struct cudnnBackendGetAttributeResponse {
    cudnnStatus_t err;
    int64_t elementCount;
    int64_t arrayOfElementsSize;
    char arrayOfElements[];
};

struct cudnnActivationForwardArg {
	cudnnHandle_t  handle;
	cudnnActivationDescriptor_t  activationDesc;
	uint64_t alpha;
	cudnnTensorDescriptor_t  xDesc;
	void *x;
	uint64_t beta;
	cudnnTensorDescriptor_t  yDesc;
	void *y;
};

struct cudnnSetTensorNdDescriptorArg {
    cudnnTensorDescriptor_t  tensorDesc;
    cudnnDataType_t  dataType;
    int  nbDims;
    int  dimA_and_strideA[];
};

struct cudnnSetConvolutionNdDescriptorArg {
    cudnnConvolutionDescriptor_t  convDesc;
    int  arrayLength;
    cudnnConvolutionMode_t  mode;
    cudnnDataType_t  computeType;
    int  padA_and_filterStrideA_and_dilationA[];
};

struct cudnnSetFilterNdDescriptorArg {
    cudnnFilterDescriptor_t  filterDesc;
    cudnnDataType_t  dataType;
    cudnnTensorFormat_t  format;
    int  nbDims;
    int  filterDimA[];
};

struct cudnnConvolutionForwardArg {
    cudnnHandle_t  handle;
    uint64_t alpha;
    cudnnTensorDescriptor_t  xDesc;
    void *x;
    cudnnFilterDescriptor_t  wDesc;
    void *w;
    cudnnConvolutionDescriptor_t  convDesc;
    cudnnConvolutionFwdAlgo_t  algo;
    void *workSpace;
    size_t  workSpaceSizeInBytes;
    uint64_t beta;
    cudnnTensorDescriptor_t  yDesc;
    void *y;
};

struct cudnnGetConvolutionNdForwardOutputDimArg {
    cudnnConvolutionDescriptor_t  convDesc;
    cudnnTensorDescriptor_t  inputTensorDesc;
    cudnnFilterDescriptor_t  filterDesc;
    int  nbDims;
};

struct cudnnGetConvolutionNdForwardOutputDimResponse {
    cudnnStatus_t err;
    int  tensorOuputDimA[];
};

struct cudnnGetConvolutionForwardAlgorithm_v7Arg {
    cudnnHandle_t  handle;
    cudnnTensorDescriptor_t  srcDesc;
    cudnnFilterDescriptor_t  filterDesc;
    cudnnConvolutionDescriptor_t  convDesc;
    cudnnTensorDescriptor_t  destDesc;
    int  requestedAlgoCount;
};

struct cudnnGetConvolutionForwardAlgorithm_v7Response {
    cudnnStatus_t err;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[];
};

struct cudnnFindConvolutionForwardAlgorithmArg {
    cudnnHandle_t  handle;
    cudnnTensorDescriptor_t  xDesc;
    cudnnFilterDescriptor_t  wDesc;
    cudnnConvolutionDescriptor_t  convDesc;
    cudnnTensorDescriptor_t  yDesc;
    int  requestedAlgoCount;
};

struct cudnnFindConvolutionForwardAlgorithmResponse {
    cudnnStatus_t err;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[];
};

struct cudnnAddTensorArg {
    cudnnHandle_t  handle;
    uint64_t alpha;
    cudnnTensorDescriptor_t  aDesc;
    void *A;
    uint64_t beta;
    cudnnTensorDescriptor_t  cDesc;
    void *C;
};

struct cudnnSetPoolingNdDescriptorArg {
    cudnnPoolingDescriptor_t  poolingDesc;
    cudnnPoolingMode_t  mode;
    cudnnNanPropagation_t  maxpoolingNanOpt;
    int  nbDims;
    int  windowDimA_paddingA_strideA[];
};

struct cudnnGetPoolingNdDescriptorArg {
    cudnnPoolingDescriptor_t  poolingDesc;
    int  nbDimsRequested;
};

struct cudnnGetPoolingNdDescriptorResponse {
    cudnnStatus_t err;
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt;
    int nbDims;
    int windowDimA_paddingA_strideA[];
};

struct cudnnGetPoolingNdForwardOutputDimArg {
    cudnnPoolingDescriptor_t  poolingDesc;
    cudnnTensorDescriptor_t  inputTensorDesc;
    int  nbDims;
};

struct cudnnGetPoolingNdForwardOutputDimResponse {
    cudnnStatus_t err;
    int  outputTensorDimA[];
};

struct cudnnPoolingForwardArg {
    cudnnHandle_t  handle;
    cudnnPoolingDescriptor_t  poolingDesc;
    uint64_t alpha;
    cudnnTensorDescriptor_t  xDesc;
    void *x;
    uint64_t beta;
    cudnnTensorDescriptor_t  yDesc;
    void *y;
};

struct cublasSgemv_v2Arg {
    cublasHandle_t  handle;
    cublasOperation_t  trans;
    int  m;
    int  n;
    float alpha;
    float*  A;
    int  lda;
    float*  x;
    int  incx;
    float beta;
    float*  y;
    int  incy;
};

struct cudnnLRNCrossChannelForwardArg {
    cudnnHandle_t  handle;
    cudnnLRNDescriptor_t  normDesc;
    cudnnLRNMode_t  lrnMode;
    uint64_t alpha;
    cudnnTensorDescriptor_t  xDesc;
    void *x;
    uint64_t beta;
    cudnnTensorDescriptor_t  yDesc;
    void *y;
};

struct cudnnSoftmaxForwardArg {
    cudnnHandle_t  handle;
    cudnnSoftmaxAlgorithm_t  algo;
    cudnnSoftmaxMode_t  mode;
    uint64_t alpha;
    cudnnTensorDescriptor_t  xDesc;
    void *x;
    uint64_t beta;
    cudnnTensorDescriptor_t  yDesc;
    void *y;
};

struct cudnnTransformTensorArg {
    cudnnHandle_t  handle;
    uint64_t alpha;
    cudnnTensorDescriptor_t  xDesc;
    void * x;
    uint64_t beta;
    cudnnTensorDescriptor_t  yDesc;
    void * y;
};

struct cublasSgemmExArg {
    cublasHandle_t  handle;
    cublasOperation_t  transa;
    cublasOperation_t  transb;
    int  m;
    int  n;
    int  k;
    float  alpha;
    void*  A;
    cudaDataType  Atype;
    int  lda;
    void*  B;
    cudaDataType  Btype;
    int  ldb;
    float  beta;
    void*  C;
    cudaDataType  Ctype;
    int  ldc;
};

struct cudnnSetSeqDataDescriptorArg {
    cudnnSeqDataDescriptor_t  seqDataDesc;
    cudnnDataType_t  dataType;
    int nbDims = 4;
    int dimA[4];
    cudnnSeqDataAxis_t axes[4];
    size_t seqLengthArraySize;
    void * paddingFill = NULL;
    int  seqLengthArray[];
};

struct cudnnGetSeqDataDescriptorArg {
    cudnnSeqDataDescriptor_t seqDataDesc;
    int nbDimsRequested;
    size_t seqLengthSizeRequested;
    void *paddingFill = NULL;
};

struct cudnnGetSeqDataDescriptorResponse {
    cudnnStatus_t err;
    cudnnDataType_t dataType;
    int nbDims;
    size_t seqLengthArraySize;
    char dimA_axes_seqLengthArray[];
};

struct cudnnMultiHeadAttnForwardArg {
    cudnnHandle_t  handle;
    cudnnAttnDescriptor_t  attnDesc;
    int  currIdx;
    int  *devSeqLengthsQO;
    int  *devSeqLengthsKV;
    cudnnSeqDataDescriptor_t  qDesc;
    void * queries;
    void * residuals;
    cudnnSeqDataDescriptor_t  kDesc;
    void * keys;
    cudnnSeqDataDescriptor_t  vDesc;
    void * values;
    cudnnSeqDataDescriptor_t  oDesc;
    void * out;
    size_t  weightSizeInBytes;
    void * weights;
    size_t  workSpaceSizeInBytes;
    void * workSpace;
    size_t  reserveSpaceSizeInBytes;
    void * reserveSpace;

    // The length of this array is determined by currIdx
    // if currIdx is negative, need to find out seq length from qDesc
    int winIdxLen;
    int  loWinIdx_hiWinIdx[];
};

struct cudnnMultiHeadAttnBackwardDataArg {
    cudnnHandle_t  handle;
    cudnnAttnDescriptor_t  attnDesc;
    int  *devSeqLengthsDQDO;
    int  *devSeqLengthsDKDV;
    cudnnSeqDataDescriptor_t  doDesc;
    void * dout;
    cudnnSeqDataDescriptor_t  dqDesc;
    void * dqueries;
    void * queries;
    cudnnSeqDataDescriptor_t  dkDesc;
    void * dkeys;
    void * keys;
    cudnnSeqDataDescriptor_t  dvDesc;
    void * dvalues;
    void * values;
    size_t  weightSizeInBytes;
    void * weights;
    size_t  workSpaceSizeInBytes;
    void * workSpace;
    size_t  reserveSpaceSizeInBytes;
    void * reserveSpace;

    // The length of this array is determined by currIdx
    // if currIdx is negative, need to find out seq length from qDesc
    int winIdxLen;
    int loWinIdx_hiWinIdx[];
};

struct cudnnMultiHeadAttnBackwardWeightsArg {
	cudnnHandle_t  handle;
	cudnnAttnDescriptor_t  attnDesc;
	cudnnWgradMode_t  addGrad;
	cudnnSeqDataDescriptor_t  qDesc;
	void * queries;
	cudnnSeqDataDescriptor_t  kDesc;
	void * keys;
	cudnnSeqDataDescriptor_t  vDesc;
	void * values;
	cudnnSeqDataDescriptor_t  doDesc;
	void * dout;
	size_t  weightSizeInBytes;
	void * weights;
	void * dweights;
	size_t  workSpaceSizeInBytes;
	void * workSpace;
	size_t  reserveSpaceSizeInBytes;
	void * reserveSpace;
};

struct cudnnReorderFilterAndBiasArg {
	cudnnHandle_t  handle;
	cudnnFilterDescriptor_t  filterDesc;
	cudnnReorderType_t  reorderType;
	void * filterData;
	void * reorderedFilterData;
	int  reorderBias;
	void * biasData;
	void * reorderedBiasData;
};

#endif // TALLY_DEF_H