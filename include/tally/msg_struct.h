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
    uint32_t cubin_uid;
    char data[];
};

struct __cudaRegisterFunctionArg {
    void *host_func;
    uint32_t kernel_func_len; 
    char data[]; // kernel_func_name
};

struct HandshakeMessgae {
    int32_t client_id;
    int32_t priority;
};

struct HandshakeResponse {
    bool success;
};

typedef struct MessageHeader {
    CUDA_API_ENUM api_id;
    int32_t client_id;
} MessageHeader_t;

struct cudaMallocArg {
	void ** devPtr;
	size_t  size;
};

struct cudaMallocResponse {
	void * devPtr;
	cudaError_t err;
};

struct cudaFreeArg {
	void * devPtr;
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

struct cublasGemmExArg {
    cublasHandle_t  handle;
    cublasOperation_t  transa;
    cublasOperation_t  transb;
    int  m;
    int  n;
    int  k;
    uint64_t  alpha;
    void*  A;
    cudaDataType  Atype;
    int  lda;
    void*  B;
    cudaDataType  Btype;
    int  ldb;
    uint64_t  beta;
    void*  C;
    cudaDataType  Ctype;
    int  ldc;
    cublasComputeType_t  computeType;
    cublasGemmAlgo_t  algo;
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

struct cudnnGetRNNWorkspaceSizeArg {
    cudnnHandle_t  handle;
    cudnnRNNDescriptor_t  rnnDesc;
    int  seqLength;
    cudnnTensorDescriptor_t xDesc[];
};

struct cudnnGetRNNWorkspaceSizeResponse {
    cudnnStatus_t err;
    size_t sizeInBytes;
};

struct cudnnGetRNNTrainingReserveSizeArg {
    cudnnHandle_t  handle;
    cudnnRNNDescriptor_t  rnnDesc;
    int  seqLength;
    cudnnTensorDescriptor_t xDesc[];
};

struct cudnnGetRNNTrainingReserveSizeResponse {
    cudnnStatus_t err;
    size_t sizeInBytes;
};

struct cudnnGetFilterNdDescriptorArg {
    cudnnFilterDescriptor_t  filterDesc;
    int  nbDimsRequested;
};

struct cudnnGetFilterNdDescriptorResponse {
    cudnnStatus_t err;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int nbDims;
    int filterDimA[];
};

struct cudnnRNNForwardTrainingArg {
    cudnnHandle_t  handle;
    cudnnRNNDescriptor_t  rnnDesc;
    int  seqLength;
    void * x;
    cudnnTensorDescriptor_t  hxDesc;
    void * hx;
    cudnnTensorDescriptor_t  cxDesc;
    void * cx;
    cudnnFilterDescriptor_t  wDesc;
    void * w;
    void * y;
    cudnnTensorDescriptor_t  hyDesc;
    void * hy;
    cudnnTensorDescriptor_t  cyDesc;
    void * cy;
    void * workSpace;
    size_t  workSpaceSizeInBytes;
    void * reserveSpace;
    size_t  reserveSpaceSizeInBytes;
    cudnnTensorDescriptor_t xDesc_yDesc[];
};

struct cudnnRNNBackwardDataArg {
    cudnnHandle_t  handle;
    cudnnRNNDescriptor_t  rnnDesc;
    int  seqLength;
    void * y;
    void * dy;
    cudnnTensorDescriptor_t  dhyDesc;
    void * dhy;
    cudnnTensorDescriptor_t  dcyDesc;
    void * dcy;
    cudnnFilterDescriptor_t  wDesc;
    void * w;
    cudnnTensorDescriptor_t  hxDesc;
    void * hx;
    cudnnTensorDescriptor_t  cxDesc;
    void * cx;
    void * dx;
    cudnnTensorDescriptor_t  dhxDesc;
    void * dhx;
    cudnnTensorDescriptor_t  dcxDesc;
    void * dcx;
    void * workSpace;
    size_t  workSpaceSizeInBytes;
    void * reserveSpace;
    size_t  reserveSpaceSizeInBytes;

    cudnnTensorDescriptor_t yDesc_dyDesc_dxDesc[];
};

struct cudnnRNNBackwardWeightsArg {
    cudnnHandle_t  handle;
    cudnnRNNDescriptor_t  rnnDesc;
    int  seqLength;
    void * x;
    cudnnTensorDescriptor_t  hxDesc;
    void * hx;
    void * y;
    void * workSpace;
    size_t  workSpaceSizeInBytes;
    cudnnFilterDescriptor_t  dwDesc;
    void * dw;
    void * reserveSpace;
    size_t  reserveSpaceSizeInBytes;

    cudnnTensorDescriptor_t xDesc_yDesc[];
};

struct cudnnSetRNNDataDescriptorArg {
    cudnnRNNDataDescriptor_t  rnnDataDesc;
    cudnnDataType_t  dataType;
    cudnnRNNDataLayout_t  layout;
    int  maxSeqLength;
    int  batchSize;
    int  vectorSize;
    void * paddingFill;
    uint64_t paddingFillVal;
    int  seqLengthArray[];
};

struct cudnnGetTensorNdDescriptorArg {
    cudnnTensorDescriptor_t  tensorDesc;
    int  nbDimsRequested;
};

struct cudnnGetTensorNdDescriptorResponse {
    cudnnStatus_t err;
    cudnnDataType_t dataType;
    int nbDims;
    int dimA_strideA[];
};

struct cudnnBatchNormalizationForwardTrainingExArg {
	cudnnHandle_t  handle;
	cudnnBatchNormMode_t  mode;
	cudnnBatchNormOps_t  bnOps;
	uint64_t alpha;
	uint64_t beta;
	cudnnTensorDescriptor_t  xDesc;
	void * xData;
	cudnnTensorDescriptor_t  zDesc;
	void * zData;
	cudnnTensorDescriptor_t  yDesc;
	void * yData;
	cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc;
	void * bnScale;
	void * bnBias;
	double  exponentialAverageFactor;
	void * resultRunningMean;
	void * resultRunningVariance;
	double  epsilon;
	void * resultSaveMean;
	void * resultSaveInvVariance;
	cudnnActivationDescriptor_t  activationDesc;
	void * workspace;
	size_t  workSpaceSizeInBytes;
	void * reserveSpace;
	size_t  reserveSpaceSizeInBytes;
};

struct cudnnBatchNormalizationBackwardExArg {
	cudnnHandle_t  handle;
	cudnnBatchNormMode_t  mode;
	cudnnBatchNormOps_t  bnOps;
	uint64_t alphaDataDiff;
	uint64_t betaDataDiff;
	uint64_t alphaParamDiff;
	uint64_t betaParamDiff;
	cudnnTensorDescriptor_t  xDesc;
	void * xData;
	cudnnTensorDescriptor_t  yDesc;
	void * yData;
	cudnnTensorDescriptor_t  dyDesc;
	void * dyData;
	cudnnTensorDescriptor_t  dzDesc;
	void * dzData;
	cudnnTensorDescriptor_t  dxDesc;
	void * dxData;
	cudnnTensorDescriptor_t  dBnScaleBiasDesc;
	void * bnScaleData;
	void * bnBiasData;
	void * dBnScaleData;
	void * dBnBiasData;
	double  epsilon;
	void * savedMean;
	void * savedInvVariance;
	cudnnActivationDescriptor_t  activationDesc;
	void * workSpace;
	size_t  workSpaceSizeInBytes;
	void * reserveSpace;
	size_t  reserveSpaceSizeInBytes;
};

struct cublasSgemmStridedBatchedArg {
	cublasHandle_t  handle;
	cublasOperation_t  transa;
	cublasOperation_t  transb;
	int  m;
	int  n;
	int  k;
	float  alpha;
	float*  A;
	int  lda;
	long long int  strideA;
	float*  B;
	int  ldb;
	long long int  strideB;
	float  beta;
	float*  C;
	int  ldc;
	long long int  strideC;
	int  batchCount;
};

struct cudaFuncGetAttributesArg {
	struct cudaFuncAttributes * attr;
	void * func;
};

struct cudaFuncGetAttributesResponse {
	struct cudaFuncAttributes  attr;
	cudaError_t err;
};

struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsArg {
	int * numBlocks;
	void * func;
	int  blockSize;
	size_t  dynamicSMemSize;
	unsigned int  flags;
};

struct cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsResponse {
	int  numBlocks;
	cudaError_t err;
};

struct cudaChooseDeviceArg {
    struct cudaDeviceProp prop;
};

struct cudaChooseDeviceResponse {
    cudaError_t err;
    int device;
};

struct cudaSetDeviceArg {
	int  device;
};

struct cudnnRNNBackwardWeights_v8Arg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnWgradMode_t  addGrad;
	int32_t * devSeqLengths;
	cudnnRNNDataDescriptor_t  xDesc;
	void * x;
	cudnnTensorDescriptor_t  hDesc;
	void * hx;
	cudnnRNNDataDescriptor_t  yDesc;
	void * y;
	size_t  weightSpaceSize;
	void * dweightSpace;
	size_t  workSpaceSize;
	void * workSpace;
	size_t  reserveSpaceSize;
	void * reserveSpace;
};

struct cudnnRNNBackwardData_v8Arg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int32_t * devSeqLengths;
	cudnnRNNDataDescriptor_t  yDesc;
	void * y;
	void * dy;
	cudnnRNNDataDescriptor_t  xDesc;
	void * dx;
	cudnnTensorDescriptor_t  hDesc;
	void * hx;
	void * dhy;
	void * dhx;
	cudnnTensorDescriptor_t  cDesc;
	void * cx;
	void * dcy;
	void * dcx;
	size_t  weightSpaceSize;
	void * weightSpace;
	size_t  workSpaceSize;
	void * workSpace;
	size_t  reserveSpaceSize;
	void * reserveSpace;
};

struct cudnnRNNForwardArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnForwardMode_t  fwdMode;
	int32_t * devSeqLengths;
	cudnnRNNDataDescriptor_t  xDesc;
	void * x;
	cudnnRNNDataDescriptor_t  yDesc;
	void * y;
	cudnnTensorDescriptor_t  hDesc;
	void * hx;
	void * hy;
	cudnnTensorDescriptor_t  cDesc;
	void * cx;
	void * cy;
	size_t  weightSpaceSize;
	void * weightSpace;
	size_t  workSpaceSize;
	void * workSpace;
	size_t  reserveSpaceSize;
	void * reserveSpace;
};

struct cudnnBackendExecuteArg {
	cudnnHandle_t  handle;
	cudnnBackendDescriptor_t  executionPlan;
	cudnnBackendDescriptor_t  variantPack;
};

struct cudaEventRecordArg {
	cudaEvent_t  event;
	cudaStream_t  stream;
};

struct cudaStreamSynchronizeArg {
	cudaStream_t  stream;
};

struct cublasCreate_v2Arg {
	cublasHandle_t*  handle;
};

struct cublasCreate_v2Response {
	cublasHandle_t handle;
	cublasStatus_t err;
};

struct cudnnCreateArg {
	cudnnHandle_t * handle;
};

struct cudnnCreateResponse {
	cudnnHandle_t  handle;
	cudnnStatus_t err;
};

struct cuModuleLoadDataArg {
    bool cached;
    uint32_t cubin_uid;
    char image[];
};

struct cuModuleLoadDataResponse {
    CUresult err;
    CUmodule module;
    char tmp_elf_file[];
};

struct cuModuleGetFunctionArg {
    CUmodule  hmod;
    char name[];
};

struct cuModuleGetFunctionResponse {
    CUresult err;
    CUfunction hfunc;
};

struct cuPointerGetAttributeArg {
    CUpointer_attribute  attribute;
    CUdeviceptr  ptr;
};

struct cuPointerGetAttributeResponse {
    CUresult err;
    char data[];
};

struct cudaStreamGetCaptureInfo_v2Arg {
	cudaStream_t  stream;
	enum cudaStreamCaptureStatus * captureStatus_out;
	unsigned long long * id_out;
	cudaGraph_t * graph_out;
	cudaGraphNode_t ** dependencies_out;
	size_t * numDependencies_out;
};

struct cudaStreamGetCaptureInfo_v2Response {
	enum cudaStreamCaptureStatus  captureStatus_out;
	unsigned long long  id_out;
	cudaGraph_t  graph_out;
	cudaGraphNode_t * dependencies_out;
	size_t  numDependencies_out;
	cudaError_t err;
};

struct cudaGraphGetNodesArg {
    cudaGraph_t  graph;
    cudaGraphNode_t *nodes;
    size_t numNodes;
};

struct cudaGraphGetNodesResponse {
    cudaError_t err;
    size_t numNodes;
    cudaGraphNode_t nodes[];
};

struct cuLaunchKernelArg {
    CUfunction  f;
    unsigned int  gridDimX;
    unsigned int  gridDimY;
    unsigned int  gridDimZ;
    unsigned int  blockDimX;
    unsigned int  blockDimY;
    unsigned int  blockDimZ;
    unsigned int  sharedMemBytes;
    CUstream  hStream;
    char kernelParams[];
};

struct cudaFuncSetAttributeArg {
	void * func;
	enum cudaFuncAttribute  attr;
	int  value;
};

struct cuMemcpyArg {
	CUdeviceptr  dst;
	CUdeviceptr  src;
	size_t  ByteCount;
    char data[];
};

struct cuMemcpyResponse {
    CUresult err;
    char data[];
};

struct cuMemcpyAsyncArg {
	CUdeviceptr  dst;
	CUdeviceptr  src;
	size_t  ByteCount;
	CUstream  hStream;
    char data[];
};

struct cuMemcpyAsyncResponse {
    CUresult err;
    char data[];
};

struct cuMemAllocAsyncArg {
	CUdeviceptr * dptr;
	size_t  bytesize;
	CUstream  hStream;
};

struct cuMemAllocAsyncResponse {
	CUdeviceptr  dptr;
	CUresult err;
};

struct cuMemFree_v2Arg {
	CUdeviceptr  dptr;
};

struct cudaMemsetArg {
	void * devPtr;
	int  value;
	size_t  count;
};

struct cudaStreamCreateArg {
	cudaStream_t * pStream;
};

struct cudaStreamCreateResponse {
	cudaStream_t  pStream;
	cudaError_t err;
};

struct cudaStreamCreateWithFlagsArg {
	cudaStream_t * pStream;
	unsigned int  flags;
};

struct cudaStreamCreateWithFlagsResponse {
	cudaStream_t  pStream;
	cudaError_t err;
};

struct cudaStreamCreateWithPriorityArg {
	cudaStream_t * pStream;
	unsigned int  flags;
	int  priority;
};

struct cudaStreamCreateWithPriorityResponse {
	cudaStream_t  pStream;
	cudaError_t err;
};

struct cudaStreamBeginCaptureArg {
	cudaStream_t  stream;
	enum cudaStreamCaptureMode  mode;
};

struct cuStreamCreateWithPriorityArg {
	CUstream * phStream;
	unsigned int  flags;
	int  priority;
};

struct cuStreamCreateWithPriorityResponse {
	CUstream  phStream;
	CUresult err;
};

struct cuFuncGetAttributeArg {
	int * pi;
	CUfunction_attribute  attrib;
	CUfunction  hfunc;
};

struct cuFuncGetAttributeResponse {
	int  pi;
	CUresult err;
};

struct cuFuncSetAttributeArg {
	CUfunction  hfunc;
	CUfunction_attribute  attrib;
	int  value;
};

struct cuFuncSetCacheConfigArg {
	CUfunction  hfunc;
	CUfunc_cache  config;
};

struct cublasGemmStridedBatchedExArg {
    cublasHandle_t  handle;
    cublasOperation_t  transa;
    cublasOperation_t  transb;
    int  m;
    int  n;
    int  k;
    uint64_t alpha;
    void*  A;
    cudaDataType  Atype;
    int  lda;
    long long int  strideA;
    void*  B;
    cudaDataType  Btype;
    int  ldb;
    long long int  strideB;
    uint64_t  beta;
    void*  C;
    cudaDataType  Ctype;
    int  ldc;
    long long int  strideC;
    int  batchCount;
    cublasComputeType_t  computeType;
    cublasGemmAlgo_t  algo;
};

struct cuMemsetD8_v2Arg {
	CUdeviceptr  dstDevice;
	unsigned char  uc;
	size_t  N;
};

struct cuStreamCreateArg {
	CUstream * phStream;
	unsigned int  Flags;
};

struct cuStreamCreateResponse {
	CUstream  phStream;
	CUresult err;
};

struct cuMemAlloc_v2Arg {
	CUdeviceptr * dptr;
	size_t  bytesize;
};

struct cuMemAlloc_v2Response {
	CUdeviceptr  dptr;
	CUresult err;
};

struct cuMemsetD32_v2Arg {
	CUdeviceptr  dstDevice;
	unsigned int  ui;
	size_t  N;
};

#endif // TALLY_DEF_H