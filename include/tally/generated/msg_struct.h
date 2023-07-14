
#ifndef TALLY_GENERATED_MSG_STRUCT_H
#define TALLY_GENERATED_MSG_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <nvrtc.h>
#include <cublasLt.h>


struct cuInitArg {
	unsigned int  Flags;
};

struct cuDriverGetVersionArg {
	int * driverVersion;
};

struct cuDriverGetVersionResponse {
	int  driverVersion;
	CUresult err;
};

struct cuDeviceGetArg {
	CUdevice * device;
	int  ordinal;
};

struct cuDeviceGetResponse {
	CUdevice  device;
	CUresult err;
};

struct cuDeviceGetCountArg {
	int * count;
};

struct cuDeviceGetCountResponse {
	int  count;
	CUresult err;
};

struct cuDeviceGetUuidArg {
	CUuuid * uuid;
	CUdevice  dev;
};

struct cuDeviceGetUuidResponse {
	CUuuid  uuid;
	CUresult err;
};

struct cuDeviceGetUuid_v2Arg {
	CUuuid * uuid;
	CUdevice  dev;
};

struct cuDeviceGetUuid_v2Response {
	CUuuid  uuid;
	CUresult err;
};

struct cuDeviceGetLuidArg {
	char * luid;
	unsigned int * deviceNodeMask;
	CUdevice  dev;
};

struct cuDeviceGetLuidResponse {
	char  luid;
	unsigned int  deviceNodeMask;
	CUresult err;
};

struct cuDeviceTotalMem_v2Arg {
	size_t * bytes;
	CUdevice  dev;
};

struct cuDeviceTotalMem_v2Response {
	size_t  bytes;
	CUresult err;
};

struct cuDeviceGetTexture1DLinearMaxWidthArg {
	size_t * maxWidthInElements;
	CUarray_format  format;
	unsigned  numChannels;
	CUdevice  dev;
};

struct cuDeviceGetTexture1DLinearMaxWidthResponse {
	size_t  maxWidthInElements;
	CUresult err;
};

struct cuDeviceGetAttributeArg {
	int * pi;
	CUdevice_attribute  attrib;
	CUdevice  dev;
};

struct cuDeviceGetAttributeResponse {
	int  pi;
	CUresult err;
};

struct cuDeviceSetMemPoolArg {
	CUdevice  dev;
	CUmemoryPool  pool;
};

struct cuDeviceGetMemPoolArg {
	CUmemoryPool * pool;
	CUdevice  dev;
};

struct cuDeviceGetMemPoolResponse {
	CUmemoryPool  pool;
	CUresult err;
};

struct cuDeviceGetDefaultMemPoolArg {
	CUmemoryPool * pool_out;
	CUdevice  dev;
};

struct cuDeviceGetDefaultMemPoolResponse {
	CUmemoryPool  pool_out;
	CUresult err;
};

struct cuFlushGPUDirectRDMAWritesArg {
	CUflushGPUDirectRDMAWritesTarget  target;
	CUflushGPUDirectRDMAWritesScope  scope;
};

struct cuDeviceGetPropertiesArg {
	CUdevprop * prop;
	CUdevice  dev;
};

struct cuDeviceGetPropertiesResponse {
	CUdevprop  prop;
	CUresult err;
};

struct cuDeviceComputeCapabilityArg {
	int * major;
	int * minor;
	CUdevice  dev;
};

struct cuDeviceComputeCapabilityResponse {
	int  major;
	int  minor;
	CUresult err;
};

struct cuDevicePrimaryCtxRetainArg {
	CUcontext * pctx;
	CUdevice  dev;
};

struct cuDevicePrimaryCtxRetainResponse {
	CUcontext  pctx;
	CUresult err;
};

struct cuDevicePrimaryCtxRelease_v2Arg {
	CUdevice  dev;
};

struct cuDevicePrimaryCtxSetFlags_v2Arg {
	CUdevice  dev;
	unsigned int  flags;
};

struct cuDevicePrimaryCtxGetStateArg {
	CUdevice  dev;
	unsigned int * flags;
	int * active;
};

struct cuDevicePrimaryCtxGetStateResponse {
	unsigned int  flags;
	int  active;
	CUresult err;
};

struct cuDevicePrimaryCtxReset_v2Arg {
	CUdevice  dev;
};

struct cuDeviceGetExecAffinitySupportArg {
	int * pi;
	CUexecAffinityType  type;
	CUdevice  dev;
};

struct cuDeviceGetExecAffinitySupportResponse {
	int  pi;
	CUresult err;
};

struct cuCtxCreate_v2Arg {
	CUcontext * pctx;
	unsigned int  flags;
	CUdevice  dev;
};

struct cuCtxCreate_v2Response {
	CUcontext  pctx;
	CUresult err;
};

struct cuCtxDestroy_v2Arg {
	CUcontext  ctx;
};

struct cuCtxPushCurrent_v2Arg {
	CUcontext  ctx;
};

struct cuCtxPopCurrent_v2Arg {
	CUcontext * pctx;
};

struct cuCtxPopCurrent_v2Response {
	CUcontext  pctx;
	CUresult err;
};

struct cuCtxSetCurrentArg {
	CUcontext  ctx;
};

struct cuCtxGetCurrentArg {
	CUcontext * pctx;
};

struct cuCtxGetCurrentResponse {
	CUcontext  pctx;
	CUresult err;
};

struct cuCtxGetDeviceArg {
	CUdevice * device;
};

struct cuCtxGetDeviceResponse {
	CUdevice  device;
	CUresult err;
};

struct cuCtxGetFlagsArg {
	unsigned int * flags;
};

struct cuCtxGetFlagsResponse {
	unsigned int  flags;
	CUresult err;
};

struct cuCtxSynchronizeArg {
};

struct cuCtxSetLimitArg {
	CUlimit  limit;
	size_t  value;
};

struct cuCtxGetLimitArg {
	size_t * pvalue;
	CUlimit  limit;
};

struct cuCtxGetLimitResponse {
	size_t  pvalue;
	CUresult err;
};

struct cuCtxGetCacheConfigArg {
	CUfunc_cache * pconfig;
};

struct cuCtxGetCacheConfigResponse {
	CUfunc_cache  pconfig;
	CUresult err;
};

struct cuCtxSetCacheConfigArg {
	CUfunc_cache  config;
};

struct cuCtxGetSharedMemConfigArg {
	CUsharedconfig * pConfig;
};

struct cuCtxGetSharedMemConfigResponse {
	CUsharedconfig  pConfig;
	CUresult err;
};

struct cuCtxSetSharedMemConfigArg {
	CUsharedconfig  config;
};

struct cuCtxGetApiVersionArg {
	CUcontext  ctx;
	unsigned int * version;
};

struct cuCtxGetApiVersionResponse {
	unsigned int  version;
	CUresult err;
};

struct cuCtxGetStreamPriorityRangeArg {
	int * leastPriority;
	int * greatestPriority;
};

struct cuCtxGetStreamPriorityRangeResponse {
	int  leastPriority;
	int  greatestPriority;
	CUresult err;
};

struct cuCtxResetPersistingL2CacheArg {
};

struct cuCtxGetExecAffinityArg {
	CUexecAffinityParam * pExecAffinity;
	CUexecAffinityType  type;
};

struct cuCtxGetExecAffinityResponse {
	CUexecAffinityParam  pExecAffinity;
	CUresult err;
};

struct cuCtxAttachArg {
	CUcontext * pctx;
	unsigned int  flags;
};

struct cuCtxAttachResponse {
	CUcontext  pctx;
	CUresult err;
};

struct cuCtxDetachArg {
	CUcontext  ctx;
};

struct cuModuleUnloadArg {
	CUmodule  hmod;
};

struct cuMemcpyAsyncArg {
	CUdeviceptr  dst;
	CUdeviceptr  src;
	size_t  ByteCount;
	CUstream  hStream;
};

struct cudaDeviceResetArg {
};

struct cudaDeviceSynchronizeArg {
};

struct cudaDeviceSetLimitArg {
	enum cudaLimit  limit;
	size_t  value;
};

struct cudaDeviceGetLimitArg {
	size_t * pValue;
	enum cudaLimit  limit;
};

struct cudaDeviceGetLimitResponse {
	size_t  pValue;
	cudaError_t err;
};

struct cudaDeviceGetCacheConfigArg {
	enum cudaFuncCache * pCacheConfig;
};

struct cudaDeviceGetCacheConfigResponse {
	enum cudaFuncCache  pCacheConfig;
	cudaError_t err;
};

struct cudaDeviceGetStreamPriorityRangeArg {
	int * leastPriority;
	int * greatestPriority;
};

struct cudaDeviceGetStreamPriorityRangeResponse {
	int  leastPriority;
	int  greatestPriority;
	cudaError_t err;
};

struct cudaDeviceSetCacheConfigArg {
	enum cudaFuncCache  cacheConfig;
};

struct cudaDeviceSetSharedMemConfigArg {
	enum cudaSharedMemConfig  config;
};

struct cudaIpcOpenEventHandleArg {
	cudaEvent_t * event;
	cudaIpcEventHandle_t  handle;
};

struct cudaIpcOpenEventHandleResponse {
	cudaEvent_t  event;
	cudaError_t err;
};

struct cudaThreadExitArg {
};

struct cudaThreadSynchronizeArg {
};

struct cudaThreadSetLimitArg {
	enum cudaLimit  limit;
	size_t  value;
};

struct cudaThreadGetLimitArg {
	size_t * pValue;
	enum cudaLimit  limit;
};

struct cudaThreadGetLimitResponse {
	size_t  pValue;
	cudaError_t err;
};

struct cudaThreadGetCacheConfigArg {
	enum cudaFuncCache * pCacheConfig;
};

struct cudaThreadGetCacheConfigResponse {
	enum cudaFuncCache  pCacheConfig;
	cudaError_t err;
};

struct cudaThreadSetCacheConfigArg {
	enum cudaFuncCache  cacheConfig;
};

struct cudaGetLastErrorArg {
};

struct cudaPeekAtLastErrorArg {
};

struct cudaGetDeviceCountArg {
	int * count;
};

struct cudaGetDeviceCountResponse {
	int  count;
	cudaError_t err;
};

struct cudaGetDevicePropertiesArg {
	struct cudaDeviceProp * prop;
	int  device;
};

struct cudaGetDevicePropertiesResponse {
	struct cudaDeviceProp  prop;
	cudaError_t err;
};

struct cudaDeviceGetAttributeArg {
	int * value;
	enum cudaDeviceAttr  attr;
	int  device;
};

struct cudaDeviceGetAttributeResponse {
	int  value;
	cudaError_t err;
};

struct cudaDeviceGetDefaultMemPoolArg {
	cudaMemPool_t * memPool;
	int  device;
};

struct cudaDeviceGetDefaultMemPoolResponse {
	cudaMemPool_t  memPool;
	cudaError_t err;
};

struct cudaDeviceSetMemPoolArg {
	int  device;
	cudaMemPool_t  memPool;
};

struct cudaDeviceGetMemPoolArg {
	cudaMemPool_t * memPool;
	int  device;
};

struct cudaDeviceGetMemPoolResponse {
	cudaMemPool_t  memPool;
	cudaError_t err;
};

struct cudaDeviceGetP2PAttributeArg {
	int * value;
	enum cudaDeviceP2PAttr  attr;
	int  srcDevice;
	int  dstDevice;
};

struct cudaDeviceGetP2PAttributeResponse {
	int  value;
	cudaError_t err;
};

struct cudaSetDeviceArg {
	int  device;
};

struct cudaGetDeviceArg {
	int * device;
};

struct cudaGetDeviceResponse {
	int  device;
	cudaError_t err;
};

struct cudaSetDeviceFlagsArg {
	unsigned int  flags;
};

struct cudaGetDeviceFlagsArg {
	unsigned int * flags;
};

struct cudaGetDeviceFlagsResponse {
	unsigned int  flags;
	cudaError_t err;
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

struct cudaCtxResetPersistingL2CacheArg {
};

struct cudaStreamCopyAttributesArg {
	cudaStream_t  dst;
	cudaStream_t  src;
};

struct cudaStreamDestroyArg {
	cudaStream_t  stream;
};

struct cudaStreamWaitEventArg {
	cudaStream_t  stream;
	cudaEvent_t  event;
	unsigned int  flags;
};

struct cudaStreamSynchronizeArg {
	cudaStream_t  stream;
};

struct cudaStreamQueryArg {
	cudaStream_t  stream;
};

struct cudaStreamBeginCaptureArg {
	cudaStream_t  stream;
	enum cudaStreamCaptureMode  mode;
};

struct cudaStreamEndCaptureArg {
	cudaStream_t  stream;
	cudaGraph_t * pGraph;
};

struct cudaStreamIsCapturingArg {
	cudaStream_t  stream;
	enum cudaStreamCaptureStatus * pCaptureStatus;
};

struct cudaStreamIsCapturingResponse {
	enum cudaStreamCaptureStatus  pCaptureStatus;
	cudaError_t err;
};

struct cudaEventCreateArg {
	cudaEvent_t * event;
};

struct cudaEventCreateResponse {
	cudaEvent_t  event;
	cudaError_t err;
};

struct cudaEventCreateWithFlagsArg {
	cudaEvent_t * event;
	unsigned int  flags;
};

struct cudaEventCreateWithFlagsResponse {
	cudaEvent_t  event;
	cudaError_t err;
};

struct cudaEventRecordArg {
	cudaEvent_t  event;
	cudaStream_t  stream;
};

struct cudaEventRecordWithFlagsArg {
	cudaEvent_t  event;
	cudaStream_t  stream;
	unsigned int  flags;
};

struct cudaEventQueryArg {
	cudaEvent_t  event;
};

struct cudaEventSynchronizeArg {
	cudaEvent_t  event;
};

struct cudaEventDestroyArg {
	cudaEvent_t  event;
};

struct cudaEventElapsedTimeArg {
	float * ms;
	cudaEvent_t  start;
	cudaEvent_t  end;
};

struct cudaEventElapsedTimeResponse {
	float  ms;
	cudaError_t err;
};

struct cudaMemGetInfoArg {
	size_t * free;
	size_t * total;
};

struct cudaMemGetInfoResponse {
	size_t  free;
	size_t  total;
	cudaError_t err;
};

struct cudaMemsetArg {
	void * devPtr;
	int  value;
	size_t  count;
};

struct cudaGraphCreateArg {
	cudaGraph_t * pGraph;
	unsigned int  flags;
};

struct cudaGraphCreateResponse {
	cudaGraph_t  pGraph;
	cudaError_t err;
};

struct cudnnGetVersionArg {
};

struct cudnnGetMaxDeviceVersionArg {
};

struct cudnnGetCudartVersionArg {
};

struct cudnnGetPropertyArg {
	libraryPropertyType  type;
	int * value;
};

struct cudnnGetPropertyResponse {
	int  value;
	cudnnStatus_t err;
};

struct cudnnCreateArg {
	cudnnHandle_t * handle;
};

struct cudnnCreateResponse {
	cudnnHandle_t  handle;
	cudnnStatus_t err;
};

struct cudnnDestroyArg {
	cudnnHandle_t  handle;
};

struct cudnnSetStreamArg {
	cudnnHandle_t  handle;
	cudaStream_t  streamId;
};

struct cudnnGetStreamArg {
	cudnnHandle_t  handle;
	cudaStream_t * streamId;
};

struct cudnnGetStreamResponse {
	cudaStream_t  streamId;
	cudnnStatus_t err;
};

struct cudnnCreateTensorDescriptorArg {
	cudnnTensorDescriptor_t * tensorDesc;
};

struct cudnnCreateTensorDescriptorResponse {
	cudnnTensorDescriptor_t  tensorDesc;
	cudnnStatus_t err;
};

struct cudnnSetTensor4dDescriptorArg {
	cudnnTensorDescriptor_t  tensorDesc;
	cudnnTensorFormat_t  format;
	cudnnDataType_t  dataType;
	int  n;
	int  c;
	int  h;
	int  w;
};

struct cudnnSetTensor4dDescriptorExArg {
	cudnnTensorDescriptor_t  tensorDesc;
	cudnnDataType_t  dataType;
	int  n;
	int  c;
	int  h;
	int  w;
	int  nStride;
	int  cStride;
	int  hStride;
	int  wStride;
};

struct cudnnGetTensor4dDescriptorArg {
	cudnnTensorDescriptor_t  tensorDesc;
	cudnnDataType_t * dataType;
	int * n;
	int * c;
	int * h;
	int * w;
	int * nStride;
	int * cStride;
	int * hStride;
	int * wStride;
};

struct cudnnGetTensor4dDescriptorResponse {
	cudnnDataType_t  dataType;
	int  n;
	int  c;
	int  h;
	int  w;
	int  nStride;
	int  cStride;
	int  hStride;
	int  wStride;
	cudnnStatus_t err;
};

struct cudnnGetTensorSizeInBytesArg {
	cudnnTensorDescriptor_t  tensorDesc;
	size_t * size;
};

struct cudnnGetTensorSizeInBytesResponse {
	size_t  size;
	cudnnStatus_t err;
};

struct cudnnDestroyTensorDescriptorArg {
	cudnnTensorDescriptor_t  tensorDesc;
};

struct cudnnInitTransformDestArg {
	cudnnTensorTransformDescriptor_t  transformDesc;
	cudnnTensorDescriptor_t  srcDesc;
	cudnnTensorDescriptor_t  destDesc;
	size_t * destSizeInBytes;
};

struct cudnnInitTransformDestResponse {
	size_t  destSizeInBytes;
	cudnnStatus_t err;
};

struct cudnnCreateTensorTransformDescriptorArg {
	cudnnTensorTransformDescriptor_t * transformDesc;
};

struct cudnnCreateTensorTransformDescriptorResponse {
	cudnnTensorTransformDescriptor_t  transformDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyTensorTransformDescriptorArg {
	cudnnTensorTransformDescriptor_t  transformDesc;
};

struct cudnnCreateFilterDescriptorArg {
	cudnnFilterDescriptor_t * filterDesc;
};

struct cudnnCreateFilterDescriptorResponse {
	cudnnFilterDescriptor_t  filterDesc;
	cudnnStatus_t err;
};

struct cudnnSetFilter4dDescriptorArg {
	cudnnFilterDescriptor_t  filterDesc;
	cudnnDataType_t  dataType;
	cudnnTensorFormat_t  format;
	int  k;
	int  c;
	int  h;
	int  w;
};

struct cudnnGetFilterSizeInBytesArg {
	cudnnFilterDescriptor_t  filterDesc;
	size_t * size;
};

struct cudnnGetFilterSizeInBytesResponse {
	size_t  size;
	cudnnStatus_t err;
};

struct cudnnDestroyFilterDescriptorArg {
	cudnnFilterDescriptor_t  filterDesc;
};

struct cudnnCreatePoolingDescriptorArg {
	cudnnPoolingDescriptor_t * poolingDesc;
};

struct cudnnCreatePoolingDescriptorResponse {
	cudnnPoolingDescriptor_t  poolingDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyPoolingDescriptorArg {
	cudnnPoolingDescriptor_t  poolingDesc;
};

struct cudnnCreateActivationDescriptorArg {
	cudnnActivationDescriptor_t * activationDesc;
};

struct cudnnCreateActivationDescriptorResponse {
	cudnnActivationDescriptor_t  activationDesc;
	cudnnStatus_t err;
};

struct cudnnSetActivationDescriptorArg {
	cudnnActivationDescriptor_t  activationDesc;
	cudnnActivationMode_t  mode;
	cudnnNanPropagation_t  reluNanOpt;
	double  coef;
};

struct cudnnDestroyActivationDescriptorArg {
	cudnnActivationDescriptor_t  activationDesc;
};

struct cudnnCreateLRNDescriptorArg {
	cudnnLRNDescriptor_t * normDesc;
};

struct cudnnCreateLRNDescriptorResponse {
	cudnnLRNDescriptor_t  normDesc;
	cudnnStatus_t err;
};

struct cudnnSetLRNDescriptorArg {
	cudnnLRNDescriptor_t  normDesc;
	unsigned  lrnN;
	double  lrnAlpha;
	double  lrnBeta;
	double  lrnK;
};

struct cudnnDestroyLRNDescriptorArg {
	cudnnLRNDescriptor_t  lrnDesc;
};

struct cudnnCreateDropoutDescriptorArg {
	cudnnDropoutDescriptor_t * dropoutDesc;
};

struct cudnnCreateDropoutDescriptorResponse {
	cudnnDropoutDescriptor_t  dropoutDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyDropoutDescriptorArg {
	cudnnDropoutDescriptor_t  dropoutDesc;
};

struct cudnnDropoutGetStatesSizeArg {
	cudnnHandle_t  handle;
	size_t * sizeInBytes;
};

struct cudnnDropoutGetStatesSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnSetDropoutDescriptorArg {
	cudnnDropoutDescriptor_t  dropoutDesc;
	cudnnHandle_t  handle;
	float  dropout;
	void * states;
	size_t  stateSizeInBytes;
	unsigned long long  seed;
};

struct cudnnCreateRNNDataDescriptorArg {
	cudnnRNNDataDescriptor_t * rnnDataDesc;
};

struct cudnnCreateRNNDataDescriptorResponse {
	cudnnRNNDataDescriptor_t  rnnDataDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyRNNDataDescriptorArg {
	cudnnRNNDataDescriptor_t  rnnDataDesc;
};

struct cudnnCreateSeqDataDescriptorArg {
	cudnnSeqDataDescriptor_t * seqDataDesc;
};

struct cudnnCreateSeqDataDescriptorResponse {
	cudnnSeqDataDescriptor_t  seqDataDesc;
	cudnnStatus_t err;
};

struct cudnnDestroySeqDataDescriptorArg {
	cudnnSeqDataDescriptor_t  seqDataDesc;
};

struct cudnnCreateAttnDescriptorArg {
	cudnnAttnDescriptor_t * attnDesc;
};

struct cudnnCreateAttnDescriptorResponse {
	cudnnAttnDescriptor_t  attnDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyAttnDescriptorArg {
	cudnnAttnDescriptor_t  attnDesc;
};

struct cudnnSetAttnDescriptorArg {
	cudnnAttnDescriptor_t  attnDesc;
	unsigned  attnMode;
	int  nHeads;
	double  smScaler;
	cudnnDataType_t  dataType;
	cudnnDataType_t  computePrec;
	cudnnMathType_t  mathType;
	cudnnDropoutDescriptor_t  attnDropoutDesc;
	cudnnDropoutDescriptor_t  postDropoutDesc;
	int  qSize;
	int  kSize;
	int  vSize;
	int  qProjSize;
	int  kProjSize;
	int  vProjSize;
	int  oProjSize;
	int  qoMaxSeqLength;
	int  kvMaxSeqLength;
	int  maxBatchSize;
	int  maxBeamSize;
};

struct cudnnGetMultiHeadAttnBuffersArg {
	cudnnHandle_t  handle;
	cudnnAttnDescriptor_t  attnDesc;
	size_t * weightSizeInBytes;
	size_t * workSpaceSizeInBytes;
	size_t * reserveSpaceSizeInBytes;
};

struct cudnnGetMultiHeadAttnBuffersResponse {
	size_t  weightSizeInBytes;
	size_t  workSpaceSizeInBytes;
	size_t  reserveSpaceSizeInBytes;
	cudnnStatus_t err;
};

struct cudnnCreateConvolutionDescriptorArg {
	cudnnConvolutionDescriptor_t * convDesc;
};

struct cudnnCreateConvolutionDescriptorResponse {
	cudnnConvolutionDescriptor_t  convDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyConvolutionDescriptorArg {
	cudnnConvolutionDescriptor_t  convDesc;
};

struct cudnnGetConvolutionForwardWorkspaceSizeArg {
	cudnnHandle_t  handle;
	cudnnTensorDescriptor_t  xDesc;
	cudnnFilterDescriptor_t  wDesc;
	cudnnConvolutionDescriptor_t  convDesc;
	cudnnTensorDescriptor_t  yDesc;
	cudnnConvolutionFwdAlgo_t  algo;
	size_t * sizeInBytes;
};

struct cudnnGetConvolutionForwardWorkspaceSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnCnnTrainVersionCheckArg {
};

struct cudnnBackendCreateDescriptorArg {
	cudnnBackendDescriptorType_t  descriptorType;
	cudnnBackendDescriptor_t * descriptor;
};

struct cudnnBackendCreateDescriptorResponse {
	cudnnBackendDescriptor_t  descriptor;
	cudnnStatus_t err;
};

struct cudnnBackendDestroyDescriptorArg {
	cudnnBackendDescriptor_t  descriptor;
};

struct cudnnBackendInitializeArg {
	cudnnBackendDescriptor_t  descriptor;
};

struct cudnnBackendFinalizeArg {
	cudnnBackendDescriptor_t  descriptor;
};

struct cudnnBackendExecuteArg {
	cudnnHandle_t  handle;
	cudnnBackendDescriptor_t  executionPlan;
	cudnnBackendDescriptor_t  variantPack;
};

struct cublasCreate_v2Arg {
	cublasHandle_t*  handle;
};

struct cublasCreate_v2Response {
	cublasHandle_t handle;
	cublasStatus_t err;
};

struct cublasDestroy_v2Arg {
	cublasHandle_t  handle;
};

struct cublasGetVersion_v2Arg {
	cublasHandle_t  handle;
	int*  version;
};

struct cublasGetVersion_v2Response {
	int version;
	cublasStatus_t err;
};

struct cublasGetCudartVersionArg {
};

struct cublasSetWorkspace_v2Arg {
	cublasHandle_t  handle;
	void*  workspace;
	size_t  workspaceSizeInBytes;
};

struct cublasSetStream_v2Arg {
	cublasHandle_t  handle;
	cudaStream_t  streamId;
};

struct cublasGetStream_v2Arg {
	cublasHandle_t  handle;
	cudaStream_t*  streamId;
};

struct cublasGetStream_v2Response {
	cudaStream_t streamId;
	cublasStatus_t err;
};

struct cublasGetPointerMode_v2Arg {
	cublasHandle_t  handle;
	cublasPointerMode_t*  mode;
};

struct cublasGetPointerMode_v2Response {
	cublasPointerMode_t mode;
	cublasStatus_t err;
};

struct cublasSetPointerMode_v2Arg {
	cublasHandle_t  handle;
	cublasPointerMode_t  mode;
};

struct cublasSetMathModeArg {
	cublasHandle_t  handle;
	cublasMath_t  mode;
};

struct cudaProfilerStartArg {
};

struct cudaProfilerStopArg {
};

struct cublasLtCreateArg {
	cublasLtHandle_t*  lightHandle;
};

struct cublasLtCreateResponse {
	cublasLtHandle_t lightHandle;
	cublasStatus_t err;
};

struct cublasLtDestroyArg {
	cublasLtHandle_t  lightHandle;
};

struct cublasLtMatrixLayoutCreateArg {
	cublasLtMatrixLayout_t*  matLayout;
	cudaDataType  type;
	uint64_t  rows;
	uint64_t  cols;
	int64_t  ld;
};

struct cublasLtMatrixLayoutCreateResponse {
	cublasLtMatrixLayout_t matLayout;
	cublasStatus_t err;
};

struct cublasLtMatrixLayoutDestroyArg {
	cublasLtMatrixLayout_t  matLayout;
};

struct cublasLtMatmulDescCreateArg {
	cublasLtMatmulDesc_t*  matmulDesc;
	cublasComputeType_t  computeType;
	cudaDataType_t  scaleType;
};

struct cublasLtMatmulDescCreateResponse {
	cublasLtMatmulDesc_t matmulDesc;
	cublasStatus_t err;
};

struct cublasLtMatmulDescDestroyArg {
	cublasLtMatmulDesc_t  matmulDesc;
};

struct cublasLtMatmulPreferenceCreateArg {
	cublasLtMatmulPreference_t*  pref;
};

struct cublasLtMatmulPreferenceCreateResponse {
	cublasLtMatmulPreference_t pref;
	cublasStatus_t err;
};

struct cublasLtMatmulPreferenceDestroyArg {
	cublasLtMatmulPreference_t  pref;
};



#endif // TALLY_GENERATED_MSG_STRUCT_H
