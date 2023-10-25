
#ifndef TALLY_GENERATED_MSG_STRUCT_H
#define TALLY_GENERATED_MSG_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>


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

struct cuDeviceGetExecAffinitySupportArg {
	int * pi;
	CUexecAffinityType  type;
	CUdevice  dev;
};

struct cuDeviceGetExecAffinitySupportResponse {
	int  pi;
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

struct cuModuleGetLoadingModeArg {
	CUmoduleLoadingMode * mode;
};

struct cuModuleGetLoadingModeResponse {
	CUmoduleLoadingMode  mode;
	CUresult err;
};

struct cuMemGetInfo_v2Arg {
	size_t * free;
	size_t * total;
};

struct cuMemGetInfo_v2Response {
	size_t  free;
	size_t  total;
	CUresult err;
};

struct cuMemAllocFromPoolAsyncArg {
	CUdeviceptr * dptr;
	size_t  bytesize;
	CUmemoryPool  pool;
	CUstream  hStream;
};

struct cuMemAllocFromPoolAsyncResponse {
	CUdeviceptr  dptr;
	CUresult err;
};

struct cuStreamWaitEventArg {
	CUstream  hStream;
	CUevent  hEvent;
	unsigned int  Flags;
};

struct cuStreamBeginCapture_v2Arg {
	CUstream  hStream;
	CUstreamCaptureMode  mode;
};

struct cuStreamIsCapturingArg {
	CUstream  hStream;
	CUstreamCaptureStatus * captureStatus;
};

struct cuStreamIsCapturingResponse {
	CUstreamCaptureStatus  captureStatus;
	CUresult err;
};

struct cuEventCreateArg {
	CUevent * phEvent;
	unsigned int  Flags;
};

struct cuEventCreateResponse {
	CUevent  phEvent;
	CUresult err;
};

struct cuEventRecordArg {
	CUevent  hEvent;
	CUstream  hStream;
};

struct cuEventQueryArg {
	CUevent  hEvent;
};

struct cuEventSynchronizeArg {
	CUevent  hEvent;
};

struct cuEventDestroy_v2Arg {
	CUevent  hEvent;
};

struct cuEventElapsedTimeArg {
	float * pMilliseconds;
	CUevent  hStart;
	CUevent  hEnd;
};

struct cuEventElapsedTimeResponse {
	float  pMilliseconds;
	CUresult err;
};

struct cuDestroyExternalMemoryArg {
	CUexternalMemory  extMem;
};

struct cuGraphInstantiateWithFlagsArg {
	CUgraphExec * phGraphExec;
	CUgraph  hGraph;
	unsigned long long  flags;
};

struct cuGraphInstantiateWithFlagsResponse {
	CUgraphExec  phGraphExec;
	CUresult err;
};

struct cuGraphLaunchArg {
	CUgraphExec  hGraphExec;
	CUstream  hStream;
};

struct cuGraphExecDestroyArg {
	CUgraphExec  hGraphExec;
};

struct cuGraphDestroyArg {
	CUgraph  hGraph;
};

struct cuGraphExecUpdate_v2Arg {
	CUgraphExec  hGraphExec;
	CUgraph  hGraph;
	CUgraphExecUpdateResultInfo * resultInfo;
};

struct cuGraphExecUpdate_v2Response {
	CUgraphExecUpdateResultInfo  resultInfo;
	CUresult err;
};

struct cudaDeviceResetArg {
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

struct cudaDeviceGetPCIBusIdArg {
	char * pciBusId;
	int  len;
	int  device;
};

struct cudaDeviceGetPCIBusIdResponse {
	char  pciBusId;
	cudaError_t err;
};

struct cudaIpcGetEventHandleArg {
	cudaIpcEventHandle_t * handle;
	cudaEvent_t  event;
};

struct cudaIpcGetEventHandleResponse {
	cudaIpcEventHandle_t  handle;
	cudaError_t err;
};

struct cudaIpcOpenEventHandleArg {
	cudaEvent_t * event;
	cudaIpcEventHandle_t  handle;
};

struct cudaIpcOpenEventHandleResponse {
	cudaEvent_t  event;
	cudaError_t err;
};

struct cudaIpcGetMemHandleArg {
	cudaIpcMemHandle_t * handle;
	void * devPtr;
};

struct cudaIpcGetMemHandleResponse {
	cudaIpcMemHandle_t  handle;
	cudaError_t err;
};

struct cudaIpcOpenMemHandleArg {
	void ** devPtr;
	cudaIpcMemHandle_t  handle;
	unsigned int  flags;
};

struct cudaIpcOpenMemHandleResponse {
	void * devPtr;
	cudaError_t err;
};

struct cudaIpcCloseMemHandleArg {
	void * devPtr;
};

struct cudaDeviceFlushGPUDirectRDMAWritesArg {
	enum cudaFlushGPUDirectRDMAWritesTarget  target;
	enum cudaFlushGPUDirectRDMAWritesScope  scope;
};

struct cudaThreadExitArg {
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

struct cudaGetDeviceProperties_v2Arg {
	struct cudaDeviceProp * prop;
	int  device;
};

struct cudaGetDeviceProperties_v2Response {
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

struct cudaStreamGetPriorityArg {
	cudaStream_t  hStream;
	int * priority;
};

struct cudaStreamGetPriorityResponse {
	int  priority;
	cudaError_t err;
};

struct cudaStreamGetFlagsArg {
	cudaStream_t  hStream;
	unsigned int * flags;
};

struct cudaStreamGetFlagsResponse {
	unsigned int  flags;
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

struct cudaStreamQueryArg {
	cudaStream_t  stream;
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

struct cudaMemsetAsyncArg {
	void * devPtr;
	int  value;
	size_t  count;
	cudaStream_t  stream;
};

struct cudaGraphCreateArg {
	cudaGraph_t * pGraph;
	unsigned int  flags;
};

struct cudaGraphCreateResponse {
	cudaGraph_t  pGraph;
	cudaError_t err;
};

struct cudaGraphInstantiateWithFlagsArg {
	cudaGraphExec_t * pGraphExec;
	cudaGraph_t  graph;
	unsigned long long  flags;
};

struct cudaGraphInstantiateWithFlagsResponse {
	cudaGraphExec_t  pGraphExec;
	cudaError_t err;
};

struct cudaGraphUploadArg {
	cudaGraphExec_t  graphExec;
	cudaStream_t  stream;
};

struct cudaGraphLaunchArg {
	cudaGraphExec_t  graphExec;
	cudaStream_t  stream;
};

struct cudaGraphExecDestroyArg {
	cudaGraphExec_t  graphExec;
};

struct cudaGraphDestroyArg {
	cudaGraph_t  graph;
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

struct cudnnCreateOpTensorDescriptorArg {
	cudnnOpTensorDescriptor_t * opTensorDesc;
};

struct cudnnCreateOpTensorDescriptorResponse {
	cudnnOpTensorDescriptor_t  opTensorDesc;
	cudnnStatus_t err;
};

struct cudnnSetOpTensorDescriptorArg {
	cudnnOpTensorDescriptor_t  opTensorDesc;
	cudnnOpTensorOp_t  opTensorOp;
	cudnnDataType_t  opTensorCompType;
	cudnnNanPropagation_t  opTensorNanOpt;
};

struct cudnnGetOpTensorDescriptorArg {
	cudnnOpTensorDescriptor_t  opTensorDesc;
	cudnnOpTensorOp_t * opTensorOp;
	cudnnDataType_t * opTensorCompType;
	cudnnNanPropagation_t * opTensorNanOpt;
};

struct cudnnGetOpTensorDescriptorResponse {
	cudnnOpTensorOp_t  opTensorOp;
	cudnnDataType_t  opTensorCompType;
	cudnnNanPropagation_t  opTensorNanOpt;
	cudnnStatus_t err;
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

struct cudnnRestoreDropoutDescriptorArg {
	cudnnDropoutDescriptor_t  dropoutDesc;
	cudnnHandle_t  handle;
	float  dropout;
	void * states;
	size_t  stateSizeInBytes;
	unsigned long long  seed;
};

struct cudnnOpsInferVersionCheckArg {
};

struct cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeArg {
	cudnnHandle_t  handle;
	cudnnBatchNormMode_t  mode;
	cudnnBatchNormOps_t  bnOps;
	cudnnTensorDescriptor_t  xDesc;
	cudnnTensorDescriptor_t  zDesc;
	cudnnTensorDescriptor_t  yDesc;
	cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc;
	cudnnActivationDescriptor_t  activationDesc;
	size_t * sizeInBytes;
};

struct cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnGetBatchNormalizationBackwardExWorkspaceSizeArg {
	cudnnHandle_t  handle;
	cudnnBatchNormMode_t  mode;
	cudnnBatchNormOps_t  bnOps;
	cudnnTensorDescriptor_t  xDesc;
	cudnnTensorDescriptor_t  yDesc;
	cudnnTensorDescriptor_t  dyDesc;
	cudnnTensorDescriptor_t  dzDesc;
	cudnnTensorDescriptor_t  dxDesc;
	cudnnTensorDescriptor_t  dBnScaleBiasDesc;
	cudnnActivationDescriptor_t  activationDesc;
	size_t * sizeInBytes;
};

struct cudnnGetBatchNormalizationBackwardExWorkspaceSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnGetBatchNormalizationTrainingExReserveSpaceSizeArg {
	cudnnHandle_t  handle;
	cudnnBatchNormMode_t  mode;
	cudnnBatchNormOps_t  bnOps;
	cudnnActivationDescriptor_t  activationDesc;
	cudnnTensorDescriptor_t  xDesc;
	size_t * sizeInBytes;
};

struct cudnnGetBatchNormalizationTrainingExReserveSpaceSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnOpsTrainVersionCheckArg {
};

struct cudnnCreateRNNDescriptorArg {
	cudnnRNNDescriptor_t * rnnDesc;
};

struct cudnnCreateRNNDescriptorResponse {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnStatus_t err;
};

struct cudnnDestroyRNNDescriptorArg {
	cudnnRNNDescriptor_t  rnnDesc;
};

struct cudnnSetRNNDescriptor_v8Arg {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnRNNAlgo_t  algo;
	cudnnRNNMode_t  cellMode;
	cudnnRNNBiasMode_t  biasMode;
	cudnnDirectionMode_t  dirMode;
	cudnnRNNInputMode_t  inputMode;
	cudnnDataType_t  dataType;
	cudnnDataType_t  mathPrec;
	cudnnMathType_t  mathType;
	int32_t  inputSize;
	int32_t  hiddenSize;
	int32_t  projSize;
	int32_t  numLayers;
	cudnnDropoutDescriptor_t  dropoutDesc;
	uint32_t  auxFlags;
};

struct cudnnSetRNNDescriptor_v6Arg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int  hiddenSize;
	int  numLayers;
	cudnnDropoutDescriptor_t  dropoutDesc;
	cudnnRNNInputMode_t  inputMode;
	cudnnDirectionMode_t  direction;
	cudnnRNNMode_t  cellMode;
	cudnnRNNAlgo_t  algo;
	cudnnDataType_t  mathPrec;
};

struct cudnnSetRNNMatrixMathTypeArg {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnMathType_t  mType;
};

struct cudnnGetRNNMatrixMathTypeArg {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnMathType_t * mType;
};

struct cudnnGetRNNMatrixMathTypeResponse {
	cudnnMathType_t  mType;
	cudnnStatus_t err;
};

struct cudnnSetRNNBiasModeArg {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnRNNBiasMode_t  biasMode;
};

struct cudnnGetRNNBiasModeArg {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnRNNBiasMode_t * biasMode;
};

struct cudnnGetRNNBiasModeResponse {
	cudnnRNNBiasMode_t  biasMode;
	cudnnStatus_t err;
};

struct cudnnRNNSetClip_v8Arg {
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnRNNClipMode_t  clipMode;
	cudnnNanPropagation_t  clipNanOpt;
	double  lclip;
	double  rclip;
};

struct cudnnRNNSetClipArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnRNNClipMode_t  clipMode;
	cudnnNanPropagation_t  clipNanOpt;
	double  lclip;
	double  rclip;
};

struct cudnnBuildRNNDynamicArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int  miniBatch;
};

struct cudnnGetRNNTempSpaceSizesArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnForwardMode_t  fwdMode;
	cudnnRNNDataDescriptor_t  xDesc;
	size_t * workSpaceSize;
	size_t * reserveSpaceSize;
};

struct cudnnGetRNNTempSpaceSizesResponse {
	size_t  workSpaceSize;
	size_t  reserveSpaceSize;
	cudnnStatus_t err;
};

struct cudnnGetRNNParamsSizeArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnTensorDescriptor_t  xDesc;
	size_t * sizeInBytes;
	cudnnDataType_t  dataType;
};

struct cudnnGetRNNParamsSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnGetRNNWeightSpaceSizeArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	size_t * weightSpaceSize;
};

struct cudnnGetRNNWeightSpaceSizeResponse {
	size_t  weightSpaceSize;
	cudnnStatus_t err;
};

struct cudnnGetRNNLinLayerMatrixParamsArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int  pseudoLayer;
	cudnnTensorDescriptor_t  xDesc;
	cudnnFilterDescriptor_t  wDesc;
	void * w;
	int  linLayerID;
	cudnnFilterDescriptor_t  linLayerMatDesc;
	void ** linLayerMat;
};

struct cudnnGetRNNLinLayerMatrixParamsResponse {
	void * linLayerMat;
	cudnnStatus_t err;
};

struct cudnnGetRNNLinLayerBiasParamsArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int  pseudoLayer;
	cudnnTensorDescriptor_t  xDesc;
	cudnnFilterDescriptor_t  wDesc;
	void * w;
	int  linLayerID;
	cudnnFilterDescriptor_t  linLayerBiasDesc;
	void ** linLayerBias;
};

struct cudnnGetRNNLinLayerBiasParamsResponse {
	void * linLayerBias;
	cudnnStatus_t err;
};

struct cudnnGetRNNWeightParamsArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int32_t  pseudoLayer;
	size_t  weightSpaceSize;
	void * weightSpace;
	int32_t  linLayerID;
	cudnnTensorDescriptor_t  mDesc;
	void ** mAddr;
	cudnnTensorDescriptor_t  bDesc;
	void ** bAddr;
};

struct cudnnGetRNNWeightParamsResponse {
	void * mAddr;
	void * bAddr;
	cudnnStatus_t err;
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

struct cudnnSetRNNAlgorithmDescriptorArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	cudnnAlgorithmDescriptor_t  algoDesc;
};

struct cudnnGetRNNForwardInferenceAlgorithmMaxCountArg {
	cudnnHandle_t  handle;
	cudnnRNNDescriptor_t  rnnDesc;
	int * count;
};

struct cudnnGetRNNForwardInferenceAlgorithmMaxCountResponse {
	int  count;
	cudnnStatus_t err;
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

struct cudnnAdvInferVersionCheckArg {
};

struct cudnnAdvTrainVersionCheckArg {
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

struct cudnnGetConvolutionBackwardDataAlgorithmMaxCountArg {
	cudnnHandle_t  handle;
	int * count;
};

struct cudnnGetConvolutionBackwardDataAlgorithmMaxCountResponse {
	int  count;
	cudnnStatus_t err;
};

struct cudnnGetConvolutionBackwardDataWorkspaceSizeArg {
	cudnnHandle_t  handle;
	cudnnFilterDescriptor_t  wDesc;
	cudnnTensorDescriptor_t  dyDesc;
	cudnnConvolutionDescriptor_t  convDesc;
	cudnnTensorDescriptor_t  dxDesc;
	cudnnConvolutionBwdDataAlgo_t  algo;
	size_t * sizeInBytes;
};

struct cudnnGetConvolutionBackwardDataWorkspaceSizeResponse {
	size_t  sizeInBytes;
	cudnnStatus_t err;
};

struct cudnnGetFoldedConvBackwardDataDescriptorsArg {
	cudnnHandle_t  handle;
	cudnnFilterDescriptor_t  filterDesc;
	cudnnTensorDescriptor_t  diffDesc;
	cudnnConvolutionDescriptor_t  convDesc;
	cudnnTensorDescriptor_t  gradDesc;
	cudnnTensorFormat_t  transformFormat;
	cudnnFilterDescriptor_t  foldedFilterDesc;
	cudnnTensorDescriptor_t  paddedDiffDesc;
	cudnnConvolutionDescriptor_t  foldedConvDesc;
	cudnnTensorDescriptor_t  foldedGradDesc;
	cudnnTensorTransformDescriptor_t  filterFoldTransDesc;
	cudnnTensorTransformDescriptor_t  diffPadTransDesc;
	cudnnTensorTransformDescriptor_t  gradFoldTransDesc;
	cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc;
};

struct cudnnCnnInferVersionCheckArg {
};

struct cudnnGetConvolutionBackwardFilterAlgorithmMaxCountArg {
	cudnnHandle_t  handle;
	int * count;
};

struct cudnnGetConvolutionBackwardFilterAlgorithmMaxCountResponse {
	int  count;
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

struct cublasGetPropertyArg {
	libraryPropertyType  type;
	int*  value;
};

struct cublasGetPropertyResponse {
	int value;
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

struct cublasGetMathModeArg {
	cublasHandle_t  handle;
	cublasMath_t*  mode;
};

struct cublasGetMathModeResponse {
	cublasMath_t mode;
	cublasStatus_t err;
};

struct cublasSetMathModeArg {
	cublasHandle_t  handle;
	cublasMath_t  mode;
};

struct cublasGetSmCountTargetArg {
	cublasHandle_t  handle;
	int*  smCountTarget;
};

struct cublasGetSmCountTargetResponse {
	int smCountTarget;
	cublasStatus_t err;
};

struct cublasSetSmCountTargetArg {
	cublasHandle_t  handle;
	int  smCountTarget;
};

struct cublasSetLoggerCallbackArg {
	cublasLogCallback  userCallback;
};

struct cublasGetLoggerCallbackArg {
	cublasLogCallback*  userCallback;
};

struct cublasGetLoggerCallbackResponse {
	cublasLogCallback userCallback;
	cublasStatus_t err;
};

struct cublasSetVectorArg {
	int  n;
	int  elemSize;
	void*  x;
	int  incx;
	void*  devicePtr;
	int  incy;
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

struct cublasLtGetVersionArg {
};

struct cublasLtGetCudartVersionArg {
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

struct cublasLtLoggerForceDisableArg {
};



#endif // TALLY_GENERATED_MSG_STRUCT_H
