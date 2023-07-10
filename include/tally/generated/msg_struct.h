
#ifndef TALLY_GENERATED_MSG_STRUCT_H
#define TALLY_GENERATED_MSG_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>

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

struct cuDeviceTotalMem_v2Arg {
	size_t * bytes;
	CUdevice  dev;
};

struct cuDeviceTotalMem_v2Response {
	size_t  bytes;
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

struct cudaFreeArg {
	void * devPtr;
};

struct cudaGraphCreateArg {
	cudaGraph_t * pGraph;
	unsigned int  flags;
};

struct cudaGraphCreateResponse {
	cudaGraph_t  pGraph;
	cudaError_t err;
};

struct cublasDestroy_v2Arg {
	cublasHandle_t  handle;
};

struct cublasGetCudartVersionArg {
};

struct cudaProfilerStartArg {
};

struct cudaProfilerStopArg {
};



#endif // TALLY_GENERATED_MSG_STRUCT_H
