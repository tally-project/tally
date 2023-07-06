
#ifndef TALLY_GENERATED_MSG_STRUCT_H
#define TALLY_GENERATED_MSG_STRUCT_H

#include <cuda.h>
#include <cuda_runtime.h>

struct cuInitArg {
	unsigned int  Flags;
};

struct cudaDeviceResetArg {
};

struct cudaDeviceSynchronizeArg {
};

struct cudaDeviceSetLimitArg {
	enum cudaLimit  limit;
	size_t  value;
};

struct cudaDeviceSetCacheConfigArg {
	enum cudaFuncCache  cacheConfig;
};

struct cudaDeviceSetSharedMemConfigArg {
	enum cudaSharedMemConfig  config;
};

struct cudaThreadExitArg {
};

struct cudaThreadSynchronizeArg {
};

struct cudaThreadSetLimitArg {
	enum cudaLimit  limit;
	size_t  value;
};

struct cudaThreadSetCacheConfigArg {
	enum cudaFuncCache  cacheConfig;
};

struct cudaGetLastErrorArg {
};

struct cudaPeekAtLastErrorArg {
};

struct cudaSetDeviceArg {
	int  device;
};

struct cudaSetDeviceFlagsArg {
	unsigned int  flags;
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

struct cudaProfilerStartArg {
};

struct cudaProfilerStopArg {
};



#endif // TALLY_GENERATED_MSG_STRUCT_H
