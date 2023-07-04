
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

struct cudaCtxResetPersistingL2CacheArg {
};

struct cudaFreeArg {
	void * devPtr;
};

struct cudaProfilerStartArg {
};

struct cudaProfilerStopArg {
};



#endif // TALLY_GENERATED_MSG_STRUCT_H
