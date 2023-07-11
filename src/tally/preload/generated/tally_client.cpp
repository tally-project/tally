
#include <dlfcn.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cxxabi.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <unistd.h>
#include <cstring>

#include "tally/cuda_util.h"
#include "tally/msg_struct.h"
#include "tally/client.h"
#include "tally/ipc_util.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"
#include "tally/generated/msg_struct.h"



extern "C" { 

CUresult cuGetErrorString(CUresult  error, const char ** pStr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGetErrorString(error, pStr);
	return res;
}

CUresult cuGetErrorName(CUresult  error, const char ** pStr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGetErrorName(error, pStr);
	return res;
}

CUresult cuInit(unsigned int  Flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuInitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUINIT;
    
    struct cuInitArg *arg_ptr = (struct cuInitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->Flags = Flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuDriverGetVersion(int * driverVersion)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDriverGetVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDRIVERGETVERSION;
    
    struct cuDriverGetVersionArg *arg_ptr = (struct cuDriverGetVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->driverVersion = driverVersion;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDriverGetVersionResponse *) dat;
	*driverVersion = res->driverVersion;
return res->err;
}

CUresult cuDeviceGet(CUdevice * device, int  ordinal)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGET;
    
    struct cuDeviceGetArg *arg_ptr = (struct cuDeviceGetArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->device = device;
	arg_ptr->ordinal = ordinal;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetResponse *) dat;
	*device = res->device;
return res->err;
}

CUresult cuDeviceGetCount(int * count)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetCountArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETCOUNT;
    
    struct cuDeviceGetCountArg *arg_ptr = (struct cuDeviceGetCountArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->count = count;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetCountResponse *) dat;
	*count = res->count;
return res->err;
}

CUresult cuDeviceGetName(char * name, int  len, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetName(name, len, dev);
	return res;
}

CUresult cuDeviceGetUuid(CUuuid * uuid, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetUuidArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETUUID;
    
    struct cuDeviceGetUuidArg *arg_ptr = (struct cuDeviceGetUuidArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->uuid = uuid;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetUuidResponse *) dat;
	*uuid = res->uuid;
return res->err;
}

CUresult cuDeviceGetUuid_v2(CUuuid * uuid, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetUuid_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETUUID_V2;
    
    struct cuDeviceGetUuid_v2Arg *arg_ptr = (struct cuDeviceGetUuid_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->uuid = uuid;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetUuid_v2Response *) dat;
	*uuid = res->uuid;
return res->err;
}

CUresult cuDeviceGetLuid(char * luid, unsigned int * deviceNodeMask, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetLuid(luid, deviceNodeMask, dev);
	return res;
}

CUresult cuDeviceTotalMem_v2(size_t * bytes, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceTotalMem_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICETOTALMEM_V2;
    
    struct cuDeviceTotalMem_v2Arg *arg_ptr = (struct cuDeviceTotalMem_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->bytes = bytes;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceTotalMem_v2Response *) dat;
	*bytes = res->bytes;
return res->err;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, CUarray_format  format, unsigned  numChannels, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev);
	return res;
}

CUresult cuDeviceGetAttribute(int * pi, CUdevice_attribute  attrib, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetAttributeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETATTRIBUTE;
    
    struct cuDeviceGetAttributeArg *arg_ptr = (struct cuDeviceGetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pi = pi;
	arg_ptr->attrib = attrib;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetAttributeResponse *) dat;
	*pi = res->pi;
return res->err;
}

CUresult cuDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, CUdevice  dev, int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags);
	return res;
}

CUresult cuDeviceSetMemPool(CUdevice  dev, CUmemoryPool  pool)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceSetMemPoolArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICESETMEMPOOL;
    
    struct cuDeviceSetMemPoolArg *arg_ptr = (struct cuDeviceSetMemPoolArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dev = dev;
	arg_ptr->pool = pool;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuDeviceGetMemPool(CUmemoryPool * pool, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetMemPoolArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETMEMPOOL;
    
    struct cuDeviceGetMemPoolArg *arg_ptr = (struct cuDeviceGetMemPoolArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pool = pool;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetMemPoolResponse *) dat;
	*pool = res->pool;
return res->err;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool * pool_out, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetDefaultMemPoolArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETDEFAULTMEMPOOL;
    
    struct cuDeviceGetDefaultMemPoolArg *arg_ptr = (struct cuDeviceGetDefaultMemPoolArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pool_out = pool_out;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetDefaultMemPoolResponse *) dat;
	*pool_out = res->pool_out;
return res->err;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget  target, CUflushGPUDirectRDMAWritesScope  scope)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuFlushGPUDirectRDMAWritesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUFLUSHGPUDIRECTRDMAWRITES;
    
    struct cuFlushGPUDirectRDMAWritesArg *arg_ptr = (struct cuFlushGPUDirectRDMAWritesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->target = target;
	arg_ptr->scope = scope;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuDeviceGetProperties(CUdevprop * prop, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetPropertiesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETPROPERTIES;
    
    struct cuDeviceGetPropertiesArg *arg_ptr = (struct cuDeviceGetPropertiesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->prop = prop;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetPropertiesResponse *) dat;
	*prop = res->prop;
return res->err;
}

CUresult cuDeviceComputeCapability(int * major, int * minor, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceComputeCapability(major, minor, dev);
	return res;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext * pctx, CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDevicePrimaryCtxRetainArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXRETAIN;
    
    struct cuDevicePrimaryCtxRetainArg *arg_ptr = (struct cuDevicePrimaryCtxRetainArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pctx = pctx;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDevicePrimaryCtxRetainResponse *) dat;
	*pctx = res->pctx;
return res->err;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDevicePrimaryCtxRelease_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXRELEASE_V2;
    
    struct cuDevicePrimaryCtxRelease_v2Arg *arg_ptr = (struct cuDevicePrimaryCtxRelease_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice  dev, unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDevicePrimaryCtxSetFlags_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXSETFLAGS_V2;
    
    struct cuDevicePrimaryCtxSetFlags_v2Arg *arg_ptr = (struct cuDevicePrimaryCtxSetFlags_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dev = dev;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice  dev, unsigned int * flags, int * active)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDevicePrimaryCtxGetStateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXGETSTATE;
    
    struct cuDevicePrimaryCtxGetStateArg *arg_ptr = (struct cuDevicePrimaryCtxGetStateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dev = dev;
	arg_ptr->flags = flags;
	arg_ptr->active = active;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDevicePrimaryCtxGetStateResponse *) dat;
	*flags = res->flags;
	*active = res->active;
return res->err;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice  dev)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDevicePrimaryCtxReset_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXRESET_V2;
    
    struct cuDevicePrimaryCtxReset_v2Arg *arg_ptr = (struct cuDevicePrimaryCtxReset_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuDeviceGetExecAffinitySupport(int * pi, CUexecAffinityType  type, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetExecAffinitySupport(pi, type, dev);
	return res;
}

CUresult cuCtxCreate_v2(CUcontext * pctx, unsigned int  flags, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxCreate_v2(pctx, flags, dev);
	return res;
}

CUresult cuCtxCreate_v3(CUcontext * pctx, CUexecAffinityParam * paramsArray, int  numParams, unsigned int  flags, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
	return res;
}

CUresult cuCtxDestroy_v2(CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxDestroy_v2(ctx);
	return res;
}

CUresult cuCtxPushCurrent_v2(CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxPushCurrent_v2(ctx);
	return res;
}

CUresult cuCtxPopCurrent_v2(CUcontext * pctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxPopCurrent_v2(pctx);
	return res;
}

CUresult cuCtxSetCurrent(CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxSetCurrent(ctx);
	return res;
}

CUresult cuCtxGetCurrent(CUcontext * pctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetCurrent(pctx);
	return res;
}

CUresult cuCtxGetDevice(CUdevice * device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetDevice(device);
	return res;
}

CUresult cuCtxGetFlags(unsigned int * flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetFlags(flags);
	return res;
}

CUresult cuCtxSynchronize()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxSynchronize();
	return res;
}

CUresult cuCtxSetLimit(CUlimit  limit, size_t  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxSetLimit(limit, value);
	return res;
}

CUresult cuCtxGetLimit(size_t * pvalue, CUlimit  limit)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetLimit(pvalue, limit);
	return res;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache * pconfig)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetCacheConfig(pconfig);
	return res;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache  config)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxSetCacheConfig(config);
	return res;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetSharedMemConfig(pConfig);
	return res;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig  config)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxSetSharedMemConfig(config);
	return res;
}

CUresult cuCtxGetApiVersion(CUcontext  ctx, unsigned int * version)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetApiVersion(ctx, version);
	return res;
}

CUresult cuCtxGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
	return res;
}

CUresult cuCtxResetPersistingL2Cache()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxResetPersistingL2Cache();
	return res;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType  type)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxGetExecAffinity(pExecAffinity, type);
	return res;
}

CUresult cuCtxAttach(CUcontext * pctx, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxAttach(pctx, flags);
	return res;
}

CUresult cuCtxDetach(CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxDetach(ctx);
	return res;
}

CUresult cuModuleLoad(CUmodule * module, const char * fname)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleLoad(module, fname);
	return res;
}

CUresult cuModuleLoadData(CUmodule * module, const void * image)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleLoadData(module, image);
	return res;
}

CUresult cuModuleLoadDataEx(CUmodule * module, const void * image, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleLoadDataEx(module, image, numOptions, options, optionValues);
	return res;
}

CUresult cuModuleLoadFatBinary(CUmodule * module, const void * fatCubin)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleLoadFatBinary(module, fatCubin);
	return res;
}

CUresult cuModuleUnload(CUmodule  hmod)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleUnload(hmod);
	return res;
}

CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule  hmod, const char * name)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleGetFunction(hfunc, hmod, name);
	return res;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule  hmod, const char * name)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleGetGlobal_v2(dptr, bytes, hmod, name);
	return res;
}

CUresult cuModuleGetTexRef(CUtexref * pTexRef, CUmodule  hmod, const char * name)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleGetTexRef(pTexRef, hmod, name);
	return res;
}

CUresult cuModuleGetSurfRef(CUsurfref * pSurfRef, CUmodule  hmod, const char * name)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuModuleGetSurfRef(pSurfRef, hmod, name);
	return res;
}

CUresult cuLinkCreate_v2(unsigned int  numOptions, CUjit_option * options, void ** optionValues, CUlinkState * stateOut)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLinkCreate_v2(numOptions, options, optionValues, stateOut);
	return res;
}

CUresult cuLinkAddData_v2(CUlinkState  state, CUjitInputType  type, void * data, size_t  size, const char * name, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
	return res;
}

CUresult cuLinkAddFile_v2(CUlinkState  state, CUjitInputType  type, const char * path, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
	return res;
}

CUresult cuLinkComplete(CUlinkState  state, void ** cubinOut, size_t * sizeOut)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLinkComplete(state, cubinOut, sizeOut);
	return res;
}

CUresult cuLinkDestroy(CUlinkState  state)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLinkDestroy(state);
	return res;
}

CUresult cuMemGetInfo_v2(size_t * free, size_t * total)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemGetInfo_v2(free, total);
	return res;
}

CUresult cuMemAlloc_v2(CUdeviceptr * dptr, size_t  bytesize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAlloc_v2(dptr, bytesize);
	return res;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr * dptr, size_t * pPitch, size_t  WidthInBytes, size_t  Height, unsigned int  ElementSizeBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
	return res;
}

CUresult cuMemFree_v2(CUdeviceptr  dptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemFree_v2(dptr);
	return res;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr * pbase, size_t * psize, CUdeviceptr  dptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemGetAddressRange_v2(pbase, psize, dptr);
	return res;
}

CUresult cuMemAllocHost_v2(void ** pp, size_t  bytesize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAllocHost_v2(pp, bytesize);
	return res;
}

CUresult cuMemFreeHost(void * p)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemFreeHost(p);
	return res;
}

CUresult cuMemHostAlloc(void ** pp, size_t  bytesize, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemHostAlloc(pp, bytesize, Flags);
	return res;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr * pdptr, void * p, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemHostGetDevicePointer_v2(pdptr, p, Flags);
	return res;
}

CUresult cuMemHostGetFlags(unsigned int * pFlags, void * p)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemHostGetFlags(pFlags, p);
	return res;
}

CUresult cuMemAllocManaged(CUdeviceptr * dptr, size_t  bytesize, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAllocManaged(dptr, bytesize, flags);
	return res;
}

CUresult cuDeviceGetByPCIBusId(CUdevice * dev, const char * pciBusId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetByPCIBusId(dev, pciBusId);
	return res;
}

CUresult cuDeviceGetPCIBusId(char * pciBusId, int  len, CUdevice  dev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetPCIBusId(pciBusId, len, dev);
	return res;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle * pHandle, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuIpcGetEventHandle(pHandle, event);
	return res;
}

CUresult cuIpcOpenEventHandle(CUevent * phEvent, CUipcEventHandle  handle)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuIpcOpenEventHandle(phEvent, handle);
	return res;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle * pHandle, CUdeviceptr  dptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuIpcGetMemHandle(pHandle, dptr);
	return res;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr * pdptr, CUipcMemHandle  handle, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuIpcOpenMemHandle_v2(pdptr, handle, Flags);
	return res;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr  dptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuIpcCloseMemHandle(dptr);
	return res;
}

CUresult cuMemHostRegister_v2(void * p, size_t  bytesize, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemHostRegister_v2(p, bytesize, Flags);
	return res;
}

CUresult cuMemHostUnregister(void * p)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemHostUnregister(p);
	return res;
}

CUresult cuMemcpy(CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy(dst, src, ByteCount);
	return res;
}

CUresult cuMemcpyPeer(CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
	return res;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
	return res;
}

CUresult cuMemcpyDtoH_v2(void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
	return res;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
	return res;
}

CUresult cuMemcpyDtoA_v2(CUarray  dstArray, size_t  dstOffset, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
	return res;
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr  dstDevice, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
	return res;
}

CUresult cuMemcpyHtoA_v2(CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
	return res;
}

CUresult cuMemcpyAtoH_v2(void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
	return res;
}

CUresult cuMemcpyAtoA_v2(CUarray  dstArray, size_t  dstOffset, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
	return res;
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D * pCopy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy2D_v2(pCopy);
	return res;
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * pCopy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy2DUnaligned_v2(pCopy);
	return res;
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D * pCopy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy3D_v2(pCopy);
	return res;
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER * pCopy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy3DPeer(pCopy);
	return res;
}

CUresult cuMemcpyAsync(CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyAsync(dst, src, ByteCount, hStream);
	return res;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
	return res;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
	return res;
}

CUresult cuMemcpyDtoHAsync_v2(void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
	return res;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
	return res;
}

CUresult cuMemcpyHtoAAsync_v2(CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
	return res;
}

CUresult cuMemcpyAtoHAsync_v2(void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
	return res;
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * pCopy, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy2DAsync_v2(pCopy, hStream);
	return res;
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * pCopy, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy3DAsync_v2(pCopy, hStream);
	return res;
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER * pCopy, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemcpy3DPeerAsync(pCopy, hStream);
	return res;
}

CUresult cuMemsetD8_v2(CUdeviceptr  dstDevice, unsigned char  uc, size_t  N)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD8_v2(dstDevice, uc, N);
	return res;
}

CUresult cuMemsetD16_v2(CUdeviceptr  dstDevice, unsigned short  us, size_t  N)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD16_v2(dstDevice, us, N);
	return res;
}

CUresult cuMemsetD32_v2(CUdeviceptr  dstDevice, unsigned int  ui, size_t  N)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD32_v2(dstDevice, ui, N);
	return res;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
	return res;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
	return res;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
	return res;
}

CUresult cuMemsetD8Async(CUdeviceptr  dstDevice, unsigned char  uc, size_t  N, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD8Async(dstDevice, uc, N, hStream);
	return res;
}

CUresult cuMemsetD16Async(CUdeviceptr  dstDevice, unsigned short  us, size_t  N, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD16Async(dstDevice, us, N, hStream);
	return res;
}

CUresult cuMemsetD32Async(CUdeviceptr  dstDevice, unsigned int  ui, size_t  N, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD32Async(dstDevice, ui, N, hStream);
	return res;
}

CUresult cuMemsetD2D8Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
	return res;
}

CUresult cuMemsetD2D16Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
	return res;
}

CUresult cuMemsetD2D32Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
	return res;
}

CUresult cuArrayCreate_v2(CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArrayCreate_v2(pHandle, pAllocateArray);
	return res;
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArrayGetDescriptor_v2(pArrayDescriptor, hArray);
	return res;
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray  array)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArrayGetSparseProperties(sparseProperties, array);
	return res;
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUmipmappedArray  mipmap)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
	return res;
}

CUresult cuArrayGetPlane(CUarray * pPlaneArray, CUarray  hArray, unsigned int  planeIdx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArrayGetPlane(pPlaneArray, hArray, planeIdx);
	return res;
}

CUresult cuArrayDestroy(CUarray  hArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArrayDestroy(hArray);
	return res;
}

CUresult cuArray3DCreate_v2(CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pAllocateArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArray3DCreate_v2(pHandle, pAllocateArray);
	return res;
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
	return res;
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int  numMipmapLevels)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
	return res;
}

CUresult cuMipmappedArrayGetLevel(CUarray * pLevelArray, CUmipmappedArray  hMipmappedArray, unsigned int  level)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
	return res;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray  hMipmappedArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMipmappedArrayDestroy(hMipmappedArray);
	return res;
}

CUresult cuMemAddressReserve(CUdeviceptr * ptr, size_t  size, size_t  alignment, CUdeviceptr  addr, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAddressReserve(ptr, size, alignment, addr, flags);
	return res;
}

CUresult cuMemAddressFree(CUdeviceptr  ptr, size_t  size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAddressFree(ptr, size);
	return res;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle * handle, size_t  size, const CUmemAllocationProp * prop, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemCreate(handle, size, prop, flags);
	return res;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle  handle)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemRelease(handle);
	return res;
}

CUresult cuMemMap(CUdeviceptr  ptr, size_t  size, size_t  offset, CUmemGenericAllocationHandle  handle, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemMap(ptr, size, offset, handle, flags);
	return res;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo * mapInfoList, unsigned int  count, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemMapArrayAsync(mapInfoList, count, hStream);
	return res;
}

CUresult cuMemUnmap(CUdeviceptr  ptr, size_t  size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemUnmap(ptr, size);
	return res;
}

CUresult cuMemSetAccess(CUdeviceptr  ptr, size_t  size, const CUmemAccessDesc * desc, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemSetAccess(ptr, size, desc, count);
	return res;
}

CUresult cuMemGetAccess(unsigned long long * flags, const CUmemLocation * location, CUdeviceptr  ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemGetAccess(flags, location, ptr);
	return res;
}

CUresult cuMemExportToShareableHandle(void * shareableHandle, CUmemGenericAllocationHandle  handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
	return res;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle * handle, void * osHandle, CUmemAllocationHandleType  shHandleType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemImportFromShareableHandle(handle, osHandle, shHandleType);
	return res;
}

CUresult cuMemGetAllocationGranularity(size_t * granularity, const CUmemAllocationProp * prop, CUmemAllocationGranularity_flags  option)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemGetAllocationGranularity(granularity, prop, option);
	return res;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp * prop, CUmemGenericAllocationHandle  handle)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemGetAllocationPropertiesFromHandle(prop, handle);
	return res;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle * handle, void * addr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemRetainAllocationHandle(handle, addr);
	return res;
}

CUresult cuMemFreeAsync(CUdeviceptr  dptr, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemFreeAsync(dptr, hStream);
	return res;
}

CUresult cuMemAllocAsync(CUdeviceptr * dptr, size_t  bytesize, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAllocAsync(dptr, bytesize, hStream);
	return res;
}

CUresult cuMemPoolTrimTo(CUmemoryPool  pool, size_t  minBytesToKeep)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolTrimTo(pool, minBytesToKeep);
	return res;
}

CUresult cuMemPoolSetAttribute(CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolSetAttribute(pool, attr, value);
	return res;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolGetAttribute(pool, attr, value);
	return res;
}

CUresult cuMemPoolSetAccess(CUmemoryPool  pool, const CUmemAccessDesc * map, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolSetAccess(pool, map, count);
	return res;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags * flags, CUmemoryPool  memPool, CUmemLocation * location)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolGetAccess(flags, memPool, location);
	return res;
}

CUresult cuMemPoolCreate(CUmemoryPool * pool, const CUmemPoolProps * poolProps)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolCreate(pool, poolProps);
	return res;
}

CUresult cuMemPoolDestroy(CUmemoryPool  pool)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolDestroy(pool);
	return res;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr * dptr, size_t  bytesize, CUmemoryPool  pool, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
	return res;
}

CUresult cuMemPoolExportToShareableHandle(void * handle_out, CUmemoryPool  pool, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags);
	return res;
}

CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool * pool_out, void * handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags);
	return res;
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData * shareData_out, CUdeviceptr  ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolExportPointer(shareData_out, ptr);
	return res;
}

CUresult cuMemPoolImportPointer(CUdeviceptr * ptr_out, CUmemoryPool  pool, CUmemPoolPtrExportData * shareData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPoolImportPointer(ptr_out, pool, shareData);
	return res;
}

CUresult cuPointerGetAttribute(void * data, CUpointer_attribute  attribute, CUdeviceptr  ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuPointerGetAttribute(data, attribute, ptr);
	return res;
}

CUresult cuMemPrefetchAsync(CUdeviceptr  devPtr, size_t  count, CUdevice  dstDevice, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
	return res;
}

CUresult cuMemAdvise(CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUdevice  device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemAdvise(devPtr, count, advice, device);
	return res;
}

CUresult cuMemRangeGetAttribute(void * data, size_t  dataSize, CUmem_range_attribute  attribute, CUdeviceptr  devPtr, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
	return res;
}

CUresult cuMemRangeGetAttributes(void ** data, size_t * dataSizes, CUmem_range_attribute * attributes, size_t  numAttributes, CUdeviceptr  devPtr, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
	return res;
}

CUresult cuPointerSetAttribute(const void * value, CUpointer_attribute  attribute, CUdeviceptr  ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuPointerSetAttribute(value, attribute, ptr);
	return res;
}

CUresult cuPointerGetAttributes(unsigned int  numAttributes, CUpointer_attribute * attributes, void ** data, CUdeviceptr  ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuPointerGetAttributes(numAttributes, attributes, data, ptr);
	return res;
}

CUresult cuStreamCreate(CUstream * phStream, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamCreate(phStream, Flags);
	return res;
}

CUresult cuStreamCreateWithPriority(CUstream * phStream, unsigned int  flags, int  priority)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamCreateWithPriority(phStream, flags, priority);
	return res;
}

CUresult cuStreamGetPriority(CUstream  hStream, int * priority)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamGetPriority(hStream, priority);
	return res;
}

CUresult cuStreamGetFlags(CUstream  hStream, unsigned int * flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamGetFlags(hStream, flags);
	return res;
}

CUresult cuStreamGetCtx(CUstream  hStream, CUcontext * pctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamGetCtx(hStream, pctx);
	return res;
}

CUresult cuStreamWaitEvent(CUstream  hStream, CUevent  hEvent, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamWaitEvent(hStream, hEvent, Flags);
	return res;
}

CUresult cuStreamAddCallback(CUstream  hStream, CUstreamCallback  callback, void * userData, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamAddCallback(hStream, callback, userData, flags);
	return res;
}

CUresult cuStreamBeginCapture_v2(CUstream  hStream, CUstreamCaptureMode  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamBeginCapture_v2(hStream, mode);
	return res;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode * mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuThreadExchangeStreamCaptureMode(mode);
	return res;
}

CUresult cuStreamEndCapture(CUstream  hStream, CUgraph * phGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamEndCapture(hStream, phGraph);
	return res;
}

CUresult cuStreamIsCapturing(CUstream  hStream, CUstreamCaptureStatus * captureStatus)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamIsCapturing(hStream, captureStatus);
	return res;
}

CUresult cuStreamGetCaptureInfo(CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamGetCaptureInfo(hStream, captureStatus_out, id_out);
	return res;
}

CUresult cuStreamGetCaptureInfo_v2(CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, CUgraph * graph_out, const CUgraphNode ** dependencies_out, size_t * numDependencies_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
	return res;
}

CUresult cuStreamUpdateCaptureDependencies(CUstream  hStream, CUgraphNode * dependencies, size_t  numDependencies, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags);
	return res;
}

CUresult cuStreamAttachMemAsync(CUstream  hStream, CUdeviceptr  dptr, size_t  length, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamAttachMemAsync(hStream, dptr, length, flags);
	return res;
}

CUresult cuStreamQuery(CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamQuery(hStream);
	return res;
}

CUresult cuStreamSynchronize(CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamSynchronize(hStream);
	return res;
}

CUresult cuStreamDestroy_v2(CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamDestroy_v2(hStream);
	return res;
}

CUresult cuStreamCopyAttributes(CUstream  dst, CUstream  src)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamCopyAttributes(dst, src);
	return res;
}

CUresult cuStreamGetAttribute(CUstream  hStream, CUstreamAttrID  attr, CUstreamAttrValue * value_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamGetAttribute(hStream, attr, value_out);
	return res;
}

CUresult cuStreamSetAttribute(CUstream  hStream, CUstreamAttrID  attr, const CUstreamAttrValue * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamSetAttribute(hStream, attr, value);
	return res;
}

CUresult cuEventCreate(CUevent * phEvent, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventCreate(phEvent, Flags);
	return res;
}

CUresult cuEventRecord(CUevent  hEvent, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventRecord(hEvent, hStream);
	return res;
}

CUresult cuEventRecordWithFlags(CUevent  hEvent, CUstream  hStream, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventRecordWithFlags(hEvent, hStream, flags);
	return res;
}

CUresult cuEventQuery(CUevent  hEvent)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventQuery(hEvent);
	return res;
}

CUresult cuEventSynchronize(CUevent  hEvent)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventSynchronize(hEvent);
	return res;
}

CUresult cuEventDestroy_v2(CUevent  hEvent)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventDestroy_v2(hEvent);
	return res;
}

CUresult cuEventElapsedTime(float * pMilliseconds, CUevent  hStart, CUevent  hEnd)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuEventElapsedTime(pMilliseconds, hStart, hEnd);
	return res;
}

CUresult cuImportExternalMemory(CUexternalMemory * extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuImportExternalMemory(extMem_out, memHandleDesc);
	return res;
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr * devPtr, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
	return res;
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray * mipmap, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
	return res;
}

CUresult cuDestroyExternalMemory(CUexternalMemory  extMem)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDestroyExternalMemory(extMem);
	return res;
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore * extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuImportExternalSemaphore(extSem_out, semHandleDesc);
	return res;
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
	return res;
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
	return res;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore  extSem)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDestroyExternalSemaphore(extSem);
	return res;
}

CUresult cuStreamWaitValue32(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamWaitValue32(stream, addr, value, flags);
	return res;
}

CUresult cuStreamWaitValue64(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamWaitValue64(stream, addr, value, flags);
	return res;
}

CUresult cuStreamWriteValue32(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamWriteValue32(stream, addr, value, flags);
	return res;
}

CUresult cuStreamWriteValue64(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamWriteValue64(stream, addr, value, flags);
	return res;
}

CUresult cuStreamBatchMemOp(CUstream  stream, unsigned int  count, CUstreamBatchMemOpParams * paramArray, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuStreamBatchMemOp(stream, count, paramArray, flags);
	return res;
}

CUresult cuFuncGetAttribute(int * pi, CUfunction_attribute  attrib, CUfunction  hfunc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncGetAttribute(pi, attrib, hfunc);
	return res;
}

CUresult cuFuncSetAttribute(CUfunction  hfunc, CUfunction_attribute  attrib, int  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncSetAttribute(hfunc, attrib, value);
	return res;
}

CUresult cuFuncSetCacheConfig(CUfunction  hfunc, CUfunc_cache  config)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncSetCacheConfig(hfunc, config);
	return res;
}

CUresult cuFuncSetSharedMemConfig(CUfunction  hfunc, CUsharedconfig  config)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncSetSharedMemConfig(hfunc, config);
	return res;
}

CUresult cuFuncGetModule(CUmodule * hmod, CUfunction  hfunc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncGetModule(hmod, hfunc);
	return res;
}

CUresult cuLaunchKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
	return res;
}

CUresult cuLaunchCooperativeKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
	return res;
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS * launchParamsList, unsigned int  numDevices, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
	return res;
}

CUresult cuLaunchHostFunc(CUstream  hStream, CUhostFn  fn, void * userData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunchHostFunc(hStream, fn, userData);
	return res;
}

CUresult cuFuncSetBlockShape(CUfunction  hfunc, int  x, int  y, int  z)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncSetBlockShape(hfunc, x, y, z);
	return res;
}

CUresult cuFuncSetSharedSize(CUfunction  hfunc, unsigned int  bytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuFuncSetSharedSize(hfunc, bytes);
	return res;
}

CUresult cuParamSetSize(CUfunction  hfunc, unsigned int  numbytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuParamSetSize(hfunc, numbytes);
	return res;
}

CUresult cuParamSeti(CUfunction  hfunc, int  offset, unsigned int  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuParamSeti(hfunc, offset, value);
	return res;
}

CUresult cuParamSetf(CUfunction  hfunc, int  offset, float  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuParamSetf(hfunc, offset, value);
	return res;
}

CUresult cuParamSetv(CUfunction  hfunc, int  offset, void * ptr, unsigned int  numbytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuParamSetv(hfunc, offset, ptr, numbytes);
	return res;
}

CUresult cuLaunch(CUfunction  f)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunch(f);
	return res;
}

CUresult cuLaunchGrid(CUfunction  f, int  grid_width, int  grid_height)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunchGrid(f, grid_width, grid_height);
	return res;
}

CUresult cuLaunchGridAsync(CUfunction  f, int  grid_width, int  grid_height, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuLaunchGridAsync(f, grid_width, grid_height, hStream);
	return res;
}

CUresult cuParamSetTexRef(CUfunction  hfunc, int  texunit, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuParamSetTexRef(hfunc, texunit, hTexRef);
	return res;
}

CUresult cuGraphCreate(CUgraph * phGraph, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphCreate(phGraph, flags);
	return res;
}

CUresult cuGraphAddKernelNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
	return res;
}

CUresult cuGraphKernelNodeGetParams(CUgraphNode  hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphKernelNodeGetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphKernelNodeSetParams(CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphKernelNodeSetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphAddMemcpyNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
	return res;
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode  hNode, CUDA_MEMCPY3D * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphMemcpyNodeGetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode  hNode, const CUDA_MEMCPY3D * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphMemcpyNodeSetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphAddMemsetNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
	return res;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode  hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphMemsetNodeGetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphMemsetNodeSetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphAddHostNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
	return res;
}

CUresult cuGraphHostNodeGetParams(CUgraphNode  hNode, CUDA_HOST_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphHostNodeGetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphHostNodeSetParams(CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphHostNodeSetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphAddChildGraphNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraph  childGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
	return res;
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode  hNode, CUgraph * phGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphChildGraphNodeGetGraph(hNode, phGraph);
	return res;
}

CUresult cuGraphAddEmptyNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies);
	return res;
}

CUresult cuGraphAddEventRecordNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event);
	return res;
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode  hNode, CUevent * event_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphEventRecordNodeGetEvent(hNode, event_out);
	return res;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode  hNode, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphEventRecordNodeSetEvent(hNode, event);
	return res;
}

CUresult cuGraphAddEventWaitNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event);
	return res;
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode  hNode, CUevent * event_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphEventWaitNodeGetEvent(hNode, event_out);
	return res;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode  hNode, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphEventWaitNodeSetEvent(hNode, event);
	return res;
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
	return res;
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode  hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * params_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
	return res;
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
	return res;
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode  hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS * params_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
	return res;
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
	return res;
}

CUresult cuGraphAddMemAllocNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
	return res;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode  hNode, CUDA_MEM_ALLOC_NODE_PARAMS * params_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphMemAllocNodeGetParams(hNode, params_out);
	return res;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUdeviceptr  dptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr);
	return res;
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode  hNode, CUdeviceptr * dptr_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphMemFreeNodeGetParams(hNode, dptr_out);
	return res;
}

CUresult cuDeviceGraphMemTrim(CUdevice  device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGraphMemTrim(device);
	return res;
}

CUresult cuDeviceGetGraphMemAttribute(CUdevice  device, CUgraphMem_attribute  attr, void*  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetGraphMemAttribute(device, attr, value);
	return res;
}

CUresult cuDeviceSetGraphMemAttribute(CUdevice  device, CUgraphMem_attribute  attr, void*  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceSetGraphMemAttribute(device, attr, value);
	return res;
}

CUresult cuGraphClone(CUgraph * phGraphClone, CUgraph  originalGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphClone(phGraphClone, originalGraph);
	return res;
}

CUresult cuGraphNodeFindInClone(CUgraphNode * phNode, CUgraphNode  hOriginalNode, CUgraph  hClonedGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph);
	return res;
}

CUresult cuGraphNodeGetType(CUgraphNode  hNode, CUgraphNodeType * type)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphNodeGetType(hNode, type);
	return res;
}

CUresult cuGraphGetNodes(CUgraph  hGraph, CUgraphNode * nodes, size_t * numNodes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphGetNodes(hGraph, nodes, numNodes);
	return res;
}

CUresult cuGraphGetRootNodes(CUgraph  hGraph, CUgraphNode * rootNodes, size_t * numRootNodes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphGetRootNodes(hGraph, rootNodes, numRootNodes);
	return res;
}

CUresult cuGraphGetEdges(CUgraph  hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphGetEdges(hGraph, from, to, numEdges);
	return res;
}

CUresult cuGraphNodeGetDependencies(CUgraphNode  hNode, CUgraphNode * dependencies, size_t * numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphNodeGetDependencies(hNode, dependencies, numDependencies);
	return res;
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode  hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
	return res;
}

CUresult cuGraphAddDependencies(CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphAddDependencies(hGraph, from, to, numDependencies);
	return res;
}

CUresult cuGraphRemoveDependencies(CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphRemoveDependencies(hGraph, from, to, numDependencies);
	return res;
}

CUresult cuGraphDestroyNode(CUgraphNode  hNode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphDestroyNode(hNode);
	return res;
}

CUresult cuGraphInstantiate_v2(CUgraphExec * phGraphExec, CUgraph  hGraph, CUgraphNode * phErrorNode, char * logBuffer, size_t  bufferSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
	return res;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec * phGraphExec, CUgraph  hGraph, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
	return res;
}

CUresult cuGraphExecKernelNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams);
	return res;
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx);
	return res;
}

CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx);
	return res;
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams);
	return res;
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraph  childGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);
	return res;
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
	return res;
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
	return res;
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
	return res;
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
	return res;
}

CUresult cuGraphUpload(CUgraphExec  hGraphExec, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphUpload(hGraphExec, hStream);
	return res;
}

CUresult cuGraphLaunch(CUgraphExec  hGraphExec, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphLaunch(hGraphExec, hStream);
	return res;
}

CUresult cuGraphExecDestroy(CUgraphExec  hGraphExec)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecDestroy(hGraphExec);
	return res;
}

CUresult cuGraphDestroy(CUgraph  hGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphDestroy(hGraph);
	return res;
}

CUresult cuGraphExecUpdate(CUgraphExec  hGraphExec, CUgraph  hGraph, CUgraphNode * hErrorNode_out, CUgraphExecUpdateResult * updateResult_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
	return res;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode  dst, CUgraphNode  src)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphKernelNodeCopyAttributes(dst, src);
	return res;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode  hNode, CUkernelNodeAttrID  attr, CUkernelNodeAttrValue * value_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphKernelNodeGetAttribute(hNode, attr, value_out);
	return res;
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode  hNode, CUkernelNodeAttrID  attr, const CUkernelNodeAttrValue * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphKernelNodeSetAttribute(hNode, attr, value);
	return res;
}

CUresult cuGraphDebugDotPrint(CUgraph  hGraph, const char * path, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphDebugDotPrint(hGraph, path, flags);
	return res;
}

CUresult cuUserObjectCreate(CUuserObject * object_out, void * ptr, CUhostFn  destroy, unsigned int  initialRefcount, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
	return res;
}

CUresult cuUserObjectRetain(CUuserObject  object, unsigned int  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuUserObjectRetain(object, count);
	return res;
}

CUresult cuUserObjectRelease(CUuserObject  object, unsigned int  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuUserObjectRelease(object, count);
	return res;
}

CUresult cuGraphRetainUserObject(CUgraph  graph, CUuserObject  object, unsigned int  count, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphRetainUserObject(graph, object, count, flags);
	return res;
}

CUresult cuGraphReleaseUserObject(CUgraph  graph, CUuserObject  object, unsigned int  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphReleaseUserObject(graph, object, count);
	return res;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
	return res;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
	return res;
}

CUresult cuOccupancyMaxPotentialBlockSize(int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
	return res;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
	return res;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, CUfunction  func, int  numBlocks, int  blockSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
	return res;
}

CUresult cuTexRefSetArray(CUtexref  hTexRef, CUarray  hArray, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetArray(hTexRef, hArray, Flags);
	return res;
}

CUresult cuTexRefSetMipmappedArray(CUtexref  hTexRef, CUmipmappedArray  hMipmappedArray, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);
	return res;
}

CUresult cuTexRefSetAddress_v2(size_t * ByteOffset, CUtexref  hTexRef, CUdeviceptr  dptr, size_t  bytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
	return res;
}

CUresult cuTexRefSetAddress2D_v3(CUtexref  hTexRef, const CUDA_ARRAY_DESCRIPTOR * desc, CUdeviceptr  dptr, size_t  Pitch)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch);
	return res;
}

CUresult cuTexRefSetFormat(CUtexref  hTexRef, CUarray_format  fmt, int  NumPackedComponents)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
	return res;
}

CUresult cuTexRefSetAddressMode(CUtexref  hTexRef, int  dim, CUaddress_mode  am)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetAddressMode(hTexRef, dim, am);
	return res;
}

CUresult cuTexRefSetFilterMode(CUtexref  hTexRef, CUfilter_mode  fm)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetFilterMode(hTexRef, fm);
	return res;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref  hTexRef, CUfilter_mode  fm)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetMipmapFilterMode(hTexRef, fm);
	return res;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref  hTexRef, float  bias)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetMipmapLevelBias(hTexRef, bias);
	return res;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref  hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
	return res;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref  hTexRef, unsigned int  maxAniso)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetMaxAnisotropy(hTexRef, maxAniso);
	return res;
}

CUresult cuTexRefSetBorderColor(CUtexref  hTexRef, float * pBorderColor)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetBorderColor(hTexRef, pBorderColor);
	return res;
}

CUresult cuTexRefSetFlags(CUtexref  hTexRef, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefSetFlags(hTexRef, Flags);
	return res;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr * pdptr, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetAddress_v2(pdptr, hTexRef);
	return res;
}

CUresult cuTexRefGetArray(CUarray * phArray, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetArray(phArray, hTexRef);
	return res;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray * phMipmappedArray, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetMipmappedArray(phMipmappedArray, hTexRef);
	return res;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode * pam, CUtexref  hTexRef, int  dim)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetAddressMode(pam, hTexRef, dim);
	return res;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode * pfm, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetFilterMode(pfm, hTexRef);
	return res;
}

CUresult cuTexRefGetFormat(CUarray_format * pFormat, int * pNumChannels, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetFormat(pFormat, pNumChannels, hTexRef);
	return res;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode * pfm, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetMipmapFilterMode(pfm, hTexRef);
	return res;
}

CUresult cuTexRefGetMipmapLevelBias(float * pbias, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetMipmapLevelBias(pbias, hTexRef);
	return res;
}

CUresult cuTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
	return res;
}

CUresult cuTexRefGetMaxAnisotropy(int * pmaxAniso, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef);
	return res;
}

CUresult cuTexRefGetBorderColor(float * pBorderColor, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetBorderColor(pBorderColor, hTexRef);
	return res;
}

CUresult cuTexRefGetFlags(unsigned int * pFlags, CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefGetFlags(pFlags, hTexRef);
	return res;
}

CUresult cuTexRefCreate(CUtexref * pTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefCreate(pTexRef);
	return res;
}

CUresult cuTexRefDestroy(CUtexref  hTexRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexRefDestroy(hTexRef);
	return res;
}

CUresult cuSurfRefSetArray(CUsurfref  hSurfRef, CUarray  hArray, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuSurfRefSetArray(hSurfRef, hArray, Flags);
	return res;
}

CUresult cuSurfRefGetArray(CUarray * phArray, CUsurfref  hSurfRef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuSurfRefGetArray(phArray, hSurfRef);
	return res;
}

CUresult cuTexObjectCreate(CUtexObject * pTexObject, const CUDA_RESOURCE_DESC * pResDesc, const CUDA_TEXTURE_DESC * pTexDesc, const CUDA_RESOURCE_VIEW_DESC * pResViewDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
	return res;
}

CUresult cuTexObjectDestroy(CUtexObject  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexObjectDestroy(texObject);
	return res;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUtexObject  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexObjectGetResourceDesc(pResDesc, texObject);
	return res;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC * pTexDesc, CUtexObject  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexObjectGetTextureDesc(pTexDesc, texObject);
	return res;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuTexObjectGetResourceViewDesc(pResViewDesc, texObject);
	return res;
}

CUresult cuSurfObjectCreate(CUsurfObject * pSurfObject, const CUDA_RESOURCE_DESC * pResDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuSurfObjectCreate(pSurfObject, pResDesc);
	return res;
}

CUresult cuSurfObjectDestroy(CUsurfObject  surfObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuSurfObjectDestroy(surfObject);
	return res;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUsurfObject  surfObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuSurfObjectGetResourceDesc(pResDesc, surfObject);
	return res;
}

CUresult cuDeviceCanAccessPeer(int * canAccessPeer, CUdevice  dev, CUdevice  peerDev)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceCanAccessPeer(canAccessPeer, dev, peerDev);
	return res;
}

CUresult cuCtxEnablePeerAccess(CUcontext  peerContext, unsigned int  Flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxEnablePeerAccess(peerContext, Flags);
	return res;
}

CUresult cuCtxDisablePeerAccess(CUcontext  peerContext)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuCtxDisablePeerAccess(peerContext);
	return res;
}

CUresult cuDeviceGetP2PAttribute(int*  value, CUdevice_P2PAttribute  attrib, CUdevice  srcDevice, CUdevice  dstDevice)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice);
	return res;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource  resource)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsUnregisterResource(resource);
	return res;
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray * pArray, CUgraphicsResource  resource, unsigned int  arrayIndex, unsigned int  mipLevel)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
	return res;
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray * pMipmappedArray, CUgraphicsResource  resource)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource);
	return res;
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource  resource)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource);
	return res;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource  resource, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsResourceSetMapFlags_v2(resource, flags);
	return res;
}

CUresult cuGraphicsMapResources(unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsMapResources(count, resources, hStream);
	return res;
}

CUresult cuGraphicsUnmapResources(unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGraphicsUnmapResources(count, resources, hStream);
	return res;
}

CUresult cuGetExportTable(const void ** ppExportTable, const CUuuid * pExportTableId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	CUresult res = 
		lcuGetExportTable(ppExportTable, pExportTableId);
	return res;
}

cudaError_t cudaDeviceReset()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceResetArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICERESET;
    
    struct cudaDeviceResetArg *arg_ptr = (struct cudaDeviceResetArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceSynchronize()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceSynchronizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICESYNCHRONIZE;
    
    struct cudaDeviceSynchronizeArg *arg_ptr = (struct cudaDeviceSynchronizeArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit  limit, size_t  value)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceSetLimitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICESETLIMIT;
    
    struct cudaDeviceSetLimitArg *arg_ptr = (struct cudaDeviceSetLimitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->limit = limit;
	arg_ptr->value = value;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceGetLimit(size_t * pValue, enum cudaLimit  limit)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetLimitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETLIMIT;
    
    struct cudaDeviceGetLimitArg *arg_ptr = (struct cudaDeviceGetLimitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pValue = pValue;
	arg_ptr->limit = limit;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetLimitResponse *) dat;
	*pValue = res->pValue;
return res->err;
}

cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const struct cudaChannelFormatDesc * fmtDesc, int  device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device);
	return res;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETCACHECONFIG;
    
    struct cudaDeviceGetCacheConfigArg *arg_ptr = (struct cudaDeviceGetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pCacheConfig = pCacheConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetCacheConfigResponse *) dat;
	*pCacheConfig = res->pCacheConfig;
return res->err;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetStreamPriorityRangeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETSTREAMPRIORITYRANGE;
    
    struct cudaDeviceGetStreamPriorityRangeArg *arg_ptr = (struct cudaDeviceGetStreamPriorityRangeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->leastPriority = leastPriority;
	arg_ptr->greatestPriority = greatestPriority;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetStreamPriorityRangeResponse *) dat;
	*leastPriority = res->leastPriority;
	*greatestPriority = res->greatestPriority;
return res->err;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache  cacheConfig)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceSetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICESETCACHECONFIG;
    
    struct cudaDeviceSetCacheConfigArg *arg_ptr = (struct cudaDeviceSetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->cacheConfig = cacheConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGetSharedMemConfig(pConfig);
	return res;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig  config)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceSetSharedMemConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICESETSHAREDMEMCONFIG;
    
    struct cudaDeviceSetSharedMemConfigArg *arg_ptr = (struct cudaDeviceSetSharedMemConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->config = config;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGetByPCIBusId(device, pciBusId);
	return res;
}

cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int  len, int  device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGetPCIBusId(pciBusId, len, device);
	return res;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaIpcGetEventHandle(handle, event);
	return res;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t  handle)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaIpcOpenEventHandleArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAIPCOPENEVENTHANDLE;
    
    struct cudaIpcOpenEventHandleArg *arg_ptr = (struct cudaIpcOpenEventHandleArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	arg_ptr->handle = handle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaIpcOpenEventHandleResponse *) dat;
	*event = res->event;
return res->err;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaIpcGetMemHandle(handle, devPtr);
	return res;
}

cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t  handle, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaIpcOpenMemHandle(devPtr, handle, flags);
	return res;
}

cudaError_t cudaIpcCloseMemHandle(void * devPtr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaIpcCloseMemHandle(devPtr);
	return res;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget  target, enum cudaFlushGPUDirectRDMAWritesScope  scope)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceFlushGPUDirectRDMAWrites(target, scope);
	return res;
}

cudaError_t cudaThreadExit()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadExitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADEXIT;
    
    struct cudaThreadExitArg *arg_ptr = (struct cudaThreadExitArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaThreadSynchronize()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadSynchronizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADSYNCHRONIZE;
    
    struct cudaThreadSynchronizeArg *arg_ptr = (struct cudaThreadSynchronizeArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit  limit, size_t  value)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadSetLimitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADSETLIMIT;
    
    struct cudaThreadSetLimitArg *arg_ptr = (struct cudaThreadSetLimitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->limit = limit;
	arg_ptr->value = value;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaThreadGetLimit(size_t * pValue, enum cudaLimit  limit)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadGetLimitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADGETLIMIT;
    
    struct cudaThreadGetLimitArg *arg_ptr = (struct cudaThreadGetLimitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pValue = pValue;
	arg_ptr->limit = limit;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaThreadGetLimitResponse *) dat;
	*pValue = res->pValue;
return res->err;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadGetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADGETCACHECONFIG;
    
    struct cudaThreadGetCacheConfigArg *arg_ptr = (struct cudaThreadGetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pCacheConfig = pCacheConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaThreadGetCacheConfigResponse *) dat;
	*pCacheConfig = res->pCacheConfig;
return res->err;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache  cacheConfig)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadSetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADSETCACHECONFIG;
    
    struct cudaThreadSetCacheConfigArg *arg_ptr = (struct cudaThreadSetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->cacheConfig = cacheConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaGetLastError()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetLastErrorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETLASTERROR;
    
    struct cudaGetLastErrorArg *arg_ptr = (struct cudaGetLastErrorArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaPeekAtLastError()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaPeekAtLastErrorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAPEEKATLASTERROR;
    
    struct cudaPeekAtLastErrorArg *arg_ptr = (struct cudaPeekAtLastErrorArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

const char* cudaGetErrorName(cudaError_t  error)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	const char* res = 
		lcudaGetErrorName(error);
	return res;
}

cudaError_t cudaGetDeviceCount(int * count)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDeviceCountArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICECOUNT;
    
    struct cudaGetDeviceCountArg *arg_ptr = (struct cudaGetDeviceCountArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->count = count;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDeviceCountResponse *) dat;
	*count = res->count;
return res->err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int  device)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDevicePropertiesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICEPROPERTIES;
    
    struct cudaGetDevicePropertiesArg *arg_ptr = (struct cudaGetDevicePropertiesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->prop = prop;
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDevicePropertiesResponse *) dat;
	*prop = res->prop;
return res->err;
}

cudaError_t cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr  attr, int  device)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetAttributeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETATTRIBUTE;
    
    struct cudaDeviceGetAttributeArg *arg_ptr = (struct cudaDeviceGetAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->value = value;
	arg_ptr->attr = attr;
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetAttributeResponse *) dat;
	*value = res->value;
return res->err;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int  device)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetDefaultMemPoolArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETDEFAULTMEMPOOL;
    
    struct cudaDeviceGetDefaultMemPoolArg *arg_ptr = (struct cudaDeviceGetDefaultMemPoolArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->memPool = memPool;
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetDefaultMemPoolResponse *) dat;
	*memPool = res->memPool;
return res->err;
}

cudaError_t cudaDeviceSetMemPool(int  device, cudaMemPool_t  memPool)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceSetMemPoolArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICESETMEMPOOL;
    
    struct cudaDeviceSetMemPoolArg *arg_ptr = (struct cudaDeviceSetMemPoolArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->device = device;
	arg_ptr->memPool = memPool;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int  device)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetMemPoolArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETMEMPOOL;
    
    struct cudaDeviceGetMemPoolArg *arg_ptr = (struct cudaDeviceGetMemPoolArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->memPool = memPool;
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetMemPoolResponse *) dat;
	*memPool = res->memPool;
return res->err;
}

cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int  device, int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags);
	return res;
}

cudaError_t cudaDeviceGetP2PAttribute(int * value, enum cudaDeviceP2PAttr  attr, int  srcDevice, int  dstDevice)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetP2PAttributeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETP2PATTRIBUTE;
    
    struct cudaDeviceGetP2PAttributeArg *arg_ptr = (struct cudaDeviceGetP2PAttributeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->value = value;
	arg_ptr->attr = attr;
	arg_ptr->srcDevice = srcDevice;
	arg_ptr->dstDevice = dstDevice;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetP2PAttributeResponse *) dat;
	*value = res->value;
return res->err;
}

cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaChooseDevice(device, prop);
	return res;
}

cudaError_t cudaSetDevice(int  device)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaSetDeviceArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASETDEVICE;
    
    struct cudaSetDeviceArg *arg_ptr = (struct cudaSetDeviceArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaGetDevice(int * device)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDeviceArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICE;
    
    struct cudaGetDeviceArg *arg_ptr = (struct cudaGetDeviceArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDeviceResponse *) dat;
	*device = res->device;
return res->err;
}

cudaError_t cudaSetValidDevices(int * device_arr, int  len)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaSetValidDevices(device_arr, len);
	return res;
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaSetDeviceFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASETDEVICEFLAGS;
    
    struct cudaSetDeviceFlagsArg *arg_ptr = (struct cudaSetDeviceFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaGetDeviceFlags(unsigned int * flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDeviceFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICEFLAGS;
    
    struct cudaGetDeviceFlagsArg *arg_ptr = (struct cudaGetDeviceFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDeviceFlagsResponse *) dat;
	*flags = res->flags;
return res->err;
}

cudaError_t cudaStreamCreate(cudaStream_t * pStream)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMCREATE;
    
    struct cudaStreamCreateArg *arg_ptr = (struct cudaStreamCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pStream = pStream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamCreateResponse *) dat;
	*pStream = res->pStream;
return res->err;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamCreateWithFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMCREATEWITHFLAGS;
    
    struct cudaStreamCreateWithFlagsArg *arg_ptr = (struct cudaStreamCreateWithFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pStream = pStream;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamCreateWithFlagsResponse *) dat;
	*pStream = res->pStream;
return res->err;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned int  flags, int  priority)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamCreateWithPriorityArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMCREATEWITHPRIORITY;
    
    struct cudaStreamCreateWithPriorityArg *arg_ptr = (struct cudaStreamCreateWithPriorityArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pStream = pStream;
	arg_ptr->flags = flags;
	arg_ptr->priority = priority;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamCreateWithPriorityResponse *) dat;
	*pStream = res->pStream;
return res->err;
}

cudaError_t cudaStreamGetPriority(cudaStream_t  hStream, int * priority)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamGetPriority(hStream, priority);
	return res;
}

cudaError_t cudaStreamGetFlags(cudaStream_t  hStream, unsigned int * flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamGetFlags(hStream, flags);
	return res;
}

cudaError_t cudaCtxResetPersistingL2Cache()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaCtxResetPersistingL2CacheArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDACTXRESETPERSISTINGL2CACHE;
    
    struct cudaCtxResetPersistingL2CacheArg *arg_ptr = (struct cudaCtxResetPersistingL2CacheArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t  dst, cudaStream_t  src)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamCopyAttributesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMCOPYATTRIBUTES;
    
    struct cudaStreamCopyAttributesArg *arg_ptr = (struct cudaStreamCopyAttributesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dst = dst;
	arg_ptr->src = src;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t  hStream, enum cudaStreamAttrID  attr, union cudaStreamAttrValue * value_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamGetAttribute(hStream, attr, value_out);
	return res;
}

cudaError_t cudaStreamSetAttribute(cudaStream_t  hStream, enum cudaStreamAttrID  attr, const union cudaStreamAttrValue * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamSetAttribute(hStream, attr, value);
	return res;
}

cudaError_t cudaStreamDestroy(cudaStream_t  stream)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMDESTROY;
    
    struct cudaStreamDestroyArg *arg_ptr = (struct cudaStreamDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t  stream, cudaEvent_t  event, unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamWaitEventArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMWAITEVENT;
    
    struct cudaStreamWaitEventArg *arg_ptr = (struct cudaStreamWaitEventArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	arg_ptr->event = event;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamAddCallback(cudaStream_t  stream, cudaStreamCallback_t  callback, void * userData, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamAddCallback(stream, callback, userData, flags);
	return res;
}

cudaError_t cudaStreamSynchronize(cudaStream_t  stream)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamSynchronizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMSYNCHRONIZE;
    
    struct cudaStreamSynchronizeArg *arg_ptr = (struct cudaStreamSynchronizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamQuery(cudaStream_t  stream)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamQueryArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMQUERY;
    
    struct cudaStreamQueryArg *arg_ptr = (struct cudaStreamQueryArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t  stream, void * devPtr, size_t  length, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamAttachMemAsync(stream, devPtr, length, flags);
	return res;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t  stream, enum cudaStreamCaptureMode  mode)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamBeginCaptureArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMBEGINCAPTURE;
    
    struct cudaStreamBeginCaptureArg *arg_ptr = (struct cudaStreamBeginCaptureArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	arg_ptr->mode = mode;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode * mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaThreadExchangeStreamCaptureMode(mode);
	return res;
}

cudaError_t cudaStreamEndCapture(cudaStream_t  stream, cudaGraph_t * pGraph)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamEndCaptureArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMENDCAPTURE;
    
    struct cudaStreamEndCaptureArg *arg_ptr = (struct cudaStreamEndCaptureArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	arg_ptr->pGraph = pGraph;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamIsCapturingArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMISCAPTURING;
    
    struct cudaStreamIsCapturingArg *arg_ptr = (struct cudaStreamIsCapturingArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->stream = stream;
	arg_ptr->pCaptureStatus = pCaptureStatus;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamIsCapturingResponse *) dat;
	*pCaptureStatus = res->pCaptureStatus;
return res->err;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus, unsigned long long * pId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamGetCaptureInfo(stream, pCaptureStatus, pId);
	return res;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t  stream, enum cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, cudaGraph_t * graph_out, const cudaGraphNode_t ** dependencies_out, size_t * numDependencies_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
	return res;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t  stream, cudaGraphNode_t * dependencies, size_t  numDependencies, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
	return res;
}

cudaError_t cudaEventCreate(cudaEvent_t * event)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTCREATE;
    
    struct cudaEventCreateArg *arg_ptr = (struct cudaEventCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaEventCreateResponse *) dat;
	*event = res->event;
return res->err;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventCreateWithFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTCREATEWITHFLAGS;
    
    struct cudaEventCreateWithFlagsArg *arg_ptr = (struct cudaEventCreateWithFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaEventCreateWithFlagsResponse *) dat;
	*event = res->event;
return res->err;
}

cudaError_t cudaEventRecord(cudaEvent_t  event, cudaStream_t  stream)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventRecordArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTRECORD;
    
    struct cudaEventRecordArg *arg_ptr = (struct cudaEventRecordArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	arg_ptr->stream = stream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t  event, cudaStream_t  stream, unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventRecordWithFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTRECORDWITHFLAGS;
    
    struct cudaEventRecordWithFlagsArg *arg_ptr = (struct cudaEventRecordWithFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	arg_ptr->stream = stream;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaEventQuery(cudaEvent_t  event)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventQueryArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTQUERY;
    
    struct cudaEventQueryArg *arg_ptr = (struct cudaEventQueryArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaEventSynchronize(cudaEvent_t  event)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventSynchronizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTSYNCHRONIZE;
    
    struct cudaEventSynchronizeArg *arg_ptr = (struct cudaEventSynchronizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaEventDestroy(cudaEvent_t  event)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTDESTROY;
    
    struct cudaEventDestroyArg *arg_ptr = (struct cudaEventDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t  start, cudaEvent_t  end)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventElapsedTimeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTELAPSEDTIME;
    
    struct cudaEventElapsedTimeArg *arg_ptr = (struct cudaEventElapsedTimeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->ms = ms;
	arg_ptr->start = start;
	arg_ptr->end = end;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaEventElapsedTimeResponse *) dat;
	*ms = res->ms;
return res->err;
}

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const struct cudaExternalMemoryHandleDesc * memHandleDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaImportExternalMemory(extMem_out, memHandleDesc);
	return res;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryBufferDesc * bufferDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
	return res;
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryMipmappedArrayDesc * mipmapDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
	return res;
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t  extMem)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDestroyExternalMemory(extMem);
	return res;
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const struct cudaExternalSemaphoreHandleDesc * semHandleDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaImportExternalSemaphore(extSem_out, semHandleDesc);
	return res;
}

cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreSignalParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
	return res;
}

cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreWaitParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
	return res;
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t  extSem)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDestroyExternalSemaphore(extSem);
	return res;
}

cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
	return res;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams * launchParamsList, unsigned int  numDevices, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
	return res;
}

cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache  cacheConfig)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFuncSetCacheConfig(func, cacheConfig);
	return res;
}

cudaError_t cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig  config)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFuncSetSharedMemConfig(func, config);
	return res;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFuncGetAttributes(attr, func);
	return res;
}

cudaError_t cudaFuncSetAttribute(const void * func, enum cudaFuncAttribute  attr, int  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFuncSetAttribute(func, attr, value);
	return res;
}

cudaError_t cudaSetDoubleForDevice(double * d)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaSetDoubleForDevice(d);
	return res;
}

cudaError_t cudaSetDoubleForHost(double * d)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaSetDoubleForHost(d);
	return res;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t  stream, cudaHostFn_t  fn, void * userData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaLaunchHostFunc(stream, fn, userData);
	return res;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
	return res;
}

cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int  numBlocks, int  blockSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
	return res;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
	return res;
}

cudaError_t cudaMallocManaged(void ** devPtr, size_t  size, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocManaged(devPtr, size, flags);
	return res;
}

cudaError_t cudaMallocHost(void ** ptr, size_t  size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocHost(ptr, size);
	return res;
}

cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t  width, size_t  height)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocPitch(devPtr, pitch, width, height);
	return res;
}

cudaError_t cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocArray(array, desc, width, height, flags);
	return res;
}

cudaError_t cudaFree(void * devPtr)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaFreeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAFREE;
    
    struct cudaFreeArg *arg_ptr = (struct cudaFreeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->devPtr = devPtr;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaFreeHost(void * ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFreeHost(ptr);
	return res;
}

cudaError_t cudaFreeArray(cudaArray_t  array)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFreeArray(array);
	return res;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t  mipmappedArray)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFreeMipmappedArray(mipmappedArray);
	return res;
}

cudaError_t cudaHostAlloc(void ** pHost, size_t  size, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaHostAlloc(pHost, size, flags);
	return res;
}

cudaError_t cudaHostRegister(void * ptr, size_t  size, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaHostRegister(ptr, size, flags);
	return res;
}

cudaError_t cudaHostUnregister(void * ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaHostUnregister(ptr);
	return res;
}

cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaHostGetDevicePointer(pDevice, pHost, flags);
	return res;
}

cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaHostGetFlags(pFlags, pHost);
	return res;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr*  pitchedDevPtr, struct cudaExtent  extent)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMalloc3D(pitchedDevPtr, extent);
	return res;
}

cudaError_t cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMalloc3DArray(array, desc, extent, flags);
	return res;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  numLevels, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
	return res;
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t  mipmappedArray, unsigned int  level)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
	return res;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * p)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy3D(p);
	return res;
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy3DPeer(p);
	return res;
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy3DAsync(p, stream);
	return res;
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy3DPeerAsync(p, stream);
	return res;
}

cudaError_t cudaMemGetInfo(size_t * free, size_t * total)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemGetInfo(free, total);
	return res;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t  array)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaArrayGetInfo(desc, extent, flags, array);
	return res;
}

cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t  hArray, unsigned int  planeIdx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaArrayGetPlane(pPlaneArray, hArray, planeIdx);
	return res;
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties * sparseProperties, cudaArray_t  array)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaArrayGetSparseProperties(sparseProperties, array);
	return res;
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t  mipmap)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
	return res;
}

cudaError_t cudaMemcpyPeer(void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
	return res;
}

cudaError_t cudaMemcpy2D(void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
	return res;
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
	return res;
}

cudaError_t cudaMemcpy2DFromArray(void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
	return res;
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
	return res;
}

cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyToSymbol(symbol, src, count, offset, kind);
	return res;
}

cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
	return res;
}

cudaError_t cudaMemcpyPeerAsync(void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
	return res;
}

cudaError_t cudaMemcpy2DAsync(void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
	return res;
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
	return res;
}

cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
	return res;
}

cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
	return res;
}

cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
	return res;
}

cudaError_t cudaMemset(void * devPtr, int  value, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemset(devPtr, value, count);
	return res;
}

cudaError_t cudaMemset2D(void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemset2D(devPtr, pitch, value, width, height);
	return res;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemset3D(pitchedDevPtr, value, extent);
	return res;
}

cudaError_t cudaMemsetAsync(void * devPtr, int  value, size_t  count, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemsetAsync(devPtr, value, count, stream);
	return res;
}

cudaError_t cudaMemset2DAsync(void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
	return res;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
	return res;
}

cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetSymbolAddress(devPtr, symbol);
	return res;
}

cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetSymbolSize(size, symbol);
	return res;
}

cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t  count, int  dstDevice, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
	return res;
}

cudaError_t cudaMemAdvise(const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, int  device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemAdvise(devPtr, count, advice, device);
	return res;
}

cudaError_t cudaMemRangeGetAttribute(void * data, size_t  dataSize, enum cudaMemRangeAttribute  attribute, const void * devPtr, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
	return res;
}

cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, enum cudaMemRangeAttribute * attributes, size_t  numAttributes, const void * devPtr, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
	return res;
}

cudaError_t cudaMemcpyToArray(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
	return res;
}

cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
	return res;
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  count, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
	return res;
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
	return res;
}

cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
	return res;
}

cudaError_t cudaMallocAsync(void ** devPtr, size_t  size, cudaStream_t  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocAsync(devPtr, size, hStream);
	return res;
}

cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t  hStream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaFreeAsync(devPtr, hStream);
	return res;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t  memPool, size_t  minBytesToKeep)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolTrimTo(memPool, minBytesToKeep);
	return res;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolSetAttribute(memPool, attr, value);
	return res;
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolGetAttribute(memPool, attr, value);
	return res;
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t  memPool, const struct cudaMemAccessDesc * descList, size_t  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolSetAccess(memPool, descList, count);
	return res;
}

cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags * flags, cudaMemPool_t  memPool, struct cudaMemLocation * location)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolGetAccess(flags, memPool, location);
	return res;
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const struct cudaMemPoolProps * poolProps)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolCreate(memPool, poolProps);
	return res;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t  memPool)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolDestroy(memPool);
	return res;
}

cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t  size, cudaMemPool_t  memPool, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMallocFromPoolAsync(ptr, size, memPool, stream);
	return res;
}

cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t  memPool, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags);
	return res;
}

cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags);
	return res;
}

cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData * exportData, void * ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolExportPointer(exportData, ptr);
	return res;
}

cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t  memPool, struct cudaMemPoolPtrExportData * exportData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaMemPoolImportPointer(ptr, memPool, exportData);
	return res;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attributes, const void * ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaPointerGetAttributes(attributes, ptr);
	return res;
}

cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int  device, int  peerDevice)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
	return res;
}

cudaError_t cudaDeviceEnablePeerAccess(int  peerDevice, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceEnablePeerAccess(peerDevice, flags);
	return res;
}

cudaError_t cudaDeviceDisablePeerAccess(int  peerDevice)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceDisablePeerAccess(peerDevice);
	return res;
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t  resource)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsUnregisterResource(resource);
	return res;
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t  resource, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsResourceSetMapFlags(resource, flags);
	return res;
}

cudaError_t cudaGraphicsMapResources(int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsMapResources(count, resources, stream);
	return res;
}

cudaError_t cudaGraphicsUnmapResources(int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsUnmapResources(count, resources, stream);
	return res;
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t  resource)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
	return res;
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t  resource, unsigned int  arrayIndex, unsigned int  mipLevel)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
	return res;
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t  resource)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
	return res;
}

cudaError_t cudaBindTexture(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t  size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaBindTexture(offset, texref, devPtr, desc, size);
	return res;
}

cudaError_t cudaBindTexture2D(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, size_t  pitch)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
	return res;
}

cudaError_t cudaBindTextureToArray(const struct textureReference * texref, cudaArray_const_t  array, const struct cudaChannelFormatDesc * desc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaBindTextureToArray(texref, array, desc);
	return res;
}

cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference * texref, cudaMipmappedArray_const_t  mipmappedArray, const struct cudaChannelFormatDesc * desc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaBindTextureToMipmappedArray(texref, mipmappedArray, desc);
	return res;
}

cudaError_t cudaUnbindTexture(const struct textureReference * texref)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaUnbindTexture(texref);
	return res;
}

cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const struct textureReference * texref)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetTextureAlignmentOffset(offset, texref);
	return res;
}

cudaError_t cudaGetTextureReference(const struct textureReference ** texref, const void * symbol)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetTextureReference(texref, symbol);
	return res;
}

cudaError_t cudaBindSurfaceToArray(const struct surfaceReference * surfref, cudaArray_const_t  array, const struct cudaChannelFormatDesc * desc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaBindSurfaceToArray(surfref, array, desc);
	return res;
}

cudaError_t cudaGetSurfaceReference(const struct surfaceReference ** surfref, const void * symbol)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetSurfaceReference(surfref, symbol);
	return res;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t  array)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetChannelDesc(desc, array);
	return res;
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int  x, int  y, int  z, int  w, enum cudaChannelFormatKind  f)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	struct cudaChannelFormatDesc res = 
		lcudaCreateChannelDesc(x, y, z, w, f);
	return res;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
	return res;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDestroyTextureObject(texObject);
	return res;
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetTextureObjectResourceDesc(pResDesc, texObject);
	return res;
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetTextureObjectTextureDesc(pTexDesc, texObject);
	return res;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t  texObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
	return res;
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaCreateSurfaceObject(pSurfObject, pResDesc);
	return res;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t  surfObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDestroySurfaceObject(surfObject);
	return res;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t  surfObject)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
	return res;
}

cudaError_t cudaDriverGetVersion(int * driverVersion)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDriverGetVersion(driverVersion);
	return res;
}

cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaRuntimeGetVersion(runtimeVersion);
	return res;
}

cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned int  flags)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGraphCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGRAPHCREATE;
    
    struct cudaGraphCreateArg *arg_ptr = (struct cudaGraphCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pGraph = pGraph;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGraphCreateResponse *) dat;
	*pGraph = res->pGraph;
return res->err;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaKernelNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
	return res;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t  node, struct cudaKernelNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphKernelNodeGetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphKernelNodeSetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t  hSrc, cudaGraphNode_t  hDst)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphKernelNodeCopyAttributes(hSrc, hDst);
	return res;
}

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t  hNode, enum cudaKernelNodeAttrID  attr, union cudaKernelNodeAttrValue * value_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphKernelNodeGetAttribute(hNode, attr, value_out);
	return res;
}

cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t  hNode, enum cudaKernelNodeAttrID  attr, const union cudaKernelNodeAttrValue * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphKernelNodeSetAttribute(hNode, attr, value);
	return res;
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemcpy3DParms * pCopyParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
	return res;
}

cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
	return res;
}

cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t*  pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t*  pDependencies, size_t  numDependencies, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
	return res;
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
	return res;
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t  node, struct cudaMemcpy3DParms * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemcpyNodeGetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemcpyNodeSetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind);
	return res;
}

cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind);
	return res;
}

cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind);
	return res;
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemsetParams * pMemsetParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
	return res;
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t  node, struct cudaMemsetParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemsetNodeGetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemsetNodeSetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaHostNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
	return res;
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t  node, struct cudaHostNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphHostNodeGetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphHostNodeSetParams(node, pNodeParams);
	return res;
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaGraph_t  childGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph);
	return res;
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t  node, cudaGraph_t * pGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphChildGraphNodeGetGraph(node, pGraph);
	return res;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies);
	return res;
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event);
	return res;
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t  node, cudaEvent_t * event_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphEventRecordNodeGetEvent(node, event_out);
	return res;
}

cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t  node, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphEventRecordNodeSetEvent(node, event);
	return res;
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event);
	return res;
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t  node, cudaEvent_t * event_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphEventWaitNodeGetEvent(node, event_out);
	return res;
}

cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t  node, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphEventWaitNodeSetEvent(node, event);
	return res;
}

cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
	return res;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t  hNode, struct cudaExternalSemaphoreSignalNodeParams * params_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
	return res;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
	return res;
}

cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
	return res;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t  hNode, struct cudaExternalSemaphoreWaitNodeParams * params_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
	return res;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
	return res;
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaMemAllocNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
	return res;
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t  node, struct cudaMemAllocNodeParams * params_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemAllocNodeGetParams(node, params_out);
	return res;
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void * dptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr);
	return res;
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t  node, void * dptr_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphMemFreeNodeGetParams(node, dptr_out);
	return res;
}

cudaError_t cudaDeviceGraphMemTrim(int  device)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGraphMemTrim(device);
	return res;
}

cudaError_t cudaDeviceGetGraphMemAttribute(int  device, enum cudaGraphMemAttributeType  attr, void*  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceGetGraphMemAttribute(device, attr, value);
	return res;
}

cudaError_t cudaDeviceSetGraphMemAttribute(int  device, enum cudaGraphMemAttributeType  attr, void*  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaDeviceSetGraphMemAttribute(device, attr, value);
	return res;
}

cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t  originalGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphClone(pGraphClone, originalGraph);
	return res;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t  originalNode, cudaGraph_t  clonedGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);
	return res;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t  node, enum cudaGraphNodeType * pType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphNodeGetType(node, pType);
	return res;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t  graph, cudaGraphNode_t * nodes, size_t * numNodes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphGetNodes(graph, nodes, numNodes);
	return res;
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t  graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes);
	return res;
}

cudaError_t cudaGraphGetEdges(cudaGraph_t  graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphGetEdges(graph, from, to, numEdges);
	return res;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t  node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);
	return res;
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t  node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);
	return res;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphAddDependencies(graph, from, to, numDependencies);
	return res;
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphRemoveDependencies(graph, from, to, numDependencies);
	return res;
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t  node)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphDestroyNode(node);
	return res;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t  bufferSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
	return res;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphInstantiateWithFlags(pGraphExec, graph, flags);
	return res;
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams);
	return res;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams);
	return res;
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind);
	return res;
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind);
	return res;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind);
	return res;
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams);
	return res;
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams);
	return res;
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, cudaGraph_t  childGraph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);
	return res;
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
	return res;
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
	return res;
}

cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
	return res;
}

cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
	return res;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t  hGraphExec, cudaGraph_t  hGraph, cudaGraphNode_t * hErrorNode_out, enum cudaGraphExecUpdateResult * updateResult_out)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
	return res;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphUpload(graphExec, stream);
	return res;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphLaunch(graphExec, stream);
	return res;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t  graphExec)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphExecDestroy(graphExec);
	return res;
}

cudaError_t cudaGraphDestroy(cudaGraph_t  graph)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphDestroy(graph);
	return res;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t  graph, const char * path, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphDebugDotPrint(graph, path, flags);
	return res;
}

cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t  destroy, unsigned int  initialRefcount, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
	return res;
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t  object, unsigned int  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaUserObjectRetain(object, count);
	return res;
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t  object, unsigned int  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaUserObjectRelease(object, count);
	return res;
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count, unsigned int  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphRetainUserObject(graph, object, count, flags);
	return res;
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGraphReleaseUserObject(graph, object, count);
	return res;
}

cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long  flags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetDriverEntryPoint(symbol, funcPtr, flags);
	return res;
}

cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetExportTable(ppExportTable, pExportTableId);
	return res;
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t*  functionPtr, const void*  symbolPtr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaGetFuncBySymbol(functionPtr, symbolPtr);
	return res;
}

size_t cudnnGetVersion()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	size_t res = 
		lcudnnGetVersion();
	return res;
}

size_t cudnnGetMaxDeviceVersion()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	size_t res = 
		lcudnnGetMaxDeviceVersion();
	return res;
}

size_t cudnnGetCudartVersion()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	size_t res = 
		lcudnnGetCudartVersion();
	return res;
}

const char * cudnnGetErrorString(cudnnStatus_t  status)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	const char * res = 
		lcudnnGetErrorString(status);
	return res;
}

cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t  handle, cudnnStatus_t * rstatus, cudnnErrQueryMode_t  mode, cudnnRuntimeTag_t * tag)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnQueryRuntimeError(handle, rstatus, mode, tag);
	return res;
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType  type, int * value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetProperty(type, value);
	return res;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t * handle)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreate(handle);
	return res;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t  handle)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroy(handle);
	return res;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t  handle, cudaStream_t  streamId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetStream(handle, streamId);
	return res;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t  handle, cudaStream_t * streamId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetStream(handle, streamId);
	return res;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t * tensorDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateTensorDescriptor(tensorDesc);
	return res;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w);
	return res;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w, int  nStride, int  cStride, int  hStride, int  wStride)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
	return res;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t * dataType, int * n, int * c, int * h, int * w, int * nStride, int * cStride, int * hStride, int * wStride)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
	return res;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const int  strideA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
	return res;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  nbDims, const int  dimA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
	return res;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t  tensorDesc, int  nbDimsRequested, cudnnDataType_t * dataType, int * nbDims, int  dimA[], int  strideA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
	return res;
}

cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t  tensorDesc, size_t * size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetTensorSizeInBytes(tensorDesc, size);
	return res;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t  tensorDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyTensorDescriptor(tensorDesc);
	return res;
}

cudnnStatus_t cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t  transformDesc, const cudnnTensorDescriptor_t  srcDesc, cudnnTensorDescriptor_t  destDesc, size_t * destSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes);
	return res;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t * transformDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateTensorTransformDescriptor(transformDesc);
	return res;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc, const uint32_t  nbDims, const cudnnTensorFormat_t  destFormat, const int32_t  padBeforeA[], const int32_t  padAfterA[], const uint32_t  foldA[], const cudnnFoldingDirection_t  direction)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);
	return res;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc, uint32_t  nbDimsRequested, cudnnTensorFormat_t * destFormat, int32_t  padBeforeA[], int32_t  padAfterA[], uint32_t  foldA[], cudnnFoldingDirection_t * direction)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction);
	return res;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyTensorTransformDescriptor(transformDesc);
	return res;
}

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnTransformTensorEx(cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnTensorDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnTensorDescriptor_t  destDesc, void * destData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
	return res;
}

cudnnStatus_t cudnnAddTensor(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
	return res;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t * opTensorDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateOpTensorDescriptor(opTensorDesc);
	return res;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t  opTensorOp, cudnnDataType_t  opTensorCompType, cudnnNanPropagation_t  opTensorNanOpt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
	return res;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t * opTensorOp, cudnnDataType_t * opTensorCompType, cudnnNanPropagation_t * opTensorNanOpt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
	return res;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t  opTensorDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyOpTensorDescriptor(opTensorDesc);
	return res;
}

cudnnStatus_t cudnnOpTensor(cudnnHandle_t  handle, const cudnnOpTensorDescriptor_t  opTensorDesc, const void * alpha1, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * alpha2, const cudnnTensorDescriptor_t  bDesc, const void * B, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
	return res;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t * reduceTensorDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateReduceTensorDescriptor(reduceTensorDesc);
	return res;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t  reduceTensorOp, cudnnDataType_t  reduceTensorCompType, cudnnNanPropagation_t  reduceTensorNanOpt, cudnnReduceTensorIndices_t  reduceTensorIndices, cudnnIndicesType_t  reduceTensorIndicesType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
	return res;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t * reduceTensorOp, cudnnDataType_t * reduceTensorCompType, cudnnNanPropagation_t * reduceTensorNanOpt, cudnnReduceTensorIndices_t * reduceTensorIndices, cudnnIndicesType_t * reduceTensorIndicesType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
	return res;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t  reduceTensorDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
	return res;
}

cudnnStatus_t cudnnGetReductionIndicesSize(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnReduceTensor(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, void * indices, size_t  indicesSizeInBytes, void * workspace, size_t  workspaceSizeInBytes, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
	return res;
}

cudnnStatus_t cudnnSetTensor(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * valuePtr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetTensor(handle, yDesc, y, valuePtr);
	return res;
}

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * alpha)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnScaleTensor(handle, yDesc, y, alpha);
	return res;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t * filterDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateFilterDescriptor(filterDesc);
	return res;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  k, int  c, int  h, int  w)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
	return res;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * k, int * c, int * h, int * w)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
	return res;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  nbDims, const int  filterDimA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
	return res;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t  filterDesc, int  nbDimsRequested, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * nbDims, int  filterDimA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetFilterNdDescriptor(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
	return res;
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t  filterDesc, size_t * size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetFilterSizeInBytes(filterDesc, size);
	return res;
}

cudnnStatus_t cudnnTransformFilter(cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnFilterDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnFilterDescriptor_t  destDesc, void * destData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
	return res;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t  filterDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyFilterDescriptor(filterDesc);
	return res;
}

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t * poolingDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreatePoolingDescriptor(poolingDesc);
	return res;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t  mode, cudnnNanPropagation_t  maxpoolingNanOpt, int  windowHeight, int  windowWidth, int  verticalPadding, int  horizontalPadding, int  verticalStride, int  horizontalStride)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
	return res;
}

cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * windowHeight, int * windowWidth, int * verticalPadding, int * horizontalPadding, int * verticalStride, int * horizontalStride)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
	return res;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t  poolingDesc, const cudnnPoolingMode_t  mode, const cudnnNanPropagation_t  maxpoolingNanOpt, int  nbDims, const int  windowDimA[], const int  paddingA[], const int  strideA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
	return res;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t  poolingDesc, int  nbDimsRequested, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * nbDims, int  windowDimA[], int  paddingA[], int  strideA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
	return res;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int  nbDims, int  outputTensorDimA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
	return res;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int * n, int * c, int * h, int * w)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w);
	return res;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t  poolingDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyPoolingDescriptor(poolingDesc);
	return res;
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t * activationDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateActivationDescriptor(activationDesc);
	return res;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t  mode, cudnnNanPropagation_t  reluNanOpt, double  coef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
	return res;
}

cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t * mode, cudnnNanPropagation_t * reluNanOpt, double * coef)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
	return res;
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t  activationDesc, double  swish_beta)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetActivationDescriptorSwishBeta(activationDesc, swish_beta);
	return res;
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t  activationDesc, double * swish_beta)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetActivationDescriptorSwishBeta(activationDesc, swish_beta);
	return res;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t  activationDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyActivationDescriptor(activationDesc);
	return res;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t * normDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateLRNDescriptor(normDesc);
	return res;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t  normDesc, unsigned  lrnN, double  lrnAlpha, double  lrnBeta, double  lrnK)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
	return res;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t  normDesc, unsigned * lrnN, double * lrnAlpha, double * lrnBeta, double * lrnK)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
	return res;
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t  lrnDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyLRNDescriptor(lrnDesc);
	return res;
}

cudnnStatus_t cudnnLRNCrossChannelForward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t  derivedBnDesc, const cudnnTensorDescriptor_t  xDesc, cudnnBatchNormMode_t  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
	return res;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, const void * estimatedMean, const void * estimatedVariance, double  epsilon)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
	return res;
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor(cudnnTensorDescriptor_t  derivedNormScaleBiasDesc, cudnnTensorDescriptor_t  derivedNormMeanVarDesc, const cudnnTensorDescriptor_t  xDesc, cudnnNormMode_t  mode, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt);
	return res;
}

cudnnStatus_t cudnnNormalizationForwardInference(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * estimatedMean, const void * estimatedVariance, const cudnnTensorDescriptor_t  zDesc, const void * z, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y, double  epsilon, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt);
	return res;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t * stDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateSpatialTransformerDescriptor(stDesc);
	return res;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t  stDesc, cudnnSamplerType_t  samplerType, cudnnDataType_t  dataType, const int  nbDims, const int  dimA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA);
	return res;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t  stDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroySpatialTransformerDescriptor(stDesc);
	return res;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * theta, void * grid)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);
	return res;
}

cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * grid, const void * beta, cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t * dropoutDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateDropoutDescriptor(dropoutDesc);
	return res;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyDropoutDescriptor(dropoutDesc);
	return res;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t  handle, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDropoutGetStatesSize(handle, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t  xdesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
	return res;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
	return res;
}

cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float * dropout, void ** states, unsigned long long * seed)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed);
	return res;
}

cudnnStatus_t cudnnDropoutForward(cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  xdesc, const void * x, const cudnnTensorDescriptor_t  ydesc, void * y, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t * algoDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateAlgorithmDescriptor(algoDesc);
	return res;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t  algorithm)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetAlgorithmDescriptor(algoDesc, algorithm);
	return res;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t * algorithm)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetAlgorithmDescriptor(algoDesc, algorithm);
	return res;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t  src, cudnnAlgorithmDescriptor_t  dest)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCopyAlgorithmDescriptor(src, dest);
	return res;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t  algoDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyAlgorithmDescriptor(algoDesc);
	return res;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t * algoPerf, int  numberToCreate)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateAlgorithmPerformance(algoPerf, numberToCreate);
	return res;
}

cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t  algoDesc, cudnnStatus_t  status, float  time, size_t  memory)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
	return res;
}

cudnnStatus_t cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t * algoDesc, cudnnStatus_t * status, float * time, size_t * memory)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
	return res;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t * algoPerf, int  numberToDestroy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy);
	return res;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, size_t * algoSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnSaveAlgorithm(cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, void * algoSpace, size_t  algoSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRestoreAlgorithm(cudnnHandle_t  handle, void * algoSpace, size_t  algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t  algoDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
	return res;
}

cudnnStatus_t cudnnSetCallback(unsigned  mask, void * udata, cudnnCallback_t  fptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetCallback(mask, udata, fptr);
	return res;
}

cudnnStatus_t cudnnGetCallback(unsigned * mask, void ** udata, cudnnCallback_t * fptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetCallback(mask, udata, fptr);
	return res;
}

cudnnStatus_t cudnnOpsInferVersionCheck()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnOpsInferVersionCheck();
	return res;
}

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
	return res;
}

cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
	return res;
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
	return res;
}

cudnnStatus_t cudnnLRNCrossChannelBackward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
	return res;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, const void * dy, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  dXdMeansDesc, void * dx, void * dMeans)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans);
	return res;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
	return res;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScale, void * dBnScaleResult, void * dBnBiasResult, double  epsilon, const void * savedMean, const void * savedInvVariance)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance);
	return res;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScaleData, const void * bnBiasData, void * dBnScaleData, void * dBnBiasData, double  epsilon, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  normScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetNormalizationForwardTrainingWorkspaceSize(handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);
	return res;
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetNormalizationBackwardWorkspaceSize(handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);
	return res;
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetNormalizationTrainingReserveSpaceSize(handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt);
	return res;
}

cudnnStatus_t cudnnNormalizationForwardTraining(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, double  exponentialAverageFactor, const cudnnTensorDescriptor_t  normMeanVarDesc, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnNormalizationForwardTraining(handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);
	return res;
}

cudnnStatus_t cudnnNormalizationBackward(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const void * normScaleData, const void * normBiasData, void * dNormScaleData, void * dNormBiasData, double  epsilon, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);
	return res;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * dgrid, void * dtheta)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);
	return res;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx, const void * alphaDgrid, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * grid, const void * betaDgrid, void * dgrid)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid);
	return res;
}

cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  dydesc, const void * dy, const cudnnTensorDescriptor_t  dxdesc, void * dx, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnOpsTrainVersionCheck()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnOpsTrainVersionCheck();
	return res;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t * rnnDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateRNNDescriptor(rnnDesc);
	return res;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t  rnnDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyRNNDescriptor(rnnDesc);
	return res;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t  algo, cudnnRNNMode_t  cellMode, cudnnRNNBiasMode_t  biasMode, cudnnDirectionMode_t  dirMode, cudnnRNNInputMode_t  inputMode, cudnnDataType_t  dataType, cudnnDataType_t  mathPrec, cudnnMathType_t  mathType, int32_t  inputSize, int32_t  hiddenSize, int32_t  projSize, int32_t  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, uint32_t  auxFlags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);
	return res;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t * algo, cudnnRNNMode_t * cellMode, cudnnRNNBiasMode_t * biasMode, cudnnDirectionMode_t * dirMode, cudnnRNNInputMode_t * inputMode, cudnnDataType_t * dataType, cudnnDataType_t * mathPrec, cudnnMathType_t * mathType, int32_t * inputSize, int32_t * hiddenSize, int32_t * projSize, int32_t * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, uint32_t * auxFlags)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);
	return res;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  hiddenSize, const int  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, cudnnRNNInputMode_t  inputMode, cudnnDirectionMode_t  direction, cudnnRNNMode_t  cellMode, cudnnRNNAlgo_t  algo, cudnnDataType_t  mathPrec)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);
	return res;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int * hiddenSize, int * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, cudnnRNNInputMode_t * inputMode, cudnnDirectionMode_t * direction, cudnnRNNMode_t * cellMode, cudnnRNNAlgo_t * algo, cudnnDataType_t * mathPrec)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);
	return res;
}

cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t  mType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNMatrixMathType(rnnDesc, mType);
	return res;
}

cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t * mType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNMatrixMathType(rnnDesc, mType);
	return res;
}

cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t  biasMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNBiasMode(rnnDesc, biasMode);
	return res;
}

cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t * biasMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNBiasMode(rnnDesc, biasMode);
	return res;
}

cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNSetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip);
	return res;
}

cudnnStatus_t cudnnRNNGetClip_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNGetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip);
	return res;
}

cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
	return res;
}

cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNGetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
	return res;
}

cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  recProjSize, const int  outProjSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
	return res;
}

cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * recProjSize, int * outProjSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
	return res;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t  rnnDesc, const int  minibatch, const cudnnDataType_t  dataType, cudnnPersistentRNNPlan_t * plan)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan);
	return res;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t  plan)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyPersistentRNNPlan(plan);
	return res;
}

cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t  rnnDesc, cudnnPersistentRNNPlan_t  plan)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetPersistentRNNPlan(rnnDesc, plan);
	return res;
}

cudnnStatus_t cudnnBuildRNNDynamic(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  miniBatch)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBuildRNNDynamic(handle, rnnDesc, miniBatch);
	return res;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fMode, cudnnRNNDataDescriptor_t  xDesc, size_t * workSpaceSize, size_t * reserveSpaceSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNTempSpaceSizes(handle, rnnDesc, fMode, xDesc, workSpaceSize, reserveSpaceSize);
	return res;
}

cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, cudnnDataType_t  dataType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
	return res;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, size_t * weightSpaceSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNWeightSpaceSize(handle, rnnDesc, weightSpaceSize);
	return res;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerMatDesc, void ** linLayerMat)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat);
	return res;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerBiasDesc, void ** linLayerBias)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias);
	return res;
}

cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int32_t  pseudoLayer, size_t  weightSpaceSize, const void * weightSpace, int32_t  linLayerID, cudnnTensorDescriptor_t  mDesc, void ** mAddr, cudnnTensorDescriptor_t  bDesc, void ** bAddr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNWeightParams(handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr);
	return res;
}

cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t  rnnDesc, unsigned  paddingMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNPaddingMode(rnnDesc, paddingMode);
	return res;
}

cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t  rnnDesc, unsigned * paddingMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNPaddingMode(rnnDesc, paddingMode);
	return res;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t * rnnDataDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateRNNDataDescriptor(rnnDataDesc);
	return res;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyRNNDataDescriptor(rnnDataDesc);
	return res;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t  dataType, cudnnRNNDataLayout_t  layout, int  maxSeqLength, int  batchSize, int  vectorSize, const int  seqLengthArray[], void * paddingFill)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);
	return res;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t * dataType, cudnnRNNDataLayout_t * layout, int * maxSeqLength, int * batchSize, int * vectorSize, int  arrayLengthRequested, int  seqLengthArray[], void * paddingFill)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill);
	return res;
}

cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRNNForward(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnRNNDataDescriptor_t  yDesc, void * y, cudnnTensorDescriptor_t  hDesc, const void * hx, void * hy, cudnnTensorDescriptor_t  cDesc, const void * cx, void * cy, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNForward(handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);
	return res;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnAlgorithmDescriptor_t  algoDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
	return res;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
	return res;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t * seqDataDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateSeqDataDescriptor(seqDataDesc);
	return res;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t  seqDataDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroySeqDataDescriptor(seqDataDesc);
	return res;
}

cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t  dataType, int  nbDims, const int  dimA[], const cudnnSeqDataAxis_t  axes[], size_t  seqLengthArraySize, const int  seqLengthArray[], void * paddingFill)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);
	return res;
}

cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t  seqDataDesc, cudnnDataType_t * dataType, int * nbDims, int  nbDimsRequested, int  dimA[], cudnnSeqDataAxis_t  axes[], size_t * seqLengthArraySize, size_t  seqLengthSizeRequested, int  seqLengthArray[], void * paddingFill)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetSeqDataDescriptor(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);
	return res;
}

cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t * attnDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateAttnDescriptor(attnDesc);
	return res;
}

cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t  attnDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyAttnDescriptor(attnDesc);
	return res;
}

cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnDescriptor_t  attnDesc, unsigned  attnMode, int  nHeads, double  smScaler, cudnnDataType_t  dataType, cudnnDataType_t  computePrec, cudnnMathType_t  mathType, cudnnDropoutDescriptor_t  attnDropoutDesc, cudnnDropoutDescriptor_t  postDropoutDesc, int  qSize, int  kSize, int  vSize, int  qProjSize, int  kProjSize, int  vProjSize, int  oProjSize, int  qoMaxSeqLength, int  kvMaxSeqLength, int  maxBatchSize, int  maxBeamSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
	return res;
}

cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnDescriptor_t  attnDesc, unsigned * attnMode, int * nHeads, double * smScaler, cudnnDataType_t * dataType, cudnnDataType_t * computePrec, cudnnMathType_t * mathType, cudnnDropoutDescriptor_t * attnDropoutDesc, cudnnDropoutDescriptor_t * postDropoutDesc, int * qSize, int * kSize, int * vSize, int * qProjSize, int * kProjSize, int * vProjSize, int * oProjSize, int * qoMaxSeqLength, int * kvMaxSeqLength, int * maxBatchSize, int * maxBeamSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
	return res;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, size_t * weightSizeInBytes, size_t * workSpaceSizeInBytes, size_t * reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnMultiHeadAttnWeightKind_t  wKind, size_t  weightSizeInBytes, const void * weights, cudnnTensorDescriptor_t  wDesc, void ** wAddr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr);
	return res;
}

cudnnStatus_t cudnnMultiHeadAttnForward(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, int  currIdx, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsQO[], const int  devSeqLengthsKV[], const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const void * residuals, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  oDesc, void * out, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
	return res;
}

cudnnStatus_t cudnnAdvInferVersionCheck()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnAdvInferVersionCheck();
	return res;
}

cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNForwardTraining(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  yDesc, const void * y, const void * dy, cudnnRNNDataDescriptor_t  xDesc, void * dx, cudnnTensorDescriptor_t  hDesc, const void * hx, const void * dhy, void * dhx, cudnnTensorDescriptor_t  cDesc, const void * cx, const void * dcy, void * dcx, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNBackwardData_v8(handle, rnnDesc, devSeqLengths, yDesc, y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);
	return res;
}

cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnWgradMode_t  addGrad, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnTensorDescriptor_t  hDesc, const void * hx, cudnnRNNDataDescriptor_t  yDesc, const void * y, size_t  weightSpaceSize, void * dweightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNBackwardWeights_v8(handle, rnnDesc, addGrad, devSeqLengths, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize, dweightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);
	return res;
}

cudnnStatus_t cudnnRNNForwardTrainingEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRNNBackwardDataEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  yDesc, const void * y, const cudnnRNNDataDescriptor_t  dyDesc, const void * dy, const cudnnRNNDataDescriptor_t  dcDesc, const void * dcAttn, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnRNNDataDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const cudnnRNNDataDescriptor_t  dkDesc, void * dkeys, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnRNNDataDescriptor_t  yDesc, const void * y, void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count);
	return res;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count);
	return res;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count);
	return res;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, const void * workspace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, const int  loWinIdx[], const int  hiWinIdx[], const int  devSeqLengthsDQDO[], const int  devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t  doDesc, const void * dout, const cudnnSeqDataDescriptor_t  dqDesc, void * dqueries, const void * queries, const cudnnSeqDataDescriptor_t  dkDesc, void * dkeys, const void * keys, const cudnnSeqDataDescriptor_t  dvDesc, void * dvalues, const void * values, size_t  weightSizeInBytes, const void * weights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
	return res;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnWgradMode_t  addGrad, const cudnnSeqDataDescriptor_t  qDesc, const void * queries, const cudnnSeqDataDescriptor_t  kDesc, const void * keys, const cudnnSeqDataDescriptor_t  vDesc, const void * values, const cudnnSeqDataDescriptor_t  doDesc, const void * dout, size_t  weightSizeInBytes, const void * weights, void * dweights, size_t  workSpaceSizeInBytes, void * workSpace, size_t  reserveSpaceSizeInBytes, void * reserveSpace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
	return res;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t * ctcLossDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateCTCLossDescriptor(ctcLossDesc);
	return res;
}

cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetCTCLossDescriptor(ctcLossDesc, compType);
	return res;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode);
	return res;
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode, int  maxLabelLength)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength);
	return res;
}

cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetCTCLossDescriptor(ctcLossDesc, compType);
	return res;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode);
	return res;
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode, int * maxLabelLength)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength);
	return res;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyCTCLossDescriptor(ctcLossDesc);
	return res;
}

cudnnStatus_t cudnnCTCLoss(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  hostLabels[], const int  hostLabelLengths[], const int  hostInputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, void * workspace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCTCLoss(handle, probsDesc, probs, hostLabels, hostLabelLengths, hostInputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  labels[], const int  labelLengths[], const int  inputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, size_t  workSpaceSizeInBytes, void * workspace)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCTCLoss_v8(handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace);
	return res;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, const int * labels, const int * labelLengths, const int * inputLengths, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetCTCLossWorkspaceSize_v8(handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnAdvTrainVersionCheck()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnAdvTrainVersionCheck();
	return res;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t * convDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateConvolutionDescriptor(convDesc);
	return res;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t  convDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyConvolutionDescriptor(convDesc);
	return res;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t  mathType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetConvolutionMathType(convDesc, mathType);
	return res;
}

cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t * mathType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionMathType(convDesc, mathType);
	return res;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t  convDesc, int  groupCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetConvolutionGroupCount(convDesc, groupCount);
	return res;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t  convDesc, int * groupCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionGroupCount(convDesc, groupCount);
	return res;
}

cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t  reorderType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetConvolutionReorderType(convDesc, reorderType);
	return res;
}

cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t * reorderType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionReorderType(convDesc, reorderType);
	return res;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t  convDesc, int  pad_h, int  pad_w, int  u, int  v, int  dilation_h, int  dilation_w, cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
	return res;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t  convDesc, int * pad_h, int * pad_w, int * u, int * v, int * dilation_h, int * dilation_w, cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
	return res;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t  convDesc, int  arrayLength, const int  padA[], const int  filterStrideA[], const int  dilationA[], cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
	return res;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t  convDesc, int  arrayLengthRequested, int * arrayLength, int  padA[], int  strideA[], int  dilationA[], cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType);
	return res;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int * n, int * c, int * h, int * w)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
	return res;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int  nbDims, int  tensorOuputDimA[])
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
	return res;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionForwardAlgorithmMaxCount(handle, count);
	return res;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnFilterDescriptor_t  filterDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  destDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionForwardAlgorithm_v7(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
	return res;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
	return res;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, void * y, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnIm2Col(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, void * colBuffer)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
	return res;
}

cudnnStatus_t cudnnReorderFilterAndBias(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, cudnnReorderType_t  reorderType, const void * filterData, void * reorderedFilterData, int  reorderBias, const void * biasData, void * reorderedBiasData)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);
	return res;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, cudnnConvolutionFwdAlgo_t  algo, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
	return res;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward(cudnnHandle_t  handle, const void * alpha1, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * alpha2, const cudnnTensorDescriptor_t  zDesc, const void * z, const cudnnTensorDescriptor_t  biasDesc, const void * bias, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);
	return res;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
	return res;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
	return res;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, void * dx, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
	return res;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, cudnnConvolutionBwdDataAlgo_t  algo, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t  handle, const void * alpha, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdDataAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
	return res;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const cudnnTensorFormat_t  transformFormat, cudnnFilterDescriptor_t  foldedFilterDesc, cudnnTensorDescriptor_t  paddedDiffDesc, cudnnConvolutionDescriptor_t  foldedConvDesc, cudnnTensorDescriptor_t  foldedGradDesc, cudnnTensorTransformDescriptor_t  filterFoldTransDesc, cudnnTensorTransformDescriptor_t  diffPadTransDesc, cudnnTensorTransformDescriptor_t  gradFoldTransDesc, cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);
	return res;
}

cudnnStatus_t cudnnCnnInferVersionCheck()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCnnInferVersionCheck();
	return res;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
	return res;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
	return res;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * y, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, void * dw, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
	return res;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, cudnnConvolutionBwdFilterAlgo_t  algo, size_t * sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
	return res;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdFilterAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnFilterDescriptor_t  dwDesc, void * dw)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
	return res;
}

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dbDesc, void * db)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
	return res;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t * constPack, cudnnFusedOps_t  ops)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateFusedOpsConstParamPack(constPack, ops);
	return res;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t  constPack)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyFusedOpsConstParamPack(constPack);
	return res;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, const void * param)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param);
	return res;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, void * param, int * isNULL)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param, isNULL);
	return res;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t * varPack, cudnnFusedOps_t  ops)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateFusedOpsVariantParamPack(varPack, ops);
	return res;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t  varPack)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyFusedOpsVariantParamPack(varPack);
	return res;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);
	return res;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);
	return res;
}

cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t * plan, cudnnFusedOps_t  ops)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCreateFusedOpsPlan(plan, ops);
	return res;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t  plan)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnDestroyFusedOpsPlan(plan);
	return res;
}

cudnnStatus_t cudnnMakeFusedOpsPlan(cudnnHandle_t  handle, cudnnFusedOpsPlan_t  plan, const cudnnFusedOpsConstParamPack_t  constPack, size_t * workspaceSizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnMakeFusedOpsPlan(handle, plan, constPack, workspaceSizeInBytes);
	return res;
}

cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t  handle, const cudnnFusedOpsPlan_t  plan, cudnnFusedOpsVariantParamPack_t  varPack)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnFusedOpsExecute(handle, plan, varPack);
	return res;
}

cudnnStatus_t cudnnCnnTrainVersionCheck()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnCnnTrainVersionCheck();
	return res;
}

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t  descriptorType, cudnnBackendDescriptor_t * descriptor)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendCreateDescriptor(descriptorType, descriptor);
	return res;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t  descriptor)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendDestroyDescriptor(descriptor);
	return res;
}

cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t  descriptor)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendInitialize(descriptor);
	return res;
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t  descriptor)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendFinalize(descriptor);
	return res;
}

cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  elementCount, const void * arrayOfElements)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendSetAttribute(descriptor, attributeName, attributeType, elementCount, arrayOfElements);
	return res;
}

cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t const  descriptor, cudnnBackendAttributeName_t  attributeName, cudnnBackendAttributeType_t  attributeType, int64_t  requestedElementCount, int64_t * elementCount, void * arrayOfElements)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendGetAttribute(descriptor, attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
	return res;
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t  handle, cudnnBackendDescriptor_t  executionPlan, cudnnBackendDescriptor_t  variantPack)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudnnStatus_t res = 
		lcudnnBackendExecute(handle, executionPlan, variantPack);
	return res;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t*  handle)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasCreate_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASCREATE_V2;
    
    struct cublasCreate_v2Arg *arg_ptr = (struct cublasCreate_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasCreate_v2Response *) dat;
	*handle = res->handle;
return res->err;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t  handle)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasDestroy_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASDESTROY_V2;
    
    struct cublasDestroy_v2Arg *arg_ptr = (struct cublasDestroy_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t  handle, int*  version)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasGetVersion_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASGETVERSION_V2;
    
    struct cublasGetVersion_v2Arg *arg_ptr = (struct cublasGetVersion_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->version = version;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasGetVersion_v2Response *) dat;
	*version = res->version;
return res->err;
}

cublasStatus_t cublasGetProperty(libraryPropertyType  type, int*  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetProperty(type, value);
	return res;
}

size_t cublasGetCudartVersion()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasGetCudartVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASGETCUDARTVERSION;
    
    struct cublasGetCudartVersionArg *arg_ptr = (struct cublasGetCudartVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (size_t *) dat;
    return *res;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t  handle, void*  workspace, size_t  workspaceSizeInBytes)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSetWorkspace_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSETWORKSPACE_V2;
    
    struct cublasSetWorkspace_v2Arg *arg_ptr = (struct cublasSetWorkspace_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->workspace = workspace;
	arg_ptr->workspaceSizeInBytes = workspaceSizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t  handle, cudaStream_t  streamId)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSetStream_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSETSTREAM_V2;
    
    struct cublasSetStream_v2Arg *arg_ptr = (struct cublasSetStream_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->streamId = streamId;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t  handle, cudaStream_t*  streamId)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetStream_v2(handle, streamId);
	return res;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t  handle, cublasPointerMode_t*  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetPointerMode_v2(handle, mode);
	return res;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t  handle, cublasPointerMode_t  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetPointerMode_v2(handle, mode);
	return res;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t  handle, cublasAtomicsMode_t*  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetAtomicsMode(handle, mode);
	return res;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t  handle, cublasAtomicsMode_t  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetAtomicsMode(handle, mode);
	return res;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t  handle, cublasMath_t*  mode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetMathMode(handle, mode);
	return res;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t  handle, cublasMath_t  mode)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSetMathModeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSETMATHMODE;
    
    struct cublasSetMathModeArg *arg_ptr = (struct cublasSetMathModeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->mode = mode;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasGetSmCountTarget(cublasHandle_t  handle, int*  smCountTarget)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetSmCountTarget(handle, smCountTarget);
	return res;
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t  handle, int  smCountTarget)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetSmCountTarget(handle, smCountTarget);
	return res;
}

const char* cublasGetStatusName(cublasStatus_t  status)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	const char* res = 
		lcublasGetStatusName(status);
	return res;
}

const char* cublasGetStatusString(cublasStatus_t  status)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	const char* res = 
		lcublasGetStatusString(status);
	return res;
}

cublasStatus_t cublasLoggerConfigure(int  logIsOn, int  logToStdOut, int  logToStdErr, const char*  logFileName)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName);
	return res;
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback  userCallback)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetLoggerCallback(userCallback);
	return res;
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback*  userCallback)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetLoggerCallback(userCallback);
	return res;
}

cublasStatus_t cublasSetVector(int  n, int  elemSize, const void*  x, int  incx, void*  devicePtr, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetVector(n, elemSize, x, incx, devicePtr, incy);
	return res;
}

cublasStatus_t cublasGetVector(int  n, int  elemSize, const void*  x, int  incx, void*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetVector(n, elemSize, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasSetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
	return res;
}

cublasStatus_t cublasGetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
	return res;
}

cublasStatus_t cublasSetVectorAsync(int  n, int  elemSize, const void*  hostPtr, int  incx, void*  devicePtr, int  incy, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
	return res;
}

cublasStatus_t cublasGetVectorAsync(int  n, int  elemSize, const void*  devicePtr, int  incx, void*  hostPtr, int  incy, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
	return res;
}

cublasStatus_t cublasSetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
	return res;
}

cublasStatus_t cublasGetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
	return res;
}

void cublasXerbla(const char*  srName, int  info)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
		lcublasXerbla(srName, info);
}

cublasStatus_t cublasNrm2Ex(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType);
	return res;
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSnrm2_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDnrm2_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasScnrm2_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasScnrm2_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDznrm2_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasDotEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
	return res;
}

cublasStatus_t cublasDotcEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
	return res;
}

cublasStatus_t cublasSdot_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, const float*  y, int  incy, float*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSdot_v2(handle, n, x, incx, y, incy, result);
	return res;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, const double*  y, int  incy, double*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDdot_v2(handle, n, x, incx, y, incy, result);
	return res;
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCdotu_v2(handle, n, x, incx, y, incy, result);
	return res;
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCdotc_v2(handle, n, x, incx, y, incy, result);
	return res;
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZdotu_v2(handle, n, x, incx, y, incy, result);
	return res;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZdotc_v2(handle, n, x, incx, y, incy, result);
	return res;
}

cublasStatus_t cublasScalEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int  incx, cudaDataType  executionType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType);
	return res;
}

cublasStatus_t cublasSscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSscal_v2(handle, n, alpha, x, incx);
	return res;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDscal_v2(handle, n, alpha, x, incx);
	return res;
}

cublasStatus_t cublasCscal_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCscal_v2(handle, n, alpha, x, incx);
	return res;
}

cublasStatus_t cublasCsscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsscal_v2(handle, n, alpha, x, incx);
	return res;
}

cublasStatus_t cublasZscal_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZscal_v2(handle, n, alpha, x, incx);
	return res;
}

cublasStatus_t cublasZdscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZdscal_v2(handle, n, alpha, x, incx);
	return res;
}

cublasStatus_t cublasAxpyEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, cudaDataType  executiontype)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
	return res;
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t  handle, int  n, const float*  alpha, const float*  x, int  incx, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t  handle, int  n, const double*  alpha, const double*  x, int  incx, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasCaxpy_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasCopyEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCopyEx(handle, n, x, xType, incx, y, yType, incy);
	return res;
}

cublasStatus_t cublasScopy_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasScopy_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDcopy_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasCcopy_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCcopy_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZcopy_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasSswap_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSswap_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasDswap_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDswap_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasCswap_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCswap_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasZswap_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZswap_v2(handle, n, x, incx, y, incy);
	return res;
}

cublasStatus_t cublasSwapEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSwapEx(handle, n, x, xType, incx, y, yType, incy);
	return res;
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIsamax_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIdamax_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIcamax_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIcamax_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIzamax_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIzamax_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIamaxEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIamaxEx(handle, n, x, xType, incx, result);
	return res;
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIsamin_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIdamin_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIcamin_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIcamin_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIzamin_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIzamin_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasIaminEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasIaminEx(handle, n, x, xType, incx, result);
	return res;
}

cublasStatus_t cublasAsumEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype);
	return res;
}

cublasStatus_t cublasSasum_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSasum_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasDasum_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDasum_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasScasum_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasScasum_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasDzasum_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDzasum_v2(handle, n, x, incx, result);
	return res;
}

cublasStatus_t cublasSrot_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  c, const float*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSrot_v2(handle, n, x, incx, y, incy, c, s);
	return res;
}

cublasStatus_t cublasDrot_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  c, const double*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDrot_v2(handle, n, x, incx, y, incy, c, s);
	return res;
}

cublasStatus_t cublasCrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const cuComplex*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCrot_v2(handle, n, x, incx, y, incy, c, s);
	return res;
}

cublasStatus_t cublasCsrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const float*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
	return res;
}

cublasStatus_t cublasZrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const cuDoubleComplex*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZrot_v2(handle, n, x, incx, y, incy, c, s);
	return res;
}

cublasStatus_t cublasZdrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const double*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
	return res;
}

cublasStatus_t cublasRotEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
	return res;
}

cublasStatus_t cublasSrotg_v2(cublasHandle_t  handle, float*  a, float*  b, float*  c, float*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSrotg_v2(handle, a, b, c, s);
	return res;
}

cublasStatus_t cublasDrotg_v2(cublasHandle_t  handle, double*  a, double*  b, double*  c, double*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDrotg_v2(handle, a, b, c, s);
	return res;
}

cublasStatus_t cublasCrotg_v2(cublasHandle_t  handle, cuComplex*  a, cuComplex*  b, float*  c, cuComplex*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCrotg_v2(handle, a, b, c, s);
	return res;
}

cublasStatus_t cublasZrotg_v2(cublasHandle_t  handle, cuDoubleComplex*  a, cuDoubleComplex*  b, double*  c, cuDoubleComplex*  s)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZrotg_v2(handle, a, b, c, s);
	return res;
}

cublasStatus_t cublasRotgEx(cublasHandle_t  handle, void*  a, void*  b, cudaDataType  abType, void*  c, void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasRotgEx(handle, a, b, abType, c, s, csType, executiontype);
	return res;
}

cublasStatus_t cublasSrotm_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  param)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSrotm_v2(handle, n, x, incx, y, incy, param);
	return res;
}

cublasStatus_t cublasDrotm_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  param)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDrotm_v2(handle, n, x, incx, y, incy, param);
	return res;
}

cublasStatus_t cublasRotmEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
	return res;
}

cublasStatus_t cublasSrotmg_v2(cublasHandle_t  handle, float*  d1, float*  d2, float*  x1, const float*  y1, float*  param)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSrotmg_v2(handle, d1, d2, x1, y1, param);
	return res;
}

cublasStatus_t cublasDrotmg_v2(cublasHandle_t  handle, double*  d1, double*  d2, double*  x1, const double*  y1, double*  param)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDrotmg_v2(handle, d1, d2, x1, y1, param);
	return res;
}

cublasStatus_t cublasRotmgEx(cublasHandle_t  handle, void*  d1, cudaDataType  d1Type, void*  d2, cudaDataType  d2Type, void*  x1, cudaDataType  x1Type, const void*  y1, cudaDataType  y1Type, void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
	return res;
}

cublasStatus_t cublasSgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasSgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasDgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasCgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasZgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasStrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasDtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasCtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasZtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasStbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasDtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasCtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasZtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasStpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasDtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasCtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasZtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasStrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasDtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasCtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasZtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasStpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasDtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasCtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasZtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
	return res;
}

cublasStatus_t cublasStbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasDtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasCtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasZtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
	return res;
}

cublasStatus_t cublasSsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasDsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasCsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasZsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasChemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasZhemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasSsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasDsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasChbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasZhbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasSspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  AP, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasDspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  AP, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasChpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasZhpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
	return res;
}

cublasStatus_t cublasSger_v2(cublasHandle_t  handle, int  m, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasDger_v2(cublasHandle_t  handle, int  m, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasCgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasCgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasZgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasZgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasSsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
	return res;
}

cublasStatus_t cublasDsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
	return res;
}

cublasStatus_t cublasCsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
	return res;
}

cublasStatus_t cublasZsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
	return res;
}

cublasStatus_t cublasCher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda);
	return res;
}

cublasStatus_t cublasZher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda);
	return res;
}

cublasStatus_t cublasSspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSspr_v2(handle, uplo, n, alpha, x, incx, AP);
	return res;
}

cublasStatus_t cublasDspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDspr_v2(handle, uplo, n, alpha, x, incx, AP);
	return res;
}

cublasStatus_t cublasChpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasChpr_v2(handle, uplo, n, alpha, x, incx, AP);
	return res;
}

cublasStatus_t cublasZhpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP);
	return res;
}

cublasStatus_t cublasSsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasDsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasCsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasZsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasCher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasZher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
	return res;
}

cublasStatus_t cublasSspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
	return res;
}

cublasStatus_t cublasDspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
	return res;
}

cublasStatus_t cublasChpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
	return res;
}

cublasStatus_t cublasZhpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
	return res;
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCgemm3mEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasHgemm(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, const __half*  B, int  ldb, const __half*  beta, __half*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasSgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasGemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
	return res;
}

cublasStatus_t cublasCgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasUint8gemmBias(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, cublasOperation_t  transc, int  m, int  n, int  k, const unsigned char*  A, int  A_bias, int  lda, const unsigned char*  B, int  B_bias, int  ldb, unsigned char*  C, int  C_bias, int  ldc, int  C_mult, int  C_shift)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift);
	return res;
}

cublasStatus_t cublasSsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  beta, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	return res;
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  beta, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCsyrkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasCsyrk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasCherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const cuComplex*  A, int  lda, const float*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const cuDoubleComplex*  A, int  lda, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCherkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasCherk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
	return res;
}

cublasStatus_t cublasSsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasSsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasDsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasSsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasCsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasChemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasZhemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
	return res;
}

cublasStatus_t cublasStrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, float*  B, int  ldb)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
	return res;
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, double*  B, int  ldb)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
	return res;
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, cuComplex*  B, int  ldb)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
	return res;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  B, int  ldb)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
	return res;
}

cublasStatus_t cublasStrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasCtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasZtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasHgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half* const  Aarray[], int  lda, const __half* const  Barray[], int  ldb, const __half*  beta, __half* const  Carray[], int  ldc, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
	return res;
}

cublasStatus_t cublasSgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  Barray[], int  ldb, const float*  beta, float* const  Carray[], int  ldc, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
	return res;
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  Barray[], int  ldb, const double*  beta, double* const  Carray[], int  ldc, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
	return res;
}

cublasStatus_t cublasCgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
	return res;
}

cublasStatus_t cublasCgemm3mBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
	return res;
}

cublasStatus_t cublasZgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  Barray[], int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int  ldc, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
	return res;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int  lda, const void* const  Barray[], cudaDataType  Btype, int  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int  ldc, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
	return res;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, long long int  strideC, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
	return res;
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  B, int  ldb, long long int  strideB, const float*  beta, float*  C, int  ldc, long long int  strideC, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	return res;
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  B, int  ldb, long long int  strideB, const double*  beta, double*  C, int  ldc, long long int  strideC, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	return res;
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	return res;
}

cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	return res;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  B, int  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	return res;
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  B, int  ldb, long long int  strideB, const __half*  beta, __half*  C, int  ldc, long long int  strideC, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
	return res;
}

cublasStatus_t cublasSgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  beta, const float*  B, int  ldb, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasDgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  beta, const double*  B, int  ldb, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasCgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasZgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	return res;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t  handle, int  n, float* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
	return res;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t  handle, int  n, double* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
	return res;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t  handle, int  n, cuComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgetrfBatched(handle, n, A, lda, P, info, batchSize);
	return res;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t  handle, int  n, cuDoubleComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgetrfBatched(handle, n, A, lda, P, info, batchSize);
	return res;
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t  handle, int  n, const float* const  A[], int  lda, const int*  P, float* const  C[], int  ldc, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
	return res;
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t  handle, int  n, const double* const  A[], int  lda, const int*  P, double* const  C[], int  ldc, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
	return res;
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, const int*  P, cuComplex* const  C[], int  ldc, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
	return res;
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, const int*  P, cuDoubleComplex* const  C[], int  ldc, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
	return res;
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const float* const  Aarray[], int  lda, const int*  devIpiv, float* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
	return res;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const double* const  Aarray[], int  lda, const int*  devIpiv, double* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
	return res;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuComplex* const  Aarray[], int  lda, const int*  devIpiv, cuComplex* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
	return res;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuDoubleComplex* const  Aarray[], int  lda, const int*  devIpiv, cuDoubleComplex* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
	return res;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float* const  A[], int  lda, float* const  B[], int  ldb, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
	return res;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double* const  A[], int  lda, double* const  B[], int  ldb, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
	return res;
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  A[], int  lda, cuComplex* const  B[], int  ldb, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
	return res;
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  B[], int  ldb, int  batchCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
	return res;
}

cublasStatus_t cublasSmatinvBatched(cublasHandle_t  handle, int  n, const float* const  A[], int  lda, float* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
	return res;
}

cublasStatus_t cublasDmatinvBatched(cublasHandle_t  handle, int  n, const double* const  A[], int  lda, double* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
	return res;
}

cublasStatus_t cublasCmatinvBatched(cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, cuComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
	return res;
}

cublasStatus_t cublasZmatinvBatched(cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
	return res;
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t  handle, int  m, int  n, float* const  Aarray[], int  lda, float* const  TauArray[], int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
	return res;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t  handle, int  m, int  n, double* const  Aarray[], int  lda, double* const  TauArray[], int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
	return res;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t  handle, int  m, int  n, cuComplex* const  Aarray[], int  lda, cuComplex* const  TauArray[], int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
	return res;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t  handle, int  m, int  n, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  TauArray[], int*  info, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
	return res;
}

cublasStatus_t cublasSgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, float* const  Aarray[], int  lda, float* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
	return res;
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, double* const  Aarray[], int  lda, double* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
	return res;
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuComplex* const  Aarray[], int  lda, cuComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
	return res;
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
	return res;
}

cublasStatus_t cublasSdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const float*  A, int  lda, const float*  x, int  incx, float*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
	return res;
}

cublasStatus_t cublasDdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const double*  A, int  lda, const double*  x, int  incx, double*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
	return res;
}

cublasStatus_t cublasCdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, cuComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
	return res;
}

cublasStatus_t cublasZdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  C, int  ldc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
	return res;
}

cublasStatus_t cublasStpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  AP, float*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStpttr(handle, uplo, n, AP, A, lda);
	return res;
}

cublasStatus_t cublasDtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  AP, double*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtpttr(handle, uplo, n, AP, A, lda);
	return res;
}

cublasStatus_t cublasCtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  AP, cuComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtpttr(handle, uplo, n, AP, A, lda);
	return res;
}

cublasStatus_t cublasZtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  A, int  lda)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtpttr(handle, uplo, n, AP, A, lda);
	return res;
}

cublasStatus_t cublasStrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  A, int  lda, float*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasStrttp(handle, uplo, n, A, lda, AP);
	return res;
}

cublasStatus_t cublasDtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  A, int  lda, double*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasDtrttp(handle, uplo, n, A, lda, AP);
	return res;
}

cublasStatus_t cublasCtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  A, int  lda, cuComplex*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasCtrttp(handle, uplo, n, A, lda, AP);
	return res;
}

cublasStatus_t cublasZtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  AP)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasZtrttp(handle, uplo, n, A, lda, AP);
	return res;
}

cudaError_t cudaProfilerInitialize(const char * configFile, const char * outputFile, cudaOutputMode_t  outputMode)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cudaError_t res = 
		lcudaProfilerInitialize(configFile, outputFile, outputMode);
	return res;
}

cudaError_t cudaProfilerStart()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaProfilerStartArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAPROFILERSTART;
    
    struct cudaProfilerStartArg *arg_ptr = (struct cudaProfilerStartArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaProfilerStop()
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaProfilerStopArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAPROFILERSTOP;
    
    struct cudaProfilerStopArg *arg_ptr = (struct cudaProfilerStopArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

nvrtcResult nvrtcGetCUBINSize(nvrtcProgram  prog, size_t * cubinSizeRet)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	nvrtcResult res = 
		lnvrtcGetCUBINSize(prog, cubinSizeRet);
	return res;
}

nvrtcResult nvrtcGetCUBIN(nvrtcProgram  prog, char * cubin)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	nvrtcResult res = 
		lnvrtcGetCUBIN(prog, cubin);
	return res;
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t*  lightHandle)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTCREATE;
    
    struct cublasLtCreateArg *arg_ptr = (struct cublasLtCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->lightHandle = lightHandle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasLtCreateResponse *) dat;
	*lightHandle = res->lightHandle;
return res->err;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t  lightHandle)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTDESTROY;
    
    struct cublasLtDestroyArg *arg_ptr = (struct cublasLtDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->lightHandle = lightHandle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

const char* cublasLtGetStatusName(cublasStatus_t  status)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	const char* res = 
		lcublasLtGetStatusName(status);
	return res;
}

const char* cublasLtGetStatusString(cublasStatus_t  status)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	const char* res = 
		lcublasLtGetStatusString(status);
	return res;
}

size_t cublasLtGetVersion()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	size_t res = 
		lcublasLtGetVersion();
	return res;
}

size_t cublasLtGetCudartVersion()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	size_t res = 
		lcublasLtGetCudartVersion();
	return res;
}

cublasStatus_t cublasLtGetProperty(libraryPropertyType  type, int*  value)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtGetProperty(type, value);
	return res;
}

cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t  lightHandle, cublasLtMatrixTransformDesc_t  transformDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  beta, const void*  B, cublasLtMatrixLayout_t  Bdesc, void*  C, cublasLtMatrixLayout_t  Cdesc, cudaStream_t  stream)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixTransform(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
	return res;
}

cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t  matLayout, size_t  size, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixLayoutInit_internal(matLayout, size, type, rows, cols, ld);
	return res;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*  matLayout, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatrixLayoutCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTCREATE;
    
    struct cublasLtMatrixLayoutCreateArg *arg_ptr = (struct cublasLtMatrixLayoutCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matLayout = matLayout;
	arg_ptr->type = type;
	arg_ptr->rows = rows;
	arg_ptr->cols = cols;
	arg_ptr->ld = ld;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasLtMatrixLayoutCreateResponse *) dat;
	*matLayout = res->matLayout;
return res->err;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t  matLayout)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatrixLayoutDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTDESTROY;
    
    struct cublasLtMatrixLayoutDestroyArg *arg_ptr = (struct cublasLtMatrixLayoutDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matLayout = matLayout;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixLayoutGetAttribute(matLayout, attr, buf, sizeInBytes, sizeWritten);
	return res;
}

cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t  matmulDesc, size_t  size, cublasComputeType_t  computeType, cudaDataType_t  scaleType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulDescInit_internal(matmulDesc, size, computeType, scaleType);
	return res;
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*  matmulDesc, cublasComputeType_t  computeType, cudaDataType_t  scaleType)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulDescCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCCREATE;
    
    struct cublasLtMatmulDescCreateArg *arg_ptr = (struct cublasLtMatmulDescCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matmulDesc = matmulDesc;
	arg_ptr->computeType = computeType;
	arg_ptr->scaleType = scaleType;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasLtMatmulDescCreateResponse *) dat;
	*matmulDesc = res->matmulDesc;
return res->err;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t  matmulDesc)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulDescDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCDESTROY;
    
    struct cublasLtMatmulDescDestroyArg *arg_ptr = (struct cublasLtMatmulDescDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->matmulDesc = matmulDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulDescGetAttribute(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
	return res;
}

cublasStatus_t cublasLtMatrixTransformDescInit_internal(cublasLtMatrixTransformDesc_t  transformDesc, size_t  size, cudaDataType  scaleType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixTransformDescInit_internal(transformDesc, size, scaleType);
	return res;
}

cublasStatus_t cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t*  transformDesc, cudaDataType  scaleType)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixTransformDescCreate(transformDesc, scaleType);
	return res;
}

cublasStatus_t cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t  transformDesc)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixTransformDescDestroy(transformDesc);
	return res;
}

cublasStatus_t cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixTransformDescSetAttribute(transformDesc, attr, buf, sizeInBytes);
	return res;
}

cublasStatus_t cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatrixTransformDescGetAttribute(transformDesc, attr, buf, sizeInBytes, sizeWritten);
	return res;
}

cublasStatus_t cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t  pref, size_t  size)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulPreferenceInit_internal(pref, size);
	return res;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*  pref)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulPreferenceCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCECREATE;
    
    struct cublasLtMatmulPreferenceCreateArg *arg_ptr = (struct cublasLtMatmulPreferenceCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pref = pref;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasLtMatmulPreferenceCreateResponse *) dat;
	*pref = res->pref;
return res->err;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t  pref)
{

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulPreferenceDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCEDESTROY;
    
    struct cublasLtMatmulPreferenceDestroyArg *arg_ptr = (struct cublasLtMatmulPreferenceDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pref = pref;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulPreferenceGetAttribute(pref, attr, buf, sizeInBytes, sizeWritten);
	return res;
}

cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  requestedAlgoCount, int  algoIdsArray[], int*  returnAlgoCount)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulAlgoGetIds(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, requestedAlgoCount, algoIdsArray, returnAlgoCount);
	return res;
}

cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  algoId, cublasLtMatmulAlgo_t*  algo)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulAlgoInit(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo);
	return res;
}

cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulHeuristicResult_t*  result)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulAlgoCheck(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result);
	return res;
}

cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoCapAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulAlgoCapGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten);
	return res;
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulAlgoConfigSetAttribute(algo, attr, buf, sizeInBytes);
	return res;
}

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtMatmulAlgoConfigGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten);
	return res;
}

cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t  callback)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtLoggerSetCallback(callback);
	return res;
}

cublasStatus_t cublasLtLoggerSetFile(FILE*  file)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtLoggerSetFile(file);
	return res;
}

cublasStatus_t cublasLtLoggerOpenFile(const char*  logFile)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtLoggerOpenFile(logFile);
	return res;
}

cublasStatus_t cublasLtLoggerSetLevel(int  level)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtLoggerSetLevel(level);
	return res;
}

cublasStatus_t cublasLtLoggerSetMask(int  mask)
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtLoggerSetMask(mask);
	return res;
}

cublasStatus_t cublasLtLoggerForceDisable()
{
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
	cublasStatus_t res = 
		lcublasLtLoggerForceDisable();
	return res;
}



}

