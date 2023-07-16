
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
#include "tally/log.h"
#include "tally/ipc_util.h"
#include "tally/generated/cuda_api.h"
#include "tally/generated/cuda_api_enum.h"
#include "tally/generated/msg_struct.h"



extern "C" { 

CUresult cuGetErrorString(CUresult  error, const char ** pStr)
{
	TALLY_LOG("cuGetErrorString hooked");
	CUresult res = 
		lcuGetErrorString(error, pStr);
	return res;
}

CUresult cuGetErrorName(CUresult  error, const char ** pStr)
{
	TALLY_LOG("cuGetErrorName hooked");
	CUresult res = 
		lcuGetErrorName(error, pStr);
	return res;
}

CUresult cuInit(unsigned int  Flags)
{
	TALLY_LOG("cuInit hooked");

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
	TALLY_LOG("cuDriverGetVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDriverGetVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDRIVERGETVERSION;
    
    struct cuDriverGetVersionArg *arg_ptr = (struct cuDriverGetVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->driverVersion = driverVersion;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDriverGetVersionResponse *) dat;
	if (driverVersion) { *driverVersion = res->driverVersion; }
	return res->err;
}

CUresult cuDeviceGet(CUdevice * device, int  ordinal)
{
	TALLY_LOG("cuDeviceGet hooked");

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
	if (device) { *device = res->device; }
	return res->err;
}

CUresult cuDeviceGetCount(int * count)
{
	TALLY_LOG("cuDeviceGetCount hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetCountArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETCOUNT;
    
    struct cuDeviceGetCountArg *arg_ptr = (struct cuDeviceGetCountArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->count = count;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetCountResponse *) dat;
	if (count) { *count = res->count; }
	return res->err;
}

CUresult cuDeviceGetName(char * name, int  len, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGetUuid(CUuuid * uuid, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetUuid hooked");

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
	if (uuid) { *uuid = res->uuid; }
	return res->err;
}

CUresult cuDeviceGetUuid_v2(CUuuid * uuid, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetUuid_v2 hooked");

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
	if (uuid) { *uuid = res->uuid; }
	return res->err;
}

CUresult cuDeviceGetLuid(char * luid, unsigned int * deviceNodeMask, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetLuid hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetLuidArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETLUID;
    
    struct cuDeviceGetLuidArg *arg_ptr = (struct cuDeviceGetLuidArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->luid = luid;
	arg_ptr->deviceNodeMask = deviceNodeMask;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetLuidResponse *) dat;
	if (luid) { *luid = res->luid; }
	if (deviceNodeMask) { *deviceNodeMask = res->deviceNodeMask; }
	return res->err;
}

CUresult cuDeviceTotalMem_v2(size_t * bytes, CUdevice  dev)
{
	TALLY_LOG("cuDeviceTotalMem_v2 hooked");

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
	if (bytes) { *bytes = res->bytes; }
	return res->err;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, CUarray_format  format, unsigned  numChannels, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetTexture1DLinearMaxWidth hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetTexture1DLinearMaxWidthArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETTEXTURE1DLINEARMAXWIDTH;
    
    struct cuDeviceGetTexture1DLinearMaxWidthArg *arg_ptr = (struct cuDeviceGetTexture1DLinearMaxWidthArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->maxWidthInElements = maxWidthInElements;
	arg_ptr->format = format;
	arg_ptr->numChannels = numChannels;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetTexture1DLinearMaxWidthResponse *) dat;
	if (maxWidthInElements) { *maxWidthInElements = res->maxWidthInElements; }
	return res->err;
}

CUresult cuDeviceGetAttribute(int * pi, CUdevice_attribute  attrib, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetAttribute hooked");

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
	if (pi) { *pi = res->pi; }
	return res->err;
}

CUresult cuDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, CUdevice  dev, int  flags)
{
	TALLY_LOG("cuDeviceGetNvSciSyncAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceSetMemPool(CUdevice  dev, CUmemoryPool  pool)
{
	TALLY_LOG("cuDeviceSetMemPool hooked");

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
	TALLY_LOG("cuDeviceGetMemPool hooked");

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
	if (pool) { *pool = res->pool; }
	return res->err;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool * pool_out, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetDefaultMemPool hooked");

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
	if (pool_out) { *pool_out = res->pool_out; }
	return res->err;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget  target, CUflushGPUDirectRDMAWritesScope  scope)
{
	TALLY_LOG("cuFlushGPUDirectRDMAWrites hooked");

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
	TALLY_LOG("cuDeviceGetProperties hooked");

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
	if (prop) { *prop = res->prop; }
	return res->err;
}

CUresult cuDeviceComputeCapability(int * major, int * minor, CUdevice  dev)
{
	TALLY_LOG("cuDeviceComputeCapability hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceComputeCapabilityArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICECOMPUTECAPABILITY;
    
    struct cuDeviceComputeCapabilityArg *arg_ptr = (struct cuDeviceComputeCapabilityArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->major = major;
	arg_ptr->minor = minor;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceComputeCapabilityResponse *) dat;
	if (major) { *major = res->major; }
	if (minor) { *minor = res->minor; }
	return res->err;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext * pctx, CUdevice  dev)
{
	TALLY_LOG("cuDevicePrimaryCtxRetain hooked");

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
	if (pctx) { *pctx = res->pctx; }
	return res->err;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice  dev)
{
	TALLY_LOG("cuDevicePrimaryCtxRelease_v2 hooked");

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
	TALLY_LOG("cuDevicePrimaryCtxSetFlags_v2 hooked");

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
	TALLY_LOG("cuDevicePrimaryCtxGetState hooked");

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
	if (flags) { *flags = res->flags; }
	if (active) { *active = res->active; }
	return res->err;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice  dev)
{
	TALLY_LOG("cuDevicePrimaryCtxReset_v2 hooked");

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
	TALLY_LOG("cuDeviceGetExecAffinitySupport hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDeviceGetExecAffinitySupportArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDEVICEGETEXECAFFINITYSUPPORT;
    
    struct cuDeviceGetExecAffinitySupportArg *arg_ptr = (struct cuDeviceGetExecAffinitySupportArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pi = pi;
	arg_ptr->type = type;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuDeviceGetExecAffinitySupportResponse *) dat;
	if (pi) { *pi = res->pi; }
	return res->err;
}

CUresult cuCtxCreate_v2(CUcontext * pctx, unsigned int  flags, CUdevice  dev)
{
	TALLY_LOG("cuCtxCreate_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxCreate_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXCREATE_V2;
    
    struct cuCtxCreate_v2Arg *arg_ptr = (struct cuCtxCreate_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pctx = pctx;
	arg_ptr->flags = flags;
	arg_ptr->dev = dev;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxCreate_v2Response *) dat;
	if (pctx) { *pctx = res->pctx; }
	return res->err;
}

CUresult cuCtxCreate_v3(CUcontext * pctx, CUexecAffinityParam * paramsArray, int  numParams, unsigned int  flags, CUdevice  dev)
{
	TALLY_LOG("cuCtxCreate_v3 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCtxDestroy_v2(CUcontext  ctx)
{
	TALLY_LOG("cuCtxDestroy_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxDestroy_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXDESTROY_V2;
    
    struct cuCtxDestroy_v2Arg *arg_ptr = (struct cuCtxDestroy_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->ctx = ctx;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxPushCurrent_v2(CUcontext  ctx)
{
	TALLY_LOG("cuCtxPushCurrent_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxPushCurrent_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXPUSHCURRENT_V2;
    
    struct cuCtxPushCurrent_v2Arg *arg_ptr = (struct cuCtxPushCurrent_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->ctx = ctx;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxPopCurrent_v2(CUcontext * pctx)
{
	TALLY_LOG("cuCtxPopCurrent_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxPopCurrent_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXPOPCURRENT_V2;
    
    struct cuCtxPopCurrent_v2Arg *arg_ptr = (struct cuCtxPopCurrent_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pctx = pctx;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxPopCurrent_v2Response *) dat;
	if (pctx) { *pctx = res->pctx; }
	return res->err;
}

CUresult cuCtxSetCurrent(CUcontext  ctx)
{
	TALLY_LOG("cuCtxSetCurrent hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxSetCurrentArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXSETCURRENT;
    
    struct cuCtxSetCurrentArg *arg_ptr = (struct cuCtxSetCurrentArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->ctx = ctx;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxGetCurrent(CUcontext * pctx)
{
	TALLY_LOG("cuCtxGetCurrent hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetCurrentArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETCURRENT;
    
    struct cuCtxGetCurrentArg *arg_ptr = (struct cuCtxGetCurrentArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pctx = pctx;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetCurrentResponse *) dat;
	if (pctx) { *pctx = res->pctx; }
	return res->err;
}

CUresult cuCtxGetDevice(CUdevice * device)
{
	TALLY_LOG("cuCtxGetDevice hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetDeviceArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETDEVICE;
    
    struct cuCtxGetDeviceArg *arg_ptr = (struct cuCtxGetDeviceArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetDeviceResponse *) dat;
	if (device) { *device = res->device; }
	return res->err;
}

CUresult cuCtxGetFlags(unsigned int * flags)
{
	TALLY_LOG("cuCtxGetFlags hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETFLAGS;
    
    struct cuCtxGetFlagsArg *arg_ptr = (struct cuCtxGetFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetFlagsResponse *) dat;
	if (flags) { *flags = res->flags; }
	return res->err;
}

CUresult cuCtxSynchronize()
{
	TALLY_LOG("cuCtxSynchronize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxSynchronizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXSYNCHRONIZE;
    
    struct cuCtxSynchronizeArg *arg_ptr = (struct cuCtxSynchronizeArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxSetLimit(CUlimit  limit, size_t  value)
{
	TALLY_LOG("cuCtxSetLimit hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxSetLimitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXSETLIMIT;
    
    struct cuCtxSetLimitArg *arg_ptr = (struct cuCtxSetLimitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->limit = limit;
	arg_ptr->value = value;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxGetLimit(size_t * pvalue, CUlimit  limit)
{
	TALLY_LOG("cuCtxGetLimit hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetLimitArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETLIMIT;
    
    struct cuCtxGetLimitArg *arg_ptr = (struct cuCtxGetLimitArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pvalue = pvalue;
	arg_ptr->limit = limit;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetLimitResponse *) dat;
	if (pvalue) { *pvalue = res->pvalue; }
	return res->err;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache * pconfig)
{
	TALLY_LOG("cuCtxGetCacheConfig hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETCACHECONFIG;
    
    struct cuCtxGetCacheConfigArg *arg_ptr = (struct cuCtxGetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pconfig = pconfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetCacheConfigResponse *) dat;
	if (pconfig) { *pconfig = res->pconfig; }
	return res->err;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache  config)
{
	TALLY_LOG("cuCtxSetCacheConfig hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxSetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXSETCACHECONFIG;
    
    struct cuCtxSetCacheConfigArg *arg_ptr = (struct cuCtxSetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->config = config;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig)
{
	TALLY_LOG("cuCtxGetSharedMemConfig hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetSharedMemConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETSHAREDMEMCONFIG;
    
    struct cuCtxGetSharedMemConfigArg *arg_ptr = (struct cuCtxGetSharedMemConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pConfig = pConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetSharedMemConfigResponse *) dat;
	if (pConfig) { *pConfig = res->pConfig; }
	return res->err;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig  config)
{
	TALLY_LOG("cuCtxSetSharedMemConfig hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxSetSharedMemConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXSETSHAREDMEMCONFIG;
    
    struct cuCtxSetSharedMemConfigArg *arg_ptr = (struct cuCtxSetSharedMemConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->config = config;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxGetApiVersion(CUcontext  ctx, unsigned int * version)
{
	TALLY_LOG("cuCtxGetApiVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetApiVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETAPIVERSION;
    
    struct cuCtxGetApiVersionArg *arg_ptr = (struct cuCtxGetApiVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->ctx = ctx;
	arg_ptr->version = version;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetApiVersionResponse *) dat;
	if (version) { *version = res->version; }
	return res->err;
}

CUresult cuCtxGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	TALLY_LOG("cuCtxGetStreamPriorityRange hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetStreamPriorityRangeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETSTREAMPRIORITYRANGE;
    
    struct cuCtxGetStreamPriorityRangeArg *arg_ptr = (struct cuCtxGetStreamPriorityRangeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->leastPriority = leastPriority;
	arg_ptr->greatestPriority = greatestPriority;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetStreamPriorityRangeResponse *) dat;
	if (leastPriority) { *leastPriority = res->leastPriority; }
	if (greatestPriority) { *greatestPriority = res->greatestPriority; }
	return res->err;
}

CUresult cuCtxResetPersistingL2Cache()
{
	TALLY_LOG("cuCtxResetPersistingL2Cache hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxResetPersistingL2CacheArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXRESETPERSISTINGL2CACHE;
    
    struct cuCtxResetPersistingL2CacheArg *arg_ptr = (struct cuCtxResetPersistingL2CacheArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType  type)
{
	TALLY_LOG("cuCtxGetExecAffinity hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxGetExecAffinityArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXGETEXECAFFINITY;
    
    struct cuCtxGetExecAffinityArg *arg_ptr = (struct cuCtxGetExecAffinityArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pExecAffinity = pExecAffinity;
	arg_ptr->type = type;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxGetExecAffinityResponse *) dat;
	if (pExecAffinity) { *pExecAffinity = res->pExecAffinity; }
	return res->err;
}

CUresult cuCtxAttach(CUcontext * pctx, unsigned int  flags)
{
	TALLY_LOG("cuCtxAttach hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxAttachArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXATTACH;
    
    struct cuCtxAttachArg *arg_ptr = (struct cuCtxAttachArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pctx = pctx;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuCtxAttachResponse *) dat;
	if (pctx) { *pctx = res->pctx; }
	return res->err;
}

CUresult cuCtxDetach(CUcontext  ctx)
{
	TALLY_LOG("cuCtxDetach hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuCtxDetachArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUCTXDETACH;
    
    struct cuCtxDetachArg *arg_ptr = (struct cuCtxDetachArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->ctx = ctx;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuModuleLoad(CUmodule * module, const char * fname)
{
	TALLY_LOG("cuModuleLoad hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleLoadData(CUmodule * module, const void * image)
{
	TALLY_LOG("cuModuleLoadData hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleLoadDataEx(CUmodule * module, const void * image, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	TALLY_LOG("cuModuleLoadDataEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleLoadFatBinary(CUmodule * module, const void * fatCubin)
{
	TALLY_LOG("cuModuleLoadFatBinary hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleUnload(CUmodule  hmod)
{
	TALLY_LOG("cuModuleUnload hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuModuleUnloadArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUMODULEUNLOAD;
    
    struct cuModuleUnloadArg *arg_ptr = (struct cuModuleUnloadArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->hmod = hmod;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule  hmod, const char * name)
{
	TALLY_LOG("cuModuleGetFunction hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule  hmod, const char * name)
{
	TALLY_LOG("cuModuleGetGlobal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleGetTexRef(CUtexref * pTexRef, CUmodule  hmod, const char * name)
{
	TALLY_LOG("cuModuleGetTexRef hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleGetSurfRef(CUsurfref * pSurfRef, CUmodule  hmod, const char * name)
{
	TALLY_LOG("cuModuleGetSurfRef hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLinkCreate_v2(unsigned int  numOptions, CUjit_option * options, void ** optionValues, CUlinkState * stateOut)
{
	TALLY_LOG("cuLinkCreate_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLinkAddData_v2(CUlinkState  state, CUjitInputType  type, void * data, size_t  size, const char * name, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	TALLY_LOG("cuLinkAddData_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLinkAddFile_v2(CUlinkState  state, CUjitInputType  type, const char * path, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	TALLY_LOG("cuLinkAddFile_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLinkComplete(CUlinkState  state, void ** cubinOut, size_t * sizeOut)
{
	TALLY_LOG("cuLinkComplete hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLinkDestroy(CUlinkState  state)
{
	TALLY_LOG("cuLinkDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemGetInfo_v2(size_t * free, size_t * total)
{
	TALLY_LOG("cuMemGetInfo_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAlloc_v2(CUdeviceptr * dptr, size_t  bytesize)
{
	TALLY_LOG("cuMemAlloc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAllocPitch_v2(CUdeviceptr * dptr, size_t * pPitch, size_t  WidthInBytes, size_t  Height, unsigned int  ElementSizeBytes)
{
	TALLY_LOG("cuMemAllocPitch_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemFree_v2(CUdeviceptr  dptr)
{
	TALLY_LOG("cuMemFree_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuMemFree_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUMEMFREE_V2;
    
    struct cuMemFree_v2Arg *arg_ptr = (struct cuMemFree_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dptr = dptr;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr * pbase, size_t * psize, CUdeviceptr  dptr)
{
	TALLY_LOG("cuMemGetAddressRange_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAllocHost_v2(void ** pp, size_t  bytesize)
{
	TALLY_LOG("cuMemAllocHost_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemFreeHost(void * p)
{
	TALLY_LOG("cuMemFreeHost hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemHostAlloc(void ** pp, size_t  bytesize, unsigned int  Flags)
{
	TALLY_LOG("cuMemHostAlloc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr * pdptr, void * p, unsigned int  Flags)
{
	TALLY_LOG("cuMemHostGetDevicePointer_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemHostGetFlags(unsigned int * pFlags, void * p)
{
	TALLY_LOG("cuMemHostGetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAllocManaged(CUdeviceptr * dptr, size_t  bytesize, unsigned int  flags)
{
	TALLY_LOG("cuMemAllocManaged hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGetByPCIBusId(CUdevice * dev, const char * pciBusId)
{
	TALLY_LOG("cuDeviceGetByPCIBusId hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGetPCIBusId(char * pciBusId, int  len, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetPCIBusId hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuIpcGetEventHandle(CUipcEventHandle * pHandle, CUevent  event)
{
	TALLY_LOG("cuIpcGetEventHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuIpcOpenEventHandle(CUevent * phEvent, CUipcEventHandle  handle)
{
	TALLY_LOG("cuIpcOpenEventHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuIpcGetMemHandle(CUipcMemHandle * pHandle, CUdeviceptr  dptr)
{
	TALLY_LOG("cuIpcGetMemHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr * pdptr, CUipcMemHandle  handle, unsigned int  Flags)
{
	TALLY_LOG("cuIpcOpenMemHandle_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuIpcCloseMemHandle(CUdeviceptr  dptr)
{
	TALLY_LOG("cuIpcCloseMemHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemHostRegister_v2(void * p, size_t  bytesize, unsigned int  Flags)
{
	TALLY_LOG("cuMemHostRegister_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemHostUnregister(void * p)
{
	TALLY_LOG("cuMemHostUnregister hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy(CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyPeer(CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyPeer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyHtoD_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyDtoH_v2(void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyDtoH_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyDtoD_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyDtoA_v2(CUarray  dstArray, size_t  dstOffset, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyDtoA_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr  dstDevice, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyAtoD_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyHtoA_v2(CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyHtoA_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyAtoH_v2(void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyAtoH_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyAtoA_v2(CUarray  dstArray, size_t  dstOffset, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	TALLY_LOG("cuMemcpyAtoA_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D * pCopy)
{
	TALLY_LOG("cuMemcpy2D_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * pCopy)
{
	TALLY_LOG("cuMemcpy2DUnaligned_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D * pCopy)
{
	TALLY_LOG("cuMemcpy3D_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER * pCopy)
{
	TALLY_LOG("cuMemcpy3DPeer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyAsync(CUdeviceptr  dst, CUdeviceptr  src, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyAsync hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuMemcpyAsyncArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUMEMCPYASYNC;
    
    struct cuMemcpyAsyncArg *arg_ptr = (struct cuMemcpyAsyncArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dst = dst;
	arg_ptr->src = src;
	arg_ptr->ByteCount = ByteCount;
	arg_ptr->hStream = hStream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyPeerAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyHtoDAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyDtoHAsync_v2(void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyDtoHAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyDtoDAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyHtoAAsync_v2(CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyHtoAAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpyAtoHAsync_v2(void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount, CUstream  hStream)
{
	TALLY_LOG("cuMemcpyAtoHAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * pCopy, CUstream  hStream)
{
	TALLY_LOG("cuMemcpy2DAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * pCopy, CUstream  hStream)
{
	TALLY_LOG("cuMemcpy3DAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER * pCopy, CUstream  hStream)
{
	TALLY_LOG("cuMemcpy3DPeerAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD8_v2(CUdeviceptr  dstDevice, unsigned char  uc, size_t  N)
{
	TALLY_LOG("cuMemsetD8_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD16_v2(CUdeviceptr  dstDevice, unsigned short  us, size_t  N)
{
	TALLY_LOG("cuMemsetD16_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD32_v2(CUdeviceptr  dstDevice, unsigned int  ui, size_t  N)
{
	TALLY_LOG("cuMemsetD32_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD2D8_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height)
{
	TALLY_LOG("cuMemsetD2D8_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD2D16_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height)
{
	TALLY_LOG("cuMemsetD2D16_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD2D32_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height)
{
	TALLY_LOG("cuMemsetD2D32_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD8Async(CUdeviceptr  dstDevice, unsigned char  uc, size_t  N, CUstream  hStream)
{
	TALLY_LOG("cuMemsetD8Async hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD16Async(CUdeviceptr  dstDevice, unsigned short  us, size_t  N, CUstream  hStream)
{
	TALLY_LOG("cuMemsetD16Async hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD32Async(CUdeviceptr  dstDevice, unsigned int  ui, size_t  N, CUstream  hStream)
{
	TALLY_LOG("cuMemsetD32Async hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD2D8Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height, CUstream  hStream)
{
	TALLY_LOG("cuMemsetD2D8Async hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD2D16Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height, CUstream  hStream)
{
	TALLY_LOG("cuMemsetD2D16Async hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemsetD2D32Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height, CUstream  hStream)
{
	TALLY_LOG("cuMemsetD2D32Async hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArrayCreate_v2(CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray)
{
	TALLY_LOG("cuArrayCreate_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)
{
	TALLY_LOG("cuArrayGetDescriptor_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray  array)
{
	TALLY_LOG("cuArrayGetSparseProperties hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUmipmappedArray  mipmap)
{
	TALLY_LOG("cuMipmappedArrayGetSparseProperties hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArrayGetPlane(CUarray * pPlaneArray, CUarray  hArray, unsigned int  planeIdx)
{
	TALLY_LOG("cuArrayGetPlane hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArrayDestroy(CUarray  hArray)
{
	TALLY_LOG("cuArrayDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArray3DCreate_v2(CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pAllocateArray)
{
	TALLY_LOG("cuArray3DCreate_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)
{
	TALLY_LOG("cuArray3DGetDescriptor_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int  numMipmapLevels)
{
	TALLY_LOG("cuMipmappedArrayCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMipmappedArrayGetLevel(CUarray * pLevelArray, CUmipmappedArray  hMipmappedArray, unsigned int  level)
{
	TALLY_LOG("cuMipmappedArrayGetLevel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray  hMipmappedArray)
{
	TALLY_LOG("cuMipmappedArrayDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAddressReserve(CUdeviceptr * ptr, size_t  size, size_t  alignment, CUdeviceptr  addr, unsigned long long  flags)
{
	TALLY_LOG("cuMemAddressReserve hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAddressFree(CUdeviceptr  ptr, size_t  size)
{
	TALLY_LOG("cuMemAddressFree hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemCreate(CUmemGenericAllocationHandle * handle, size_t  size, const CUmemAllocationProp * prop, unsigned long long  flags)
{
	TALLY_LOG("cuMemCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemRelease(CUmemGenericAllocationHandle  handle)
{
	TALLY_LOG("cuMemRelease hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemMap(CUdeviceptr  ptr, size_t  size, size_t  offset, CUmemGenericAllocationHandle  handle, unsigned long long  flags)
{
	TALLY_LOG("cuMemMap hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo * mapInfoList, unsigned int  count, CUstream  hStream)
{
	TALLY_LOG("cuMemMapArrayAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemUnmap(CUdeviceptr  ptr, size_t  size)
{
	TALLY_LOG("cuMemUnmap hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemSetAccess(CUdeviceptr  ptr, size_t  size, const CUmemAccessDesc * desc, size_t  count)
{
	TALLY_LOG("cuMemSetAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemGetAccess(unsigned long long * flags, const CUmemLocation * location, CUdeviceptr  ptr)
{
	TALLY_LOG("cuMemGetAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemExportToShareableHandle(void * shareableHandle, CUmemGenericAllocationHandle  handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	TALLY_LOG("cuMemExportToShareableHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle * handle, void * osHandle, CUmemAllocationHandleType  shHandleType)
{
	TALLY_LOG("cuMemImportFromShareableHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemGetAllocationGranularity(size_t * granularity, const CUmemAllocationProp * prop, CUmemAllocationGranularity_flags  option)
{
	TALLY_LOG("cuMemGetAllocationGranularity hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp * prop, CUmemGenericAllocationHandle  handle)
{
	TALLY_LOG("cuMemGetAllocationPropertiesFromHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle * handle, void * addr)
{
	TALLY_LOG("cuMemRetainAllocationHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemFreeAsync(CUdeviceptr  dptr, CUstream  hStream)
{
	TALLY_LOG("cuMemFreeAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAllocAsync(CUdeviceptr * dptr, size_t  bytesize, CUstream  hStream)
{
	TALLY_LOG("cuMemAllocAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolTrimTo(CUmemoryPool  pool, size_t  minBytesToKeep)
{
	TALLY_LOG("cuMemPoolTrimTo hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolSetAttribute(CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)
{
	TALLY_LOG("cuMemPoolSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolGetAttribute(CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)
{
	TALLY_LOG("cuMemPoolGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolSetAccess(CUmemoryPool  pool, const CUmemAccessDesc * map, size_t  count)
{
	TALLY_LOG("cuMemPoolSetAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags * flags, CUmemoryPool  memPool, CUmemLocation * location)
{
	TALLY_LOG("cuMemPoolGetAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolCreate(CUmemoryPool * pool, const CUmemPoolProps * poolProps)
{
	TALLY_LOG("cuMemPoolCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolDestroy(CUmemoryPool  pool)
{
	TALLY_LOG("cuMemPoolDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr * dptr, size_t  bytesize, CUmemoryPool  pool, CUstream  hStream)
{
	TALLY_LOG("cuMemAllocFromPoolAsync hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuMemAllocFromPoolAsyncArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUMEMALLOCFROMPOOLASYNC;
    
    struct cuMemAllocFromPoolAsyncArg *arg_ptr = (struct cuMemAllocFromPoolAsyncArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dptr = dptr;
	arg_ptr->bytesize = bytesize;
	arg_ptr->pool = pool;
	arg_ptr->hStream = hStream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cuMemAllocFromPoolAsyncResponse *) dat;
	if (dptr) { *dptr = res->dptr; }
	return res->err;
}

CUresult cuMemPoolExportToShareableHandle(void * handle_out, CUmemoryPool  pool, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	TALLY_LOG("cuMemPoolExportToShareableHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool * pool_out, void * handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	TALLY_LOG("cuMemPoolImportFromShareableHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData * shareData_out, CUdeviceptr  ptr)
{
	TALLY_LOG("cuMemPoolExportPointer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPoolImportPointer(CUdeviceptr * ptr_out, CUmemoryPool  pool, CUmemPoolPtrExportData * shareData)
{
	TALLY_LOG("cuMemPoolImportPointer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuPointerGetAttribute(void * data, CUpointer_attribute  attribute, CUdeviceptr  ptr)
{
	TALLY_LOG("cuPointerGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPrefetchAsync(CUdeviceptr  devPtr, size_t  count, CUdevice  dstDevice, CUstream  hStream)
{
	TALLY_LOG("cuMemPrefetchAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAdvise(CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUdevice  device)
{
	TALLY_LOG("cuMemAdvise hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemRangeGetAttribute(void * data, size_t  dataSize, CUmem_range_attribute  attribute, CUdeviceptr  devPtr, size_t  count)
{
	TALLY_LOG("cuMemRangeGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemRangeGetAttributes(void ** data, size_t * dataSizes, CUmem_range_attribute * attributes, size_t  numAttributes, CUdeviceptr  devPtr, size_t  count)
{
	TALLY_LOG("cuMemRangeGetAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuPointerSetAttribute(const void * value, CUpointer_attribute  attribute, CUdeviceptr  ptr)
{
	TALLY_LOG("cuPointerSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuPointerGetAttributes(unsigned int  numAttributes, CUpointer_attribute * attributes, void ** data, CUdeviceptr  ptr)
{
	TALLY_LOG("cuPointerGetAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamCreate(CUstream * phStream, unsigned int  Flags)
{
	TALLY_LOG("cuStreamCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamCreateWithPriority(CUstream * phStream, unsigned int  flags, int  priority)
{
	TALLY_LOG("cuStreamCreateWithPriority hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamGetPriority(CUstream  hStream, int * priority)
{
	TALLY_LOG("cuStreamGetPriority hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamGetFlags(CUstream  hStream, unsigned int * flags)
{
	TALLY_LOG("cuStreamGetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamGetCtx(CUstream  hStream, CUcontext * pctx)
{
	TALLY_LOG("cuStreamGetCtx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWaitEvent(CUstream  hStream, CUevent  hEvent, unsigned int  Flags)
{
	TALLY_LOG("cuStreamWaitEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamAddCallback(CUstream  hStream, CUstreamCallback  callback, void * userData, unsigned int  flags)
{
	TALLY_LOG("cuStreamAddCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamBeginCapture_v2(CUstream  hStream, CUstreamCaptureMode  mode)
{
	TALLY_LOG("cuStreamBeginCapture_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode * mode)
{
	TALLY_LOG("cuThreadExchangeStreamCaptureMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamEndCapture(CUstream  hStream, CUgraph * phGraph)
{
	TALLY_LOG("cuStreamEndCapture hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamIsCapturing(CUstream  hStream, CUstreamCaptureStatus * captureStatus)
{
	TALLY_LOG("cuStreamIsCapturing hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamGetCaptureInfo(CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out)
{
	TALLY_LOG("cuStreamGetCaptureInfo hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamGetCaptureInfo_v2(CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, CUgraph * graph_out, const CUgraphNode ** dependencies_out, size_t * numDependencies_out)
{
	TALLY_LOG("cuStreamGetCaptureInfo_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamUpdateCaptureDependencies(CUstream  hStream, CUgraphNode * dependencies, size_t  numDependencies, unsigned int  flags)
{
	TALLY_LOG("cuStreamUpdateCaptureDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamAttachMemAsync(CUstream  hStream, CUdeviceptr  dptr, size_t  length, unsigned int  flags)
{
	TALLY_LOG("cuStreamAttachMemAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamQuery(CUstream  hStream)
{
	TALLY_LOG("cuStreamQuery hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamSynchronize(CUstream  hStream)
{
	TALLY_LOG("cuStreamSynchronize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamDestroy_v2(CUstream  hStream)
{
	TALLY_LOG("cuStreamDestroy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamCopyAttributes(CUstream  dst, CUstream  src)
{
	TALLY_LOG("cuStreamCopyAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamGetAttribute(CUstream  hStream, CUstreamAttrID  attr, CUstreamAttrValue * value_out)
{
	TALLY_LOG("cuStreamGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamSetAttribute(CUstream  hStream, CUstreamAttrID  attr, const CUstreamAttrValue * value)
{
	TALLY_LOG("cuStreamSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventCreate(CUevent * phEvent, unsigned int  Flags)
{
	TALLY_LOG("cuEventCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventRecord(CUevent  hEvent, CUstream  hStream)
{
	TALLY_LOG("cuEventRecord hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventRecordWithFlags(CUevent  hEvent, CUstream  hStream, unsigned int  flags)
{
	TALLY_LOG("cuEventRecordWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventQuery(CUevent  hEvent)
{
	TALLY_LOG("cuEventQuery hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventSynchronize(CUevent  hEvent)
{
	TALLY_LOG("cuEventSynchronize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventDestroy_v2(CUevent  hEvent)
{
	TALLY_LOG("cuEventDestroy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuEventElapsedTime(float * pMilliseconds, CUevent  hStart, CUevent  hEnd)
{
	TALLY_LOG("cuEventElapsedTime hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuImportExternalMemory(CUexternalMemory * extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc)
{
	TALLY_LOG("cuImportExternalMemory hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr * devPtr, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc)
{
	TALLY_LOG("cuExternalMemoryGetMappedBuffer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray * mipmap, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc)
{
	TALLY_LOG("cuExternalMemoryGetMappedMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDestroyExternalMemory(CUexternalMemory  extMem)
{
	TALLY_LOG("cuDestroyExternalMemory hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cuDestroyExternalMemoryArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDESTROYEXTERNALMEMORY;
    
    struct cuDestroyExternalMemoryArg *arg_ptr = (struct cuDestroyExternalMemoryArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->extMem = extMem;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (CUresult *) dat;
    return *res;
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore * extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc)
{
	TALLY_LOG("cuImportExternalSemaphore hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)
{
	TALLY_LOG("cuSignalExternalSemaphoresAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)
{
	TALLY_LOG("cuWaitExternalSemaphoresAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore  extSem)
{
	TALLY_LOG("cuDestroyExternalSemaphore hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWaitValue32(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWaitValue32 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWaitValue64(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWaitValue64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWriteValue32(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWriteValue32 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWriteValue64(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWriteValue64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamBatchMemOp(CUstream  stream, unsigned int  count, CUstreamBatchMemOpParams * paramArray, unsigned int  flags)
{
	TALLY_LOG("cuStreamBatchMemOp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncGetAttribute(int * pi, CUfunction_attribute  attrib, CUfunction  hfunc)
{
	TALLY_LOG("cuFuncGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncSetAttribute(CUfunction  hfunc, CUfunction_attribute  attrib, int  value)
{
	TALLY_LOG("cuFuncSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncSetCacheConfig(CUfunction  hfunc, CUfunc_cache  config)
{
	TALLY_LOG("cuFuncSetCacheConfig hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncSetSharedMemConfig(CUfunction  hfunc, CUsharedconfig  config)
{
	TALLY_LOG("cuFuncSetSharedMemConfig hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncGetModule(CUmodule * hmod, CUfunction  hfunc)
{
	TALLY_LOG("cuFuncGetModule hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunchKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra)
{
	TALLY_LOG("cuLaunchKernel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunchCooperativeKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams)
{
	TALLY_LOG("cuLaunchCooperativeKernel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS * launchParamsList, unsigned int  numDevices, unsigned int  flags)
{
	TALLY_LOG("cuLaunchCooperativeKernelMultiDevice hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunchHostFunc(CUstream  hStream, CUhostFn  fn, void * userData)
{
	TALLY_LOG("cuLaunchHostFunc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncSetBlockShape(CUfunction  hfunc, int  x, int  y, int  z)
{
	TALLY_LOG("cuFuncSetBlockShape hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuFuncSetSharedSize(CUfunction  hfunc, unsigned int  bytes)
{
	TALLY_LOG("cuFuncSetSharedSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuParamSetSize(CUfunction  hfunc, unsigned int  numbytes)
{
	TALLY_LOG("cuParamSetSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuParamSeti(CUfunction  hfunc, int  offset, unsigned int  value)
{
	TALLY_LOG("cuParamSeti hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuParamSetf(CUfunction  hfunc, int  offset, float  value)
{
	TALLY_LOG("cuParamSetf hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuParamSetv(CUfunction  hfunc, int  offset, void * ptr, unsigned int  numbytes)
{
	TALLY_LOG("cuParamSetv hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunch(CUfunction  f)
{
	TALLY_LOG("cuLaunch hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunchGrid(CUfunction  f, int  grid_width, int  grid_height)
{
	TALLY_LOG("cuLaunchGrid hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLaunchGridAsync(CUfunction  f, int  grid_width, int  grid_height, CUstream  hStream)
{
	TALLY_LOG("cuLaunchGridAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuParamSetTexRef(CUfunction  hfunc, int  texunit, CUtexref  hTexRef)
{
	TALLY_LOG("cuParamSetTexRef hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphCreate(CUgraph * phGraph, unsigned int  flags)
{
	TALLY_LOG("cuGraphCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddKernelNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddKernelNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeGetParams(CUgraphNode  hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphKernelNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeSetParams(CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphKernelNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddMemcpyNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)
{
	TALLY_LOG("cuGraphAddMemcpyNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode  hNode, CUDA_MEMCPY3D * nodeParams)
{
	TALLY_LOG("cuGraphMemcpyNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode  hNode, const CUDA_MEMCPY3D * nodeParams)
{
	TALLY_LOG("cuGraphMemcpyNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddMemsetNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)
{
	TALLY_LOG("cuGraphAddMemsetNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode  hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphMemsetNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphMemsetNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddHostNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddHostNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphHostNodeGetParams(CUgraphNode  hNode, CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphHostNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphHostNodeSetParams(CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphHostNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddChildGraphNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraph  childGraph)
{
	TALLY_LOG("cuGraphAddChildGraphNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode  hNode, CUgraph * phGraph)
{
	TALLY_LOG("cuGraphChildGraphNodeGetGraph hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddEmptyNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies)
{
	TALLY_LOG("cuGraphAddEmptyNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddEventRecordNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)
{
	TALLY_LOG("cuGraphAddEventRecordNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode  hNode, CUevent * event_out)
{
	TALLY_LOG("cuGraphEventRecordNodeGetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode  hNode, CUevent  event)
{
	TALLY_LOG("cuGraphEventRecordNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddEventWaitNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)
{
	TALLY_LOG("cuGraphAddEventWaitNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode  hNode, CUevent * event_out)
{
	TALLY_LOG("cuGraphEventWaitNodeGetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode  hNode, CUevent  event)
{
	TALLY_LOG("cuGraphEventWaitNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddExternalSemaphoresSignalNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode  hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * params_out)
{
	TALLY_LOG("cuGraphExternalSemaphoresSignalNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExternalSemaphoresSignalNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddExternalSemaphoresWaitNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode  hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS * params_out)
{
	TALLY_LOG("cuGraphExternalSemaphoresWaitNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExternalSemaphoresWaitNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddMemAllocNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddMemAllocNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode  hNode, CUDA_MEM_ALLOC_NODE_PARAMS * params_out)
{
	TALLY_LOG("cuGraphMemAllocNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddMemFreeNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUdeviceptr  dptr)
{
	TALLY_LOG("cuGraphAddMemFreeNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode  hNode, CUdeviceptr * dptr_out)
{
	TALLY_LOG("cuGraphMemFreeNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGraphMemTrim(CUdevice  device)
{
	TALLY_LOG("cuDeviceGraphMemTrim hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGetGraphMemAttribute(CUdevice  device, CUgraphMem_attribute  attr, void*  value)
{
	TALLY_LOG("cuDeviceGetGraphMemAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceSetGraphMemAttribute(CUdevice  device, CUgraphMem_attribute  attr, void*  value)
{
	TALLY_LOG("cuDeviceSetGraphMemAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphClone(CUgraph * phGraphClone, CUgraph  originalGraph)
{
	TALLY_LOG("cuGraphClone hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphNodeFindInClone(CUgraphNode * phNode, CUgraphNode  hOriginalNode, CUgraph  hClonedGraph)
{
	TALLY_LOG("cuGraphNodeFindInClone hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphNodeGetType(CUgraphNode  hNode, CUgraphNodeType * type)
{
	TALLY_LOG("cuGraphNodeGetType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphGetNodes(CUgraph  hGraph, CUgraphNode * nodes, size_t * numNodes)
{
	TALLY_LOG("cuGraphGetNodes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphGetRootNodes(CUgraph  hGraph, CUgraphNode * rootNodes, size_t * numRootNodes)
{
	TALLY_LOG("cuGraphGetRootNodes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphGetEdges(CUgraph  hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges)
{
	TALLY_LOG("cuGraphGetEdges hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphNodeGetDependencies(CUgraphNode  hNode, CUgraphNode * dependencies, size_t * numDependencies)
{
	TALLY_LOG("cuGraphNodeGetDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode  hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes)
{
	TALLY_LOG("cuGraphNodeGetDependentNodes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphAddDependencies(CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)
{
	TALLY_LOG("cuGraphAddDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphRemoveDependencies(CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)
{
	TALLY_LOG("cuGraphRemoveDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphDestroyNode(CUgraphNode  hNode)
{
	TALLY_LOG("cuGraphDestroyNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphInstantiate_v2(CUgraphExec * phGraphExec, CUgraph  hGraph, CUgraphNode * phErrorNode, char * logBuffer, size_t  bufferSize)
{
	TALLY_LOG("cuGraphInstantiate_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec * phGraphExec, CUgraph  hGraph, unsigned long long  flags)
{
	TALLY_LOG("cuGraphInstantiateWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecKernelNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExecKernelNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)
{
	TALLY_LOG("cuGraphExecMemcpyNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)
{
	TALLY_LOG("cuGraphExecMemsetNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExecHostNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraph  childGraph)
{
	TALLY_LOG("cuGraphExecChildGraphNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)
{
	TALLY_LOG("cuGraphExecEventRecordNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)
{
	TALLY_LOG("cuGraphExecEventWaitNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExecExternalSemaphoresSignalNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExecExternalSemaphoresWaitNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphUpload(CUgraphExec  hGraphExec, CUstream  hStream)
{
	TALLY_LOG("cuGraphUpload hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphLaunch(CUgraphExec  hGraphExec, CUstream  hStream)
{
	TALLY_LOG("cuGraphLaunch hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecDestroy(CUgraphExec  hGraphExec)
{
	TALLY_LOG("cuGraphExecDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphDestroy(CUgraph  hGraph)
{
	TALLY_LOG("cuGraphDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecUpdate(CUgraphExec  hGraphExec, CUgraph  hGraph, CUgraphNode * hErrorNode_out, CUgraphExecUpdateResult * updateResult_out)
{
	TALLY_LOG("cuGraphExecUpdate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode  dst, CUgraphNode  src)
{
	TALLY_LOG("cuGraphKernelNodeCopyAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode  hNode, CUkernelNodeAttrID  attr, CUkernelNodeAttrValue * value_out)
{
	TALLY_LOG("cuGraphKernelNodeGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode  hNode, CUkernelNodeAttrID  attr, const CUkernelNodeAttrValue * value)
{
	TALLY_LOG("cuGraphKernelNodeSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphDebugDotPrint(CUgraph  hGraph, const char * path, unsigned int  flags)
{
	TALLY_LOG("cuGraphDebugDotPrint hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuUserObjectCreate(CUuserObject * object_out, void * ptr, CUhostFn  destroy, unsigned int  initialRefcount, unsigned int  flags)
{
	TALLY_LOG("cuUserObjectCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuUserObjectRetain(CUuserObject  object, unsigned int  count)
{
	TALLY_LOG("cuUserObjectRetain hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuUserObjectRelease(CUuserObject  object, unsigned int  count)
{
	TALLY_LOG("cuUserObjectRelease hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphRetainUserObject(CUgraph  graph, CUuserObject  object, unsigned int  count, unsigned int  flags)
{
	TALLY_LOG("cuGraphRetainUserObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphReleaseUserObject(CUgraph  graph, CUuserObject  object, unsigned int  count)
{
	TALLY_LOG("cuGraphReleaseUserObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize)
{
	TALLY_LOG("cuOccupancyMaxActiveBlocksPerMultiprocessor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	TALLY_LOG("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuOccupancyMaxPotentialBlockSize(int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit)
{
	TALLY_LOG("cuOccupancyMaxPotentialBlockSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit, unsigned int  flags)
{
	TALLY_LOG("cuOccupancyMaxPotentialBlockSizeWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, CUfunction  func, int  numBlocks, int  blockSize)
{
	TALLY_LOG("cuOccupancyAvailableDynamicSMemPerBlock hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetArray(CUtexref  hTexRef, CUarray  hArray, unsigned int  Flags)
{
	TALLY_LOG("cuTexRefSetArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetMipmappedArray(CUtexref  hTexRef, CUmipmappedArray  hMipmappedArray, unsigned int  Flags)
{
	TALLY_LOG("cuTexRefSetMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetAddress_v2(size_t * ByteOffset, CUtexref  hTexRef, CUdeviceptr  dptr, size_t  bytes)
{
	TALLY_LOG("cuTexRefSetAddress_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetAddress2D_v3(CUtexref  hTexRef, const CUDA_ARRAY_DESCRIPTOR * desc, CUdeviceptr  dptr, size_t  Pitch)
{
	TALLY_LOG("cuTexRefSetAddress2D_v3 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetFormat(CUtexref  hTexRef, CUarray_format  fmt, int  NumPackedComponents)
{
	TALLY_LOG("cuTexRefSetFormat hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetAddressMode(CUtexref  hTexRef, int  dim, CUaddress_mode  am)
{
	TALLY_LOG("cuTexRefSetAddressMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetFilterMode(CUtexref  hTexRef, CUfilter_mode  fm)
{
	TALLY_LOG("cuTexRefSetFilterMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref  hTexRef, CUfilter_mode  fm)
{
	TALLY_LOG("cuTexRefSetMipmapFilterMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref  hTexRef, float  bias)
{
	TALLY_LOG("cuTexRefSetMipmapLevelBias hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref  hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp)
{
	TALLY_LOG("cuTexRefSetMipmapLevelClamp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref  hTexRef, unsigned int  maxAniso)
{
	TALLY_LOG("cuTexRefSetMaxAnisotropy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetBorderColor(CUtexref  hTexRef, float * pBorderColor)
{
	TALLY_LOG("cuTexRefSetBorderColor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefSetFlags(CUtexref  hTexRef, unsigned int  Flags)
{
	TALLY_LOG("cuTexRefSetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr * pdptr, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetAddress_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetArray(CUarray * phArray, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray * phMipmappedArray, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetAddressMode(CUaddress_mode * pam, CUtexref  hTexRef, int  dim)
{
	TALLY_LOG("cuTexRefGetAddressMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetFilterMode(CUfilter_mode * pfm, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetFilterMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetFormat(CUarray_format * pFormat, int * pNumChannels, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetFormat hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode * pfm, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetMipmapFilterMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetMipmapLevelBias(float * pbias, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetMipmapLevelBias hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetMipmapLevelClamp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetMaxAnisotropy(int * pmaxAniso, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetMaxAnisotropy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetBorderColor(float * pBorderColor, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetBorderColor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefGetFlags(unsigned int * pFlags, CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefGetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefCreate(CUtexref * pTexRef)
{
	TALLY_LOG("cuTexRefCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexRefDestroy(CUtexref  hTexRef)
{
	TALLY_LOG("cuTexRefDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuSurfRefSetArray(CUsurfref  hSurfRef, CUarray  hArray, unsigned int  Flags)
{
	TALLY_LOG("cuSurfRefSetArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuSurfRefGetArray(CUarray * phArray, CUsurfref  hSurfRef)
{
	TALLY_LOG("cuSurfRefGetArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexObjectCreate(CUtexObject * pTexObject, const CUDA_RESOURCE_DESC * pResDesc, const CUDA_TEXTURE_DESC * pTexDesc, const CUDA_RESOURCE_VIEW_DESC * pResViewDesc)
{
	TALLY_LOG("cuTexObjectCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexObjectDestroy(CUtexObject  texObject)
{
	TALLY_LOG("cuTexObjectDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUtexObject  texObject)
{
	TALLY_LOG("cuTexObjectGetResourceDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC * pTexDesc, CUtexObject  texObject)
{
	TALLY_LOG("cuTexObjectGetTextureDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject  texObject)
{
	TALLY_LOG("cuTexObjectGetResourceViewDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuSurfObjectCreate(CUsurfObject * pSurfObject, const CUDA_RESOURCE_DESC * pResDesc)
{
	TALLY_LOG("cuSurfObjectCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuSurfObjectDestroy(CUsurfObject  surfObject)
{
	TALLY_LOG("cuSurfObjectDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUsurfObject  surfObject)
{
	TALLY_LOG("cuSurfObjectGetResourceDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceCanAccessPeer(int * canAccessPeer, CUdevice  dev, CUdevice  peerDev)
{
	TALLY_LOG("cuDeviceCanAccessPeer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCtxEnablePeerAccess(CUcontext  peerContext, unsigned int  Flags)
{
	TALLY_LOG("cuCtxEnablePeerAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCtxDisablePeerAccess(CUcontext  peerContext)
{
	TALLY_LOG("cuCtxDisablePeerAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGetP2PAttribute(int*  value, CUdevice_P2PAttribute  attrib, CUdevice  srcDevice, CUdevice  dstDevice)
{
	TALLY_LOG("cuDeviceGetP2PAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource  resource)
{
	TALLY_LOG("cuGraphicsUnregisterResource hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray * pArray, CUgraphicsResource  resource, unsigned int  arrayIndex, unsigned int  mipLevel)
{
	TALLY_LOG("cuGraphicsSubResourceGetMappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray * pMipmappedArray, CUgraphicsResource  resource)
{
	TALLY_LOG("cuGraphicsResourceGetMappedMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource  resource)
{
	TALLY_LOG("cuGraphicsResourceGetMappedPointer_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource  resource, unsigned int  flags)
{
	TALLY_LOG("cuGraphicsResourceSetMapFlags_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsMapResources(unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)
{
	TALLY_LOG("cuGraphicsMapResources hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphicsUnmapResources(unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)
{
	TALLY_LOG("cuGraphicsUnmapResources hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGetProcAddress(const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags)
{
	TALLY_LOG("cuGetProcAddress hooked");
	CUresult res = 
		lcuGetProcAddress(symbol, pfn, cudaVersion, flags);
	return res;
}

CUresult cuGetExportTable(const void ** ppExportTable, const CUuuid * pExportTableId)
{
	TALLY_LOG("cuGetExportTable hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceReset()
{
	TALLY_LOG("cudaDeviceReset hooked");

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
	TALLY_LOG("cudaDeviceSynchronize hooked");

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
	TALLY_LOG("cudaDeviceSetLimit hooked");

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
	TALLY_LOG("cudaDeviceGetLimit hooked");

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
	if (pValue) { *pValue = res->pValue; }
	return res->err;
}

cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const struct cudaChannelFormatDesc * fmtDesc, int  device)
{
	TALLY_LOG("cudaDeviceGetTexture1DLinearMaxWidth hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
	TALLY_LOG("cudaDeviceGetCacheConfig hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETCACHECONFIG;
    
    struct cudaDeviceGetCacheConfigArg *arg_ptr = (struct cudaDeviceGetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pCacheConfig = pCacheConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetCacheConfigResponse *) dat;
	if (pCacheConfig) { *pCacheConfig = res->pCacheConfig; }
	return res->err;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	TALLY_LOG("cudaDeviceGetStreamPriorityRange hooked");

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
	if (leastPriority) { *leastPriority = res->leastPriority; }
	if (greatestPriority) { *greatestPriority = res->greatestPriority; }
	return res->err;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache  cacheConfig)
{
	TALLY_LOG("cudaDeviceSetCacheConfig hooked");

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
	TALLY_LOG("cudaDeviceGetSharedMemConfig hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig  config)
{
	TALLY_LOG("cudaDeviceSetSharedMemConfig hooked");

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
	TALLY_LOG("cudaDeviceGetByPCIBusId hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int  len, int  device)
{
	TALLY_LOG("cudaDeviceGetPCIBusId hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceGetPCIBusIdArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEGETPCIBUSID;
    
    struct cudaDeviceGetPCIBusIdArg *arg_ptr = (struct cudaDeviceGetPCIBusIdArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pciBusId = pciBusId;
	arg_ptr->len = len;
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaDeviceGetPCIBusIdResponse *) dat;
	if (pciBusId) { *pciBusId = res->pciBusId; }
	return res->err;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t  event)
{
	TALLY_LOG("cudaIpcGetEventHandle hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaIpcGetEventHandleArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAIPCGETEVENTHANDLE;
    
    struct cudaIpcGetEventHandleArg *arg_ptr = (struct cudaIpcGetEventHandleArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->event = event;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaIpcGetEventHandleResponse *) dat;
	if (handle) { *handle = res->handle; }
	return res->err;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t  handle)
{
	TALLY_LOG("cudaIpcOpenEventHandle hooked");

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
	if (event) { *event = res->event; }
	return res->err;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr)
{
	TALLY_LOG("cudaIpcGetMemHandle hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaIpcGetMemHandleArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAIPCGETMEMHANDLE;
    
    struct cudaIpcGetMemHandleArg *arg_ptr = (struct cudaIpcGetMemHandleArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->devPtr = devPtr;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaIpcGetMemHandleResponse *) dat;
	if (handle) { *handle = res->handle; }
	return res->err;
}

cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t  handle, unsigned int  flags)
{
	TALLY_LOG("cudaIpcOpenMemHandle hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaIpcOpenMemHandleArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAIPCOPENMEMHANDLE;
    
    struct cudaIpcOpenMemHandleArg *arg_ptr = (struct cudaIpcOpenMemHandleArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->devPtr = devPtr;
	arg_ptr->handle = handle;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaIpcOpenMemHandleResponse *) dat;
	if (devPtr) { *devPtr = res->devPtr; }
	return res->err;
}

cudaError_t cudaIpcCloseMemHandle(void * devPtr)
{
	TALLY_LOG("cudaIpcCloseMemHandle hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaIpcCloseMemHandleArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAIPCCLOSEMEMHANDLE;
    
    struct cudaIpcCloseMemHandleArg *arg_ptr = (struct cudaIpcCloseMemHandleArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->devPtr = devPtr;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget  target, enum cudaFlushGPUDirectRDMAWritesScope  scope)
{
	TALLY_LOG("cudaDeviceFlushGPUDirectRDMAWrites hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaDeviceFlushGPUDirectRDMAWritesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDADEVICEFLUSHGPUDIRECTRDMAWRITES;
    
    struct cudaDeviceFlushGPUDirectRDMAWritesArg *arg_ptr = (struct cudaDeviceFlushGPUDirectRDMAWritesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->target = target;
	arg_ptr->scope = scope;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaThreadExit()
{
	TALLY_LOG("cudaThreadExit hooked");

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
	TALLY_LOG("cudaThreadSynchronize hooked");

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
	TALLY_LOG("cudaThreadSetLimit hooked");

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
	TALLY_LOG("cudaThreadGetLimit hooked");

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
	if (pValue) { *pValue = res->pValue; }
	return res->err;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
	TALLY_LOG("cudaThreadGetCacheConfig hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaThreadGetCacheConfigArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDATHREADGETCACHECONFIG;
    
    struct cudaThreadGetCacheConfigArg *arg_ptr = (struct cudaThreadGetCacheConfigArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pCacheConfig = pCacheConfig;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaThreadGetCacheConfigResponse *) dat;
	if (pCacheConfig) { *pCacheConfig = res->pCacheConfig; }
	return res->err;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache  cacheConfig)
{
	TALLY_LOG("cudaThreadSetCacheConfig hooked");

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
	TALLY_LOG("cudaGetLastError hooked");

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
	TALLY_LOG("cudaPeekAtLastError hooked");

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
	TALLY_LOG("cudaGetErrorName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

const char* cudaGetErrorString(cudaError_t  error)
{
	TALLY_LOG("cudaGetErrorString hooked");
	const char* res = 
		lcudaGetErrorString(error);
	return res;
}

cudaError_t cudaGetDeviceCount(int * count)
{
	TALLY_LOG("cudaGetDeviceCount hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDeviceCountArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICECOUNT;
    
    struct cudaGetDeviceCountArg *arg_ptr = (struct cudaGetDeviceCountArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->count = count;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDeviceCountResponse *) dat;
	if (count) { *count = res->count; }
	return res->err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int  device)
{
	TALLY_LOG("cudaGetDeviceProperties hooked");

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
	if (prop) { *prop = res->prop; }
	return res->err;
}

cudaError_t cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr  attr, int  device)
{
	TALLY_LOG("cudaDeviceGetAttribute hooked");

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
	if (value) { *value = res->value; }
	return res->err;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int  device)
{
	TALLY_LOG("cudaDeviceGetDefaultMemPool hooked");

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
	if (memPool) { *memPool = res->memPool; }
	return res->err;
}

cudaError_t cudaDeviceSetMemPool(int  device, cudaMemPool_t  memPool)
{
	TALLY_LOG("cudaDeviceSetMemPool hooked");

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
	TALLY_LOG("cudaDeviceGetMemPool hooked");

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
	if (memPool) { *memPool = res->memPool; }
	return res->err;
}

cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int  device, int  flags)
{
	TALLY_LOG("cudaDeviceGetNvSciSyncAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetP2PAttribute(int * value, enum cudaDeviceP2PAttr  attr, int  srcDevice, int  dstDevice)
{
	TALLY_LOG("cudaDeviceGetP2PAttribute hooked");

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
	if (value) { *value = res->value; }
	return res->err;
}

cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop)
{
	TALLY_LOG("cudaChooseDevice hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaSetDevice(int  device)
{
	TALLY_LOG("cudaSetDevice hooked");

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
	TALLY_LOG("cudaGetDevice hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDeviceArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICE;
    
    struct cudaGetDeviceArg *arg_ptr = (struct cudaGetDeviceArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->device = device;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDeviceResponse *) dat;
	if (device) { *device = res->device; }
	return res->err;
}

cudaError_t cudaSetValidDevices(int * device_arr, int  len)
{
	TALLY_LOG("cudaSetValidDevices hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags)
{
	TALLY_LOG("cudaSetDeviceFlags hooked");

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
	TALLY_LOG("cudaGetDeviceFlags hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaGetDeviceFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAGETDEVICEFLAGS;
    
    struct cudaGetDeviceFlagsArg *arg_ptr = (struct cudaGetDeviceFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaGetDeviceFlagsResponse *) dat;
	if (flags) { *flags = res->flags; }
	return res->err;
}

cudaError_t cudaStreamCreate(cudaStream_t * pStream)
{
	TALLY_LOG("cudaStreamCreate hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMCREATE;
    
    struct cudaStreamCreateArg *arg_ptr = (struct cudaStreamCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pStream = pStream;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamCreateResponse *) dat;
	if (pStream) { *pStream = res->pStream; }
	return res->err;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int  flags)
{
	TALLY_LOG("cudaStreamCreateWithFlags hooked");

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
	if (pStream) { *pStream = res->pStream; }
	return res->err;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned int  flags, int  priority)
{
	TALLY_LOG("cudaStreamCreateWithPriority hooked");

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
	if (pStream) { *pStream = res->pStream; }
	return res->err;
}

cudaError_t cudaStreamGetPriority(cudaStream_t  hStream, int * priority)
{
	TALLY_LOG("cudaStreamGetPriority hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamGetPriorityArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMGETPRIORITY;
    
    struct cudaStreamGetPriorityArg *arg_ptr = (struct cudaStreamGetPriorityArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->hStream = hStream;
	arg_ptr->priority = priority;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamGetPriorityResponse *) dat;
	if (priority) { *priority = res->priority; }
	return res->err;
}

cudaError_t cudaStreamGetFlags(cudaStream_t  hStream, unsigned int * flags)
{
	TALLY_LOG("cudaStreamGetFlags hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaStreamGetFlagsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDASTREAMGETFLAGS;
    
    struct cudaStreamGetFlagsArg *arg_ptr = (struct cudaStreamGetFlagsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->hStream = hStream;
	arg_ptr->flags = flags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaStreamGetFlagsResponse *) dat;
	if (flags) { *flags = res->flags; }
	return res->err;
}

cudaError_t cudaCtxResetPersistingL2Cache()
{
	TALLY_LOG("cudaCtxResetPersistingL2Cache hooked");

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
	TALLY_LOG("cudaStreamCopyAttributes hooked");

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
	TALLY_LOG("cudaStreamGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamSetAttribute(cudaStream_t  hStream, enum cudaStreamAttrID  attr, const union cudaStreamAttrValue * value)
{
	TALLY_LOG("cudaStreamSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamDestroy(cudaStream_t  stream)
{
	TALLY_LOG("cudaStreamDestroy hooked");

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
	TALLY_LOG("cudaStreamWaitEvent hooked");

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
	TALLY_LOG("cudaStreamAddCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamSynchronize(cudaStream_t  stream)
{
	TALLY_LOG("cudaStreamSynchronize hooked");

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
	TALLY_LOG("cudaStreamQuery hooked");

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
	TALLY_LOG("cudaStreamAttachMemAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamBeginCapture(cudaStream_t  stream, enum cudaStreamCaptureMode  mode)
{
	TALLY_LOG("cudaStreamBeginCapture hooked");

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
	TALLY_LOG("cudaThreadExchangeStreamCaptureMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamEndCapture(cudaStream_t  stream, cudaGraph_t * pGraph)
{
	TALLY_LOG("cudaStreamEndCapture hooked");

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
	TALLY_LOG("cudaStreamIsCapturing hooked");

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
	if (pCaptureStatus) { *pCaptureStatus = res->pCaptureStatus; }
	return res->err;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus, unsigned long long * pId)
{
	TALLY_LOG("cudaStreamGetCaptureInfo hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t  stream, enum cudaStreamCaptureStatus * captureStatus_out, unsigned long long * id_out, cudaGraph_t * graph_out, const cudaGraphNode_t ** dependencies_out, size_t * numDependencies_out)
{
	TALLY_LOG("cudaStreamGetCaptureInfo_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t  stream, cudaGraphNode_t * dependencies, size_t  numDependencies, unsigned int  flags)
{
	TALLY_LOG("cudaStreamUpdateCaptureDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaEventCreate(cudaEvent_t * event)
{
	TALLY_LOG("cudaEventCreate hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaEventCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAEVENTCREATE;
    
    struct cudaEventCreateArg *arg_ptr = (struct cudaEventCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->event = event;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaEventCreateResponse *) dat;
	if (event) { *event = res->event; }
	return res->err;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int  flags)
{
	TALLY_LOG("cudaEventCreateWithFlags hooked");

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
	if (event) { *event = res->event; }
	return res->err;
}

cudaError_t cudaEventRecord(cudaEvent_t  event, cudaStream_t  stream)
{
	TALLY_LOG("cudaEventRecord hooked");

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
	TALLY_LOG("cudaEventRecordWithFlags hooked");

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
	TALLY_LOG("cudaEventQuery hooked");

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
	TALLY_LOG("cudaEventSynchronize hooked");

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
	TALLY_LOG("cudaEventDestroy hooked");

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
	TALLY_LOG("cudaEventElapsedTime hooked");

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
	if (ms) { *ms = res->ms; }
	return res->err;
}

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const struct cudaExternalMemoryHandleDesc * memHandleDesc)
{
	TALLY_LOG("cudaImportExternalMemory hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryBufferDesc * bufferDesc)
{
	TALLY_LOG("cudaExternalMemoryGetMappedBuffer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryMipmappedArrayDesc * mipmapDesc)
{
	TALLY_LOG("cudaExternalMemoryGetMappedMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t  extMem)
{
	TALLY_LOG("cudaDestroyExternalMemory hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const struct cudaExternalSemaphoreHandleDesc * semHandleDesc)
{
	TALLY_LOG("cudaImportExternalSemaphore hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreSignalParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)
{
	TALLY_LOG("cudaSignalExternalSemaphoresAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreWaitParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)
{
	TALLY_LOG("cudaWaitExternalSemaphoresAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t  extSem)
{
	TALLY_LOG("cudaDestroyExternalSemaphore hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
	TALLY_LOG("cudaLaunchCooperativeKernel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams * launchParamsList, unsigned int  numDevices, unsigned int  flags)
{
	TALLY_LOG("cudaLaunchCooperativeKernelMultiDevice hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache  cacheConfig)
{
	TALLY_LOG("cudaFuncSetCacheConfig hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig  config)
{
	TALLY_LOG("cudaFuncSetSharedMemConfig hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr, const void * func)
{
	TALLY_LOG("cudaFuncGetAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFuncSetAttribute(const void * func, enum cudaFuncAttribute  attr, int  value)
{
	TALLY_LOG("cudaFuncSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaSetDoubleForDevice(double * d)
{
	TALLY_LOG("cudaSetDoubleForDevice hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaSetDoubleForHost(double * d)
{
	TALLY_LOG("cudaSetDoubleForHost hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaLaunchHostFunc(cudaStream_t  stream, cudaHostFn_t  fn, void * userData)
{
	TALLY_LOG("cudaLaunchHostFunc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize)
{
	TALLY_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int  numBlocks, int  blockSize)
{
	TALLY_LOG("cudaOccupancyAvailableDynamicSMemPerBlock hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	TALLY_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMallocManaged(void ** devPtr, size_t  size, unsigned int  flags)
{
	TALLY_LOG("cudaMallocManaged hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMallocHost(void ** ptr, size_t  size)
{
	TALLY_LOG("cudaMallocHost hooked");
	cudaError_t res = 
		lcudaMallocHost(ptr, size);
	return res;
}

cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t  width, size_t  height)
{
	TALLY_LOG("cudaMallocPitch hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, unsigned int  flags)
{
	TALLY_LOG("cudaMallocArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFreeHost(void * ptr)
{
	TALLY_LOG("cudaFreeHost hooked");
	cudaError_t res = 
		lcudaFreeHost(ptr);
	return res;
}

cudaError_t cudaFreeArray(cudaArray_t  array)
{
	TALLY_LOG("cudaFreeArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t  mipmappedArray)
{
	TALLY_LOG("cudaFreeMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaHostAlloc(void ** pHost, size_t  size, unsigned int  flags)
{
	TALLY_LOG("cudaHostAlloc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaHostRegister(void * ptr, size_t  size, unsigned int  flags)
{
	TALLY_LOG("cudaHostRegister hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaHostUnregister(void * ptr)
{
	TALLY_LOG("cudaHostUnregister hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int  flags)
{
	TALLY_LOG("cudaHostGetDevicePointer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost)
{
	TALLY_LOG("cudaHostGetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr*  pitchedDevPtr, struct cudaExtent  extent)
{
	TALLY_LOG("cudaMalloc3D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  flags)
{
	TALLY_LOG("cudaMalloc3DArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  numLevels, unsigned int  flags)
{
	TALLY_LOG("cudaMallocMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t  mipmappedArray, unsigned int  level)
{
	TALLY_LOG("cudaGetMipmappedArrayLevel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * p)
{
	TALLY_LOG("cudaMemcpy3D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p)
{
	TALLY_LOG("cudaMemcpy3DPeer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpy3DAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpy3DPeerAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemGetInfo(size_t * free, size_t * total)
{
	TALLY_LOG("cudaMemGetInfo hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaMemGetInfoArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMEMGETINFO;
    
    struct cudaMemGetInfoArg *arg_ptr = (struct cudaMemGetInfoArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->free = free;
	arg_ptr->total = total;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudaMemGetInfoResponse *) dat;
	if (free) { *free = res->free; }
	if (total) { *total = res->total; }
	return res->err;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t  array)
{
	TALLY_LOG("cudaArrayGetInfo hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t  hArray, unsigned int  planeIdx)
{
	TALLY_LOG("cudaArrayGetPlane hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties * sparseProperties, cudaArray_t  array)
{
	TALLY_LOG("cudaArrayGetSparseProperties hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t  mipmap)
{
	TALLY_LOG("cudaMipmappedArrayGetSparseProperties hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyPeer(void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count)
{
	TALLY_LOG("cudaMemcpyPeer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2D(void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpy2D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpy2DToArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2DFromArray(void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpy2DFromArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpy2DArrayToArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpyToSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpyFromSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyPeerAsync(void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpyPeerAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2DAsync(void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpy2DAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpy2DToArrayAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpy2DFromArrayAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpyToSymbolAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpyFromSymbolAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemset(void * devPtr, int  value, size_t  count)
{
	TALLY_LOG("cudaMemset hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudaMemsetArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDAMEMSET;
    
    struct cudaMemsetArg *arg_ptr = (struct cudaMemsetArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->devPtr = devPtr;
	arg_ptr->value = value;
	arg_ptr->count = count;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudaError_t *) dat;
    return *res;
}

cudaError_t cudaMemset2D(void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height)
{
	TALLY_LOG("cudaMemset2D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent)
{
	TALLY_LOG("cudaMemset3D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemsetAsync(void * devPtr, int  value, size_t  count, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemsetAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemset2DAsync(void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemset2DAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemset3DAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol)
{
	TALLY_LOG("cudaGetSymbolAddress hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol)
{
	TALLY_LOG("cudaGetSymbolSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t  count, int  dstDevice, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemPrefetchAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemAdvise(const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, int  device)
{
	TALLY_LOG("cudaMemAdvise hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemRangeGetAttribute(void * data, size_t  dataSize, enum cudaMemRangeAttribute  attribute, const void * devPtr, size_t  count)
{
	TALLY_LOG("cudaMemRangeGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, enum cudaMemRangeAttribute * attributes, size_t  numAttributes, const void * devPtr, size_t  count)
{
	TALLY_LOG("cudaMemRangeGetAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyToArray(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpyToArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpyFromArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaMemcpyArrayToArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpyToArrayAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemcpyFromArrayAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMallocAsync(void ** devPtr, size_t  size, cudaStream_t  hStream)
{
	TALLY_LOG("cudaMallocAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t  hStream)
{
	TALLY_LOG("cudaFreeAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t  memPool, size_t  minBytesToKeep)
{
	TALLY_LOG("cudaMemPoolTrimTo hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)
{
	TALLY_LOG("cudaMemPoolSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)
{
	TALLY_LOG("cudaMemPoolGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t  memPool, const struct cudaMemAccessDesc * descList, size_t  count)
{
	TALLY_LOG("cudaMemPoolSetAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags * flags, cudaMemPool_t  memPool, struct cudaMemLocation * location)
{
	TALLY_LOG("cudaMemPoolGetAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const struct cudaMemPoolProps * poolProps)
{
	TALLY_LOG("cudaMemPoolCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t  memPool)
{
	TALLY_LOG("cudaMemPoolDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t  size, cudaMemPool_t  memPool, cudaStream_t  stream)
{
	TALLY_LOG("cudaMallocFromPoolAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t  memPool, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)
{
	TALLY_LOG("cudaMemPoolExportToShareableHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)
{
	TALLY_LOG("cudaMemPoolImportFromShareableHandle hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData * exportData, void * ptr)
{
	TALLY_LOG("cudaMemPoolExportPointer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t  memPool, struct cudaMemPoolPtrExportData * exportData)
{
	TALLY_LOG("cudaMemPoolImportPointer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes * attributes, const void * ptr)
{
	TALLY_LOG("cudaPointerGetAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int  device, int  peerDevice)
{
	TALLY_LOG("cudaDeviceCanAccessPeer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceEnablePeerAccess(int  peerDevice, unsigned int  flags)
{
	TALLY_LOG("cudaDeviceEnablePeerAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceDisablePeerAccess(int  peerDevice)
{
	TALLY_LOG("cudaDeviceDisablePeerAccess hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t  resource)
{
	TALLY_LOG("cudaGraphicsUnregisterResource hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t  resource, unsigned int  flags)
{
	TALLY_LOG("cudaGraphicsResourceSetMapFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsMapResources(int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)
{
	TALLY_LOG("cudaGraphicsMapResources hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsUnmapResources(int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)
{
	TALLY_LOG("cudaGraphicsUnmapResources hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t  resource)
{
	TALLY_LOG("cudaGraphicsResourceGetMappedPointer hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t  resource, unsigned int  arrayIndex, unsigned int  mipLevel)
{
	TALLY_LOG("cudaGraphicsSubResourceGetMappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t  resource)
{
	TALLY_LOG("cudaGraphicsResourceGetMappedMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaBindTexture(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t  size)
{
	TALLY_LOG("cudaBindTexture hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaBindTexture2D(size_t * offset, const struct textureReference * texref, const void * devPtr, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, size_t  pitch)
{
	TALLY_LOG("cudaBindTexture2D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaBindTextureToArray(const struct textureReference * texref, cudaArray_const_t  array, const struct cudaChannelFormatDesc * desc)
{
	TALLY_LOG("cudaBindTextureToArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference * texref, cudaMipmappedArray_const_t  mipmappedArray, const struct cudaChannelFormatDesc * desc)
{
	TALLY_LOG("cudaBindTextureToMipmappedArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaUnbindTexture(const struct textureReference * texref)
{
	TALLY_LOG("cudaUnbindTexture hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const struct textureReference * texref)
{
	TALLY_LOG("cudaGetTextureAlignmentOffset hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetTextureReference(const struct textureReference ** texref, const void * symbol)
{
	TALLY_LOG("cudaGetTextureReference hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaBindSurfaceToArray(const struct surfaceReference * surfref, cudaArray_const_t  array, const struct cudaChannelFormatDesc * desc)
{
	TALLY_LOG("cudaBindSurfaceToArray hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetSurfaceReference(const struct surfaceReference ** surfref, const void * symbol)
{
	TALLY_LOG("cudaGetSurfaceReference hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t  array)
{
	TALLY_LOG("cudaGetChannelDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int  x, int  y, int  z, int  w, enum cudaChannelFormatKind  f)
{
	TALLY_LOG("cudaCreateChannelDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc)
{
	TALLY_LOG("cudaCreateTextureObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t  texObject)
{
	TALLY_LOG("cudaDestroyTextureObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t  texObject)
{
	TALLY_LOG("cudaGetTextureObjectResourceDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t  texObject)
{
	TALLY_LOG("cudaGetTextureObjectTextureDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t  texObject)
{
	TALLY_LOG("cudaGetTextureObjectResourceViewDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc)
{
	TALLY_LOG("cudaCreateSurfaceObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t  surfObject)
{
	TALLY_LOG("cudaDestroySurfaceObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t  surfObject)
{
	TALLY_LOG("cudaGetSurfaceObjectResourceDesc hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDriverGetVersion(int * driverVersion)
{
	TALLY_LOG("cudaDriverGetVersion hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)
{
	TALLY_LOG("cudaRuntimeGetVersion hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned int  flags)
{
	TALLY_LOG("cudaGraphCreate hooked");

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
	if (pGraph) { *pGraph = res->pGraph; }
	return res->err;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphAddKernelNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t  node, struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphKernelNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphKernelNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t  hSrc, cudaGraphNode_t  hDst)
{
	TALLY_LOG("cudaGraphKernelNodeCopyAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t  hNode, enum cudaKernelNodeAttrID  attr, union cudaKernelNodeAttrValue * value_out)
{
	TALLY_LOG("cudaGraphKernelNodeGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t  hNode, enum cudaKernelNodeAttrID  attr, const union cudaKernelNodeAttrValue * value)
{
	TALLY_LOG("cudaGraphKernelNodeSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemcpy3DParms * pCopyParams)
{
	TALLY_LOG("cudaGraphAddMemcpyNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphAddMemcpyNodeToSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t*  pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t*  pDependencies, size_t  numDependencies, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphAddMemcpyNodeFromSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphAddMemcpyNode1D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t  node, struct cudaMemcpy3DParms * pNodeParams)
{
	TALLY_LOG("cudaGraphMemcpyNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)
{
	TALLY_LOG("cudaGraphMemcpyNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphMemcpyNodeSetParamsToSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphMemcpyNodeSetParamsFromSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphMemcpyNodeSetParams1D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemsetParams * pMemsetParams)
{
	TALLY_LOG("cudaGraphAddMemsetNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t  node, struct cudaMemsetParams * pNodeParams)
{
	TALLY_LOG("cudaGraphMemsetNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)
{
	TALLY_LOG("cudaGraphMemsetNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaHostNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphAddHostNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t  node, struct cudaHostNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphHostNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphHostNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaGraph_t  childGraph)
{
	TALLY_LOG("cudaGraphAddChildGraphNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t  node, cudaGraph_t * pGraph)
{
	TALLY_LOG("cudaGraphChildGraphNodeGetGraph hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies)
{
	TALLY_LOG("cudaGraphAddEmptyNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)
{
	TALLY_LOG("cudaGraphAddEventRecordNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t  node, cudaEvent_t * event_out)
{
	TALLY_LOG("cudaGraphEventRecordNodeGetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t  node, cudaEvent_t  event)
{
	TALLY_LOG("cudaGraphEventRecordNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)
{
	TALLY_LOG("cudaGraphAddEventWaitNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t  node, cudaEvent_t * event_out)
{
	TALLY_LOG("cudaGraphEventWaitNodeGetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t  node, cudaEvent_t  event)
{
	TALLY_LOG("cudaGraphEventWaitNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphAddExternalSemaphoresSignalNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t  hNode, struct cudaExternalSemaphoreSignalNodeParams * params_out)
{
	TALLY_LOG("cudaGraphExternalSemaphoresSignalNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphExternalSemaphoresSignalNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphAddExternalSemaphoresWaitNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t  hNode, struct cudaExternalSemaphoreWaitNodeParams * params_out)
{
	TALLY_LOG("cudaGraphExternalSemaphoresWaitNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphExternalSemaphoresWaitNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaMemAllocNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphAddMemAllocNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t  node, struct cudaMemAllocNodeParams * params_out)
{
	TALLY_LOG("cudaGraphMemAllocNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void * dptr)
{
	TALLY_LOG("cudaGraphAddMemFreeNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t  node, void * dptr_out)
{
	TALLY_LOG("cudaGraphMemFreeNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGraphMemTrim(int  device)
{
	TALLY_LOG("cudaDeviceGraphMemTrim hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetGraphMemAttribute(int  device, enum cudaGraphMemAttributeType  attr, void*  value)
{
	TALLY_LOG("cudaDeviceGetGraphMemAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceSetGraphMemAttribute(int  device, enum cudaGraphMemAttributeType  attr, void*  value)
{
	TALLY_LOG("cudaDeviceSetGraphMemAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t  originalGraph)
{
	TALLY_LOG("cudaGraphClone hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t  originalNode, cudaGraph_t  clonedGraph)
{
	TALLY_LOG("cudaGraphNodeFindInClone hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t  node, enum cudaGraphNodeType * pType)
{
	TALLY_LOG("cudaGraphNodeGetType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphGetNodes(cudaGraph_t  graph, cudaGraphNode_t * nodes, size_t * numNodes)
{
	TALLY_LOG("cudaGraphGetNodes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t  graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes)
{
	TALLY_LOG("cudaGraphGetRootNodes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphGetEdges(cudaGraph_t  graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges)
{
	TALLY_LOG("cudaGraphGetEdges hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t  node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies)
{
	TALLY_LOG("cudaGraphNodeGetDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t  node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)
{
	TALLY_LOG("cudaGraphNodeGetDependentNodes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)
{
	TALLY_LOG("cudaGraphAddDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)
{
	TALLY_LOG("cudaGraphRemoveDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t  node)
{
	TALLY_LOG("cudaGraphDestroyNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t  bufferSize)
{
	TALLY_LOG("cudaGraphInstantiate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)
{
	TALLY_LOG("cudaGraphInstantiateWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphExecKernelNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)
{
	TALLY_LOG("cudaGraphExecMemcpyNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphExecMemcpyNodeSetParamsToSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphExecMemcpyNodeSetParamsFromSymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_LOG("cudaGraphExecMemcpyNodeSetParams1D hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)
{
	TALLY_LOG("cudaGraphExecMemsetNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)
{
	TALLY_LOG("cudaGraphExecHostNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, cudaGraph_t  childGraph)
{
	TALLY_LOG("cudaGraphExecChildGraphNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)
{
	TALLY_LOG("cudaGraphExecEventRecordNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)
{
	TALLY_LOG("cudaGraphExecEventWaitNodeSetEvent hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphExecExternalSemaphoresSignalNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphExecExternalSemaphoresWaitNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t  hGraphExec, cudaGraph_t  hGraph, cudaGraphNode_t * hErrorNode_out, enum cudaGraphExecUpdateResult * updateResult_out)
{
	TALLY_LOG("cudaGraphExecUpdate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphUpload(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	TALLY_LOG("cudaGraphUpload hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	TALLY_LOG("cudaGraphLaunch hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t  graphExec)
{
	TALLY_LOG("cudaGraphExecDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphDestroy(cudaGraph_t  graph)
{
	TALLY_LOG("cudaGraphDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t  graph, const char * path, unsigned int  flags)
{
	TALLY_LOG("cudaGraphDebugDotPrint hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t  destroy, unsigned int  initialRefcount, unsigned int  flags)
{
	TALLY_LOG("cudaUserObjectCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t  object, unsigned int  count)
{
	TALLY_LOG("cudaUserObjectRetain hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t  object, unsigned int  count)
{
	TALLY_LOG("cudaUserObjectRelease hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count, unsigned int  flags)
{
	TALLY_LOG("cudaGraphRetainUserObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count)
{
	TALLY_LOG("cudaGraphReleaseUserObject hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long  flags)
{
	TALLY_LOG("cudaGetDriverEntryPoint hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId)
{
	TALLY_LOG("cudaGetExportTable hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t*  functionPtr, const void*  symbolPtr)
{
	TALLY_LOG("cudaGetFuncBySymbol hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

size_t cudnnGetVersion()
{
	TALLY_LOG("cudnnGetVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETVERSION;
    
    struct cudnnGetVersionArg *arg_ptr = (struct cudnnGetVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (size_t *) dat;
    return *res;
}

size_t cudnnGetMaxDeviceVersion()
{
	TALLY_LOG("cudnnGetMaxDeviceVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetMaxDeviceVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETMAXDEVICEVERSION;
    
    struct cudnnGetMaxDeviceVersionArg *arg_ptr = (struct cudnnGetMaxDeviceVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (size_t *) dat;
    return *res;
}

size_t cudnnGetCudartVersion()
{
	TALLY_LOG("cudnnGetCudartVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetCudartVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETCUDARTVERSION;
    
    struct cudnnGetCudartVersionArg *arg_ptr = (struct cudnnGetCudartVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (size_t *) dat;
    return *res;
}

const char * cudnnGetErrorString(cudnnStatus_t  status)
{
	TALLY_LOG("cudnnGetErrorString hooked");
	const char * res = 
		lcudnnGetErrorString(status);
	return res;
}

cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t  handle, cudnnStatus_t * rstatus, cudnnErrQueryMode_t  mode, cudnnRuntimeTag_t * tag)
{
	TALLY_LOG("cudnnQueryRuntimeError hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType  type, int * value)
{
	TALLY_LOG("cudnnGetProperty hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetPropertyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETPROPERTY;
    
    struct cudnnGetPropertyArg *arg_ptr = (struct cudnnGetPropertyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->type = type;
	arg_ptr->value = value;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetPropertyResponse *) dat;
	if (value) { *value = res->value; }
	return res->err;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t * handle)
{
	TALLY_LOG("cudnnCreate hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATE;
    
    struct cudnnCreateArg *arg_ptr = (struct cudnnCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateResponse *) dat;
	if (handle) { *handle = res->handle; }
	return res->err;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t  handle)
{
	TALLY_LOG("cudnnDestroy hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROY;
    
    struct cudnnDestroyArg *arg_ptr = (struct cudnnDestroyArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t  handle, cudaStream_t  streamId)
{
	TALLY_LOG("cudnnSetStream hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetStreamArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETSTREAM;
    
    struct cudnnSetStreamArg *arg_ptr = (struct cudnnSetStreamArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->streamId = streamId;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t  handle, cudaStream_t * streamId)
{
	TALLY_LOG("cudnnGetStream hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetStreamArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETSTREAM;
    
    struct cudnnGetStreamArg *arg_ptr = (struct cudnnGetStreamArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->streamId = streamId;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetStreamResponse *) dat;
	if (streamId) { *streamId = res->streamId; }
	return res->err;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t * tensorDesc)
{
	TALLY_LOG("cudnnCreateTensorDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateTensorDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATETENSORDESCRIPTOR;
    
    struct cudnnCreateTensorDescriptorArg *arg_ptr = (struct cudnnCreateTensorDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->tensorDesc = tensorDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateTensorDescriptorResponse *) dat;
	if (tensorDesc) { *tensorDesc = res->tensorDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w)
{
	TALLY_LOG("cudnnSetTensor4dDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetTensor4dDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOR;
    
    struct cudnnSetTensor4dDescriptorArg *arg_ptr = (struct cudnnSetTensor4dDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->tensorDesc = tensorDesc;
	arg_ptr->format = format;
	arg_ptr->dataType = dataType;
	arg_ptr->n = n;
	arg_ptr->c = c;
	arg_ptr->h = h;
	arg_ptr->w = w;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w, int  nStride, int  cStride, int  hStride, int  wStride)
{
	TALLY_LOG("cudnnSetTensor4dDescriptorEx hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetTensor4dDescriptorExArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOREX;
    
    struct cudnnSetTensor4dDescriptorExArg *arg_ptr = (struct cudnnSetTensor4dDescriptorExArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->tensorDesc = tensorDesc;
	arg_ptr->dataType = dataType;
	arg_ptr->n = n;
	arg_ptr->c = c;
	arg_ptr->h = h;
	arg_ptr->w = w;
	arg_ptr->nStride = nStride;
	arg_ptr->cStride = cStride;
	arg_ptr->hStride = hStride;
	arg_ptr->wStride = wStride;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t * dataType, int * n, int * c, int * h, int * w, int * nStride, int * cStride, int * hStride, int * wStride)
{
	TALLY_LOG("cudnnGetTensor4dDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetTensor4dDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETTENSOR4DDESCRIPTOR;
    
    struct cudnnGetTensor4dDescriptorArg *arg_ptr = (struct cudnnGetTensor4dDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->tensorDesc = tensorDesc;
	arg_ptr->dataType = dataType;
	arg_ptr->n = n;
	arg_ptr->c = c;
	arg_ptr->h = h;
	arg_ptr->w = w;
	arg_ptr->nStride = nStride;
	arg_ptr->cStride = cStride;
	arg_ptr->hStride = hStride;
	arg_ptr->wStride = wStride;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetTensor4dDescriptorResponse *) dat;
	if (dataType) { *dataType = res->dataType; }
	if (n) { *n = res->n; }
	if (c) { *c = res->c; }
	if (h) { *h = res->h; }
	if (w) { *w = res->w; }
	if (nStride) { *nStride = res->nStride; }
	if (cStride) { *cStride = res->cStride; }
	if (hStride) { *hStride = res->hStride; }
	if (wStride) { *wStride = res->wStride; }
	return res->err;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  nbDims, const int  dimA[])
{
	TALLY_LOG("cudnnSetTensorNdDescriptorEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t  tensorDesc, size_t * size)
{
	TALLY_LOG("cudnnGetTensorSizeInBytes hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetTensorSizeInBytesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETTENSORSIZEINBYTES;
    
    struct cudnnGetTensorSizeInBytesArg *arg_ptr = (struct cudnnGetTensorSizeInBytesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->tensorDesc = tensorDesc;
	arg_ptr->size = size;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetTensorSizeInBytesResponse *) dat;
	if (size) { *size = res->size; }
	return res->err;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t  tensorDesc)
{
	TALLY_LOG("cudnnDestroyTensorDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyTensorDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYTENSORDESCRIPTOR;
    
    struct cudnnDestroyTensorDescriptorArg *arg_ptr = (struct cudnnDestroyTensorDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->tensorDesc = tensorDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t  transformDesc, const cudnnTensorDescriptor_t  srcDesc, cudnnTensorDescriptor_t  destDesc, size_t * destSizeInBytes)
{
	TALLY_LOG("cudnnInitTransformDest hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnInitTransformDestArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNINITTRANSFORMDEST;
    
    struct cudnnInitTransformDestArg *arg_ptr = (struct cudnnInitTransformDestArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->transformDesc = transformDesc;
	arg_ptr->srcDesc = srcDesc;
	arg_ptr->destDesc = destDesc;
	arg_ptr->destSizeInBytes = destSizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnInitTransformDestResponse *) dat;
	if (destSizeInBytes) { *destSizeInBytes = res->destSizeInBytes; }
	return res->err;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t * transformDesc)
{
	TALLY_LOG("cudnnCreateTensorTransformDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateTensorTransformDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATETENSORTRANSFORMDESCRIPTOR;
    
    struct cudnnCreateTensorTransformDescriptorArg *arg_ptr = (struct cudnnCreateTensorTransformDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->transformDesc = transformDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateTensorTransformDescriptorResponse *) dat;
	if (transformDesc) { *transformDesc = res->transformDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc, const uint32_t  nbDims, const cudnnTensorFormat_t  destFormat, const int32_t  padBeforeA[], const int32_t  padAfterA[], const uint32_t  foldA[], const cudnnFoldingDirection_t  direction)
{
	TALLY_LOG("cudnnSetTensorTransformDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc, uint32_t  nbDimsRequested, cudnnTensorFormat_t * destFormat, int32_t  padBeforeA[], int32_t  padAfterA[], uint32_t  foldA[], cudnnFoldingDirection_t * direction)
{
	TALLY_LOG("cudnnGetTensorTransformDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc)
{
	TALLY_LOG("cudnnDestroyTensorTransformDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyTensorTransformDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYTENSORTRANSFORMDESCRIPTOR;
    
    struct cudnnDestroyTensorTransformDescriptorArg *arg_ptr = (struct cudnnDestroyTensorTransformDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->transformDesc = transformDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnTransformTensorEx(cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnTensorDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnTensorDescriptor_t  destDesc, void * destData)
{
	TALLY_LOG("cudnnTransformTensorEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t * opTensorDesc)
{
	TALLY_LOG("cudnnCreateOpTensorDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateOpTensorDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATEOPTENSORDESCRIPTOR;
    
    struct cudnnCreateOpTensorDescriptorArg *arg_ptr = (struct cudnnCreateOpTensorDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->opTensorDesc = opTensorDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateOpTensorDescriptorResponse *) dat;
	if (opTensorDesc) { *opTensorDesc = res->opTensorDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t  opTensorOp, cudnnDataType_t  opTensorCompType, cudnnNanPropagation_t  opTensorNanOpt)
{
	TALLY_LOG("cudnnSetOpTensorDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetOpTensorDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETOPTENSORDESCRIPTOR;
    
    struct cudnnSetOpTensorDescriptorArg *arg_ptr = (struct cudnnSetOpTensorDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->opTensorDesc = opTensorDesc;
	arg_ptr->opTensorOp = opTensorOp;
	arg_ptr->opTensorCompType = opTensorCompType;
	arg_ptr->opTensorNanOpt = opTensorNanOpt;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t * opTensorOp, cudnnDataType_t * opTensorCompType, cudnnNanPropagation_t * opTensorNanOpt)
{
	TALLY_LOG("cudnnGetOpTensorDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetOpTensorDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETOPTENSORDESCRIPTOR;
    
    struct cudnnGetOpTensorDescriptorArg *arg_ptr = (struct cudnnGetOpTensorDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->opTensorDesc = opTensorDesc;
	arg_ptr->opTensorOp = opTensorOp;
	arg_ptr->opTensorCompType = opTensorCompType;
	arg_ptr->opTensorNanOpt = opTensorNanOpt;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetOpTensorDescriptorResponse *) dat;
	if (opTensorOp) { *opTensorOp = res->opTensorOp; }
	if (opTensorCompType) { *opTensorCompType = res->opTensorCompType; }
	if (opTensorNanOpt) { *opTensorNanOpt = res->opTensorNanOpt; }
	return res->err;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t  opTensorDesc)
{
	TALLY_LOG("cudnnDestroyOpTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnOpTensor(cudnnHandle_t  handle, const cudnnOpTensorDescriptor_t  opTensorDesc, const void * alpha1, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * alpha2, const cudnnTensorDescriptor_t  bDesc, const void * B, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	TALLY_LOG("cudnnOpTensor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t * reduceTensorDesc)
{
	TALLY_LOG("cudnnCreateReduceTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t  reduceTensorOp, cudnnDataType_t  reduceTensorCompType, cudnnNanPropagation_t  reduceTensorNanOpt, cudnnReduceTensorIndices_t  reduceTensorIndices, cudnnIndicesType_t  reduceTensorIndicesType)
{
	TALLY_LOG("cudnnSetReduceTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t * reduceTensorOp, cudnnDataType_t * reduceTensorCompType, cudnnNanPropagation_t * reduceTensorNanOpt, cudnnReduceTensorIndices_t * reduceTensorIndices, cudnnIndicesType_t * reduceTensorIndicesType)
{
	TALLY_LOG("cudnnGetReduceTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t  reduceTensorDesc)
{
	TALLY_LOG("cudnnDestroyReduceTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetReductionIndicesSize(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetReductionIndicesSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetReductionWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnReduceTensor(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, void * indices, size_t  indicesSizeInBytes, void * workspace, size_t  workspaceSizeInBytes, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	TALLY_LOG("cudnnReduceTensor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetTensor(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * valuePtr)
{
	TALLY_LOG("cudnnSetTensor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * alpha)
{
	TALLY_LOG("cudnnScaleTensor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t * filterDesc)
{
	TALLY_LOG("cudnnCreateFilterDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateFilterDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATEFILTERDESCRIPTOR;
    
    struct cudnnCreateFilterDescriptorArg *arg_ptr = (struct cudnnCreateFilterDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->filterDesc = filterDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateFilterDescriptorResponse *) dat;
	if (filterDesc) { *filterDesc = res->filterDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  k, int  c, int  h, int  w)
{
	TALLY_LOG("cudnnSetFilter4dDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetFilter4dDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETFILTER4DDESCRIPTOR;
    
    struct cudnnSetFilter4dDescriptorArg *arg_ptr = (struct cudnnSetFilter4dDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->filterDesc = filterDesc;
	arg_ptr->dataType = dataType;
	arg_ptr->format = format;
	arg_ptr->k = k;
	arg_ptr->c = c;
	arg_ptr->h = h;
	arg_ptr->w = w;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * k, int * c, int * h, int * w)
{
	TALLY_LOG("cudnnGetFilter4dDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t  filterDesc, size_t * size)
{
	TALLY_LOG("cudnnGetFilterSizeInBytes hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetFilterSizeInBytesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETFILTERSIZEINBYTES;
    
    struct cudnnGetFilterSizeInBytesArg *arg_ptr = (struct cudnnGetFilterSizeInBytesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->filterDesc = filterDesc;
	arg_ptr->size = size;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetFilterSizeInBytesResponse *) dat;
	if (size) { *size = res->size; }
	return res->err;
}

cudnnStatus_t cudnnTransformFilter(cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnFilterDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnFilterDescriptor_t  destDesc, void * destData)
{
	TALLY_LOG("cudnnTransformFilter hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t  filterDesc)
{
	TALLY_LOG("cudnnDestroyFilterDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyFilterDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYFILTERDESCRIPTOR;
    
    struct cudnnDestroyFilterDescriptorArg *arg_ptr = (struct cudnnDestroyFilterDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->filterDesc = filterDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t * poolingDesc)
{
	TALLY_LOG("cudnnCreatePoolingDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreatePoolingDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATEPOOLINGDESCRIPTOR;
    
    struct cudnnCreatePoolingDescriptorArg *arg_ptr = (struct cudnnCreatePoolingDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->poolingDesc = poolingDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreatePoolingDescriptorResponse *) dat;
	if (poolingDesc) { *poolingDesc = res->poolingDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t  mode, cudnnNanPropagation_t  maxpoolingNanOpt, int  windowHeight, int  windowWidth, int  verticalPadding, int  horizontalPadding, int  verticalStride, int  horizontalStride)
{
	TALLY_LOG("cudnnSetPooling2dDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * windowHeight, int * windowWidth, int * verticalPadding, int * horizontalPadding, int * verticalStride, int * horizontalStride)
{
	TALLY_LOG("cudnnGetPooling2dDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int * n, int * c, int * h, int * w)
{
	TALLY_LOG("cudnnGetPooling2dForwardOutputDim hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t  poolingDesc)
{
	TALLY_LOG("cudnnDestroyPoolingDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyPoolingDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYPOOLINGDESCRIPTOR;
    
    struct cudnnDestroyPoolingDescriptorArg *arg_ptr = (struct cudnnDestroyPoolingDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->poolingDesc = poolingDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t * activationDesc)
{
	TALLY_LOG("cudnnCreateActivationDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateActivationDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATEACTIVATIONDESCRIPTOR;
    
    struct cudnnCreateActivationDescriptorArg *arg_ptr = (struct cudnnCreateActivationDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->activationDesc = activationDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateActivationDescriptorResponse *) dat;
	if (activationDesc) { *activationDesc = res->activationDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t  mode, cudnnNanPropagation_t  reluNanOpt, double  coef)
{
	TALLY_LOG("cudnnSetActivationDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetActivationDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETACTIVATIONDESCRIPTOR;
    
    struct cudnnSetActivationDescriptorArg *arg_ptr = (struct cudnnSetActivationDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->activationDesc = activationDesc;
	arg_ptr->mode = mode;
	arg_ptr->reluNanOpt = reluNanOpt;
	arg_ptr->coef = coef;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t * mode, cudnnNanPropagation_t * reluNanOpt, double * coef)
{
	TALLY_LOG("cudnnGetActivationDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t  activationDesc, double  swish_beta)
{
	TALLY_LOG("cudnnSetActivationDescriptorSwishBeta hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t  activationDesc, double * swish_beta)
{
	TALLY_LOG("cudnnGetActivationDescriptorSwishBeta hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t  activationDesc)
{
	TALLY_LOG("cudnnDestroyActivationDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyActivationDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYACTIVATIONDESCRIPTOR;
    
    struct cudnnDestroyActivationDescriptorArg *arg_ptr = (struct cudnnDestroyActivationDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->activationDesc = activationDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t * normDesc)
{
	TALLY_LOG("cudnnCreateLRNDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateLRNDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATELRNDESCRIPTOR;
    
    struct cudnnCreateLRNDescriptorArg *arg_ptr = (struct cudnnCreateLRNDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->normDesc = normDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateLRNDescriptorResponse *) dat;
	if (normDesc) { *normDesc = res->normDesc; }
	return res->err;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t  normDesc, unsigned  lrnN, double  lrnAlpha, double  lrnBeta, double  lrnK)
{
	TALLY_LOG("cudnnSetLRNDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetLRNDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETLRNDESCRIPTOR;
    
    struct cudnnSetLRNDescriptorArg *arg_ptr = (struct cudnnSetLRNDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->normDesc = normDesc;
	arg_ptr->lrnN = lrnN;
	arg_ptr->lrnAlpha = lrnAlpha;
	arg_ptr->lrnBeta = lrnBeta;
	arg_ptr->lrnK = lrnK;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t  normDesc, unsigned * lrnN, double * lrnAlpha, double * lrnBeta, double * lrnK)
{
	TALLY_LOG("cudnnGetLRNDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t  lrnDesc)
{
	TALLY_LOG("cudnnDestroyLRNDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyLRNDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYLRNDESCRIPTOR;
    
    struct cudnnDestroyLRNDescriptorArg *arg_ptr = (struct cudnnDestroyLRNDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->lrnDesc = lrnDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnDivisiveNormalizationForward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t  derivedBnDesc, const cudnnTensorDescriptor_t  xDesc, cudnnBatchNormMode_t  mode)
{
	TALLY_LOG("cudnnDeriveBNTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, const void * estimatedMean, const void * estimatedVariance, double  epsilon)
{
	TALLY_LOG("cudnnBatchNormalizationForwardInference hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor(cudnnTensorDescriptor_t  derivedNormScaleBiasDesc, cudnnTensorDescriptor_t  derivedNormMeanVarDesc, const cudnnTensorDescriptor_t  xDesc, cudnnNormMode_t  mode, int  groupCnt)
{
	TALLY_LOG("cudnnDeriveNormTensorDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnNormalizationForwardInference(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * estimatedMean, const void * estimatedVariance, const cudnnTensorDescriptor_t  zDesc, const void * z, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y, double  epsilon, int  groupCnt)
{
	TALLY_LOG("cudnnNormalizationForwardInference hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t * stDesc)
{
	TALLY_LOG("cudnnCreateSpatialTransformerDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t  stDesc, cudnnSamplerType_t  samplerType, cudnnDataType_t  dataType, const int  nbDims, const int  dimA[])
{
	TALLY_LOG("cudnnSetSpatialTransformerNdDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t  stDesc)
{
	TALLY_LOG("cudnnDestroySpatialTransformerDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * theta, void * grid)
{
	TALLY_LOG("cudnnSpatialTfGridGeneratorForward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * grid, const void * beta, cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnSpatialTfSamplerForward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t * dropoutDesc)
{
	TALLY_LOG("cudnnCreateDropoutDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateDropoutDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATEDROPOUTDESCRIPTOR;
    
    struct cudnnCreateDropoutDescriptorArg *arg_ptr = (struct cudnnCreateDropoutDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dropoutDesc = dropoutDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateDropoutDescriptorResponse *) dat;
	if (dropoutDesc) { *dropoutDesc = res->dropoutDesc; }
	return res->err;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc)
{
	TALLY_LOG("cudnnDestroyDropoutDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyDropoutDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYDROPOUTDESCRIPTOR;
    
    struct cudnnDestroyDropoutDescriptorArg *arg_ptr = (struct cudnnDestroyDropoutDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dropoutDesc = dropoutDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t  handle, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnDropoutGetStatesSize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDropoutGetStatesSizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDROPOUTGETSTATESSIZE;
    
    struct cudnnDropoutGetStatesSizeArg *arg_ptr = (struct cudnnDropoutGetStatesSizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->sizeInBytes = sizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnDropoutGetStatesSizeResponse *) dat;
	if (sizeInBytes) { *sizeInBytes = res->sizeInBytes; }
	return res->err;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t  xdesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnDropoutGetReserveSpaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	TALLY_LOG("cudnnSetDropoutDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetDropoutDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETDROPOUTDESCRIPTOR;
    
    struct cudnnSetDropoutDescriptorArg *arg_ptr = (struct cudnnSetDropoutDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->dropoutDesc = dropoutDesc;
	arg_ptr->handle = handle;
	arg_ptr->dropout = dropout;
	arg_ptr->states = states;
	arg_ptr->stateSizeInBytes = stateSizeInBytes;
	arg_ptr->seed = seed;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	TALLY_LOG("cudnnRestoreDropoutDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float * dropout, void ** states, unsigned long long * seed)
{
	TALLY_LOG("cudnnGetDropoutDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDropoutForward(cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  xdesc, const void * x, const cudnnTensorDescriptor_t  ydesc, void * y, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnDropoutForward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t * algoDesc)
{
	TALLY_LOG("cudnnCreateAlgorithmDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t  algorithm)
{
	TALLY_LOG("cudnnSetAlgorithmDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t * algorithm)
{
	TALLY_LOG("cudnnGetAlgorithmDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t  src, cudnnAlgorithmDescriptor_t  dest)
{
	TALLY_LOG("cudnnCopyAlgorithmDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_LOG("cudnnDestroyAlgorithmDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t * algoPerf, int  numberToCreate)
{
	TALLY_LOG("cudnnCreateAlgorithmPerformance hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t  algoDesc, cudnnStatus_t  status, float  time, size_t  memory)
{
	TALLY_LOG("cudnnSetAlgorithmPerformance hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t * algoDesc, cudnnStatus_t * status, float * time, size_t * memory)
{
	TALLY_LOG("cudnnGetAlgorithmPerformance hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t * algoPerf, int  numberToDestroy)
{
	TALLY_LOG("cudnnDestroyAlgorithmPerformance hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, size_t * algoSpaceSizeInBytes)
{
	TALLY_LOG("cudnnGetAlgorithmSpaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSaveAlgorithm(cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, void * algoSpace, size_t  algoSpaceSizeInBytes)
{
	TALLY_LOG("cudnnSaveAlgorithm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRestoreAlgorithm(cudnnHandle_t  handle, void * algoSpace, size_t  algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_LOG("cudnnRestoreAlgorithm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetCallback(unsigned  mask, void * udata, cudnnCallback_t  fptr)
{
	TALLY_LOG("cudnnSetCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetCallback(unsigned * mask, void ** udata, cudnnCallback_t * fptr)
{
	TALLY_LOG("cudnnGetCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnOpsInferVersionCheck()
{
	TALLY_LOG("cudnnOpsInferVersionCheck hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnOpsInferVersionCheckArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNOPSINFERVERSIONCHECK;
    
    struct cudnnOpsInferVersionCheckArg *arg_ptr = (struct cudnnOpsInferVersionCheckArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_LOG("cudnnSoftmaxBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_LOG("cudnnPoolingBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_LOG("cudnnActivationBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnLRNCrossChannelBackward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_LOG("cudnnLRNCrossChannelBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, const void * dy, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  dXdMeansDesc, void * dx, void * dMeans)
{
	TALLY_LOG("cudnnDivisiveNormalizationBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetBatchNormalizationBackwardExWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetBatchNormalizationTrainingExReserveSpaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance)
{
	TALLY_LOG("cudnnBatchNormalizationForwardTraining hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnBatchNormalizationForwardTrainingEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScale, void * dBnScaleResult, void * dBnBiasResult, double  epsilon, const void * savedMean, const void * savedInvVariance)
{
	TALLY_LOG("cudnnBatchNormalizationBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScaleData, const void * bnBiasData, void * dBnScaleData, void * dBnBiasData, double  epsilon, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnBatchNormalizationBackwardEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  normScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)
{
	TALLY_LOG("cudnnGetNormalizationForwardTrainingWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)
{
	TALLY_LOG("cudnnGetNormalizationBackwardWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, int  groupCnt)
{
	TALLY_LOG("cudnnGetNormalizationTrainingReserveSpaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnNormalizationForwardTraining(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, double  exponentialAverageFactor, const cudnnTensorDescriptor_t  normMeanVarDesc, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)
{
	TALLY_LOG("cudnnNormalizationForwardTraining hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnNormalizationBackward(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const void * normScaleData, const void * normBiasData, void * dNormScaleData, void * dNormBiasData, double  epsilon, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)
{
	TALLY_LOG("cudnnNormalizationBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * dgrid, void * dtheta)
{
	TALLY_LOG("cudnnSpatialTfGridGeneratorBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx, const void * alphaDgrid, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * grid, const void * betaDgrid, void * dgrid)
{
	TALLY_LOG("cudnnSpatialTfSamplerBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  dydesc, const void * dy, const cudnnTensorDescriptor_t  dxdesc, void * dx, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnDropoutBackward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnOpsTrainVersionCheck()
{
	TALLY_LOG("cudnnOpsTrainVersionCheck hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnOpsTrainVersionCheckArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNOPSTRAINVERSIONCHECK;
    
    struct cudnnOpsTrainVersionCheckArg *arg_ptr = (struct cudnnOpsTrainVersionCheckArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t * rnnDesc)
{
	TALLY_LOG("cudnnCreateRNNDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateRNNDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATERNNDESCRIPTOR;
    
    struct cudnnCreateRNNDescriptorArg *arg_ptr = (struct cudnnCreateRNNDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->rnnDesc = rnnDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateRNNDescriptorResponse *) dat;
	if (rnnDesc) { *rnnDesc = res->rnnDesc; }
	return res->err;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t  rnnDesc)
{
	TALLY_LOG("cudnnDestroyRNNDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyRNNDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYRNNDESCRIPTOR;
    
    struct cudnnDestroyRNNDescriptorArg *arg_ptr = (struct cudnnDestroyRNNDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->rnnDesc = rnnDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t  algo, cudnnRNNMode_t  cellMode, cudnnRNNBiasMode_t  biasMode, cudnnDirectionMode_t  dirMode, cudnnRNNInputMode_t  inputMode, cudnnDataType_t  dataType, cudnnDataType_t  mathPrec, cudnnMathType_t  mathType, int32_t  inputSize, int32_t  hiddenSize, int32_t  projSize, int32_t  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, uint32_t  auxFlags)
{
	TALLY_LOG("cudnnSetRNNDescriptor_v8 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetRNNDescriptor_v8Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V8;
    
    struct cudnnSetRNNDescriptor_v8Arg *arg_ptr = (struct cudnnSetRNNDescriptor_v8Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->algo = algo;
	arg_ptr->cellMode = cellMode;
	arg_ptr->biasMode = biasMode;
	arg_ptr->dirMode = dirMode;
	arg_ptr->inputMode = inputMode;
	arg_ptr->dataType = dataType;
	arg_ptr->mathPrec = mathPrec;
	arg_ptr->mathType = mathType;
	arg_ptr->inputSize = inputSize;
	arg_ptr->hiddenSize = hiddenSize;
	arg_ptr->projSize = projSize;
	arg_ptr->numLayers = numLayers;
	arg_ptr->dropoutDesc = dropoutDesc;
	arg_ptr->auxFlags = auxFlags;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t * algo, cudnnRNNMode_t * cellMode, cudnnRNNBiasMode_t * biasMode, cudnnDirectionMode_t * dirMode, cudnnRNNInputMode_t * inputMode, cudnnDataType_t * dataType, cudnnDataType_t * mathPrec, cudnnMathType_t * mathType, int32_t * inputSize, int32_t * hiddenSize, int32_t * projSize, int32_t * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, uint32_t * auxFlags)
{
	TALLY_LOG("cudnnGetRNNDescriptor_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  hiddenSize, const int  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, cudnnRNNInputMode_t  inputMode, cudnnDirectionMode_t  direction, cudnnRNNMode_t  cellMode, cudnnRNNAlgo_t  algo, cudnnDataType_t  mathPrec)
{
	TALLY_LOG("cudnnSetRNNDescriptor_v6 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetRNNDescriptor_v6Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V6;
    
    struct cudnnSetRNNDescriptor_v6Arg *arg_ptr = (struct cudnnSetRNNDescriptor_v6Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->hiddenSize = hiddenSize;
	arg_ptr->numLayers = numLayers;
	arg_ptr->dropoutDesc = dropoutDesc;
	arg_ptr->inputMode = inputMode;
	arg_ptr->direction = direction;
	arg_ptr->cellMode = cellMode;
	arg_ptr->algo = algo;
	arg_ptr->mathPrec = mathPrec;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int * hiddenSize, int * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, cudnnRNNInputMode_t * inputMode, cudnnDirectionMode_t * direction, cudnnRNNMode_t * cellMode, cudnnRNNAlgo_t * algo, cudnnDataType_t * mathPrec)
{
	TALLY_LOG("cudnnGetRNNDescriptor_v6 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t  mType)
{
	TALLY_LOG("cudnnSetRNNMatrixMathType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t * mType)
{
	TALLY_LOG("cudnnGetRNNMatrixMathType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t  biasMode)
{
	TALLY_LOG("cudnnSetRNNBiasMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t * biasMode)
{
	TALLY_LOG("cudnnGetRNNBiasMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	TALLY_LOG("cudnnRNNSetClip_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNGetClip_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)
{
	TALLY_LOG("cudnnRNNGetClip_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	TALLY_LOG("cudnnRNNSetClip hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)
{
	TALLY_LOG("cudnnRNNGetClip hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  recProjSize, const int  outProjSize)
{
	TALLY_LOG("cudnnSetRNNProjectionLayers hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * recProjSize, int * outProjSize)
{
	TALLY_LOG("cudnnGetRNNProjectionLayers hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t  rnnDesc, const int  minibatch, const cudnnDataType_t  dataType, cudnnPersistentRNNPlan_t * plan)
{
	TALLY_LOG("cudnnCreatePersistentRNNPlan hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t  plan)
{
	TALLY_LOG("cudnnDestroyPersistentRNNPlan hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t  rnnDesc, cudnnPersistentRNNPlan_t  plan)
{
	TALLY_LOG("cudnnSetPersistentRNNPlan hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBuildRNNDynamic(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  miniBatch)
{
	TALLY_LOG("cudnnBuildRNNDynamic hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBuildRNNDynamicArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBUILDRNNDYNAMIC;
    
    struct cudnnBuildRNNDynamicArg *arg_ptr = (struct cudnnBuildRNNDynamicArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->miniBatch = miniBatch;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fMode, cudnnRNNDataDescriptor_t  xDesc, size_t * workSpaceSize, size_t * reserveSpaceSize)
{
	TALLY_LOG("cudnnGetRNNTempSpaceSizes hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNTempSpaceSizesArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNTEMPSPACESIZES;
    
    struct cudnnGetRNNTempSpaceSizesArg *arg_ptr = (struct cudnnGetRNNTempSpaceSizesArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->fMode = fMode;
	arg_ptr->xDesc = xDesc;
	arg_ptr->workSpaceSize = workSpaceSize;
	arg_ptr->reserveSpaceSize = reserveSpaceSize;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNTempSpaceSizesResponse *) dat;
	if (workSpaceSize) { *workSpaceSize = res->workSpaceSize; }
	if (reserveSpaceSize) { *reserveSpaceSize = res->reserveSpaceSize; }
	return res->err;
}

cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, cudnnDataType_t  dataType)
{
	TALLY_LOG("cudnnGetRNNParamsSize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNParamsSizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNPARAMSSIZE;
    
    struct cudnnGetRNNParamsSizeArg *arg_ptr = (struct cudnnGetRNNParamsSizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->xDesc = xDesc;
	arg_ptr->sizeInBytes = sizeInBytes;
	arg_ptr->dataType = dataType;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNParamsSizeResponse *) dat;
	if (sizeInBytes) { *sizeInBytes = res->sizeInBytes; }
	return res->err;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, size_t * weightSpaceSize)
{
	TALLY_LOG("cudnnGetRNNWeightSpaceSize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNWeightSpaceSizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNWEIGHTSPACESIZE;
    
    struct cudnnGetRNNWeightSpaceSizeArg *arg_ptr = (struct cudnnGetRNNWeightSpaceSizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->weightSpaceSize = weightSpaceSize;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNWeightSpaceSizeResponse *) dat;
	if (weightSpaceSize) { *weightSpaceSize = res->weightSpaceSize; }
	return res->err;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerMatDesc, void ** linLayerMat)
{
	TALLY_LOG("cudnnGetRNNLinLayerMatrixParams hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNLinLayerMatrixParamsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNLINLAYERMATRIXPARAMS;
    
    struct cudnnGetRNNLinLayerMatrixParamsArg *arg_ptr = (struct cudnnGetRNNLinLayerMatrixParamsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->pseudoLayer = pseudoLayer;
	arg_ptr->xDesc = xDesc;
	arg_ptr->wDesc = wDesc;
	arg_ptr->w = const_cast<void *>(w);
	arg_ptr->linLayerID = linLayerID;
	arg_ptr->linLayerMatDesc = linLayerMatDesc;
	arg_ptr->linLayerMat = linLayerMat;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNLinLayerMatrixParamsResponse *) dat;
	if (linLayerMat) { *linLayerMat = res->linLayerMat; }
	return res->err;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerBiasDesc, void ** linLayerBias)
{
	TALLY_LOG("cudnnGetRNNLinLayerBiasParams hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNLinLayerBiasParamsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNLINLAYERBIASPARAMS;
    
    struct cudnnGetRNNLinLayerBiasParamsArg *arg_ptr = (struct cudnnGetRNNLinLayerBiasParamsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->pseudoLayer = pseudoLayer;
	arg_ptr->xDesc = xDesc;
	arg_ptr->wDesc = wDesc;
	arg_ptr->w = const_cast<void *>(w);
	arg_ptr->linLayerID = linLayerID;
	arg_ptr->linLayerBiasDesc = linLayerBiasDesc;
	arg_ptr->linLayerBias = linLayerBias;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNLinLayerBiasParamsResponse *) dat;
	if (linLayerBias) { *linLayerBias = res->linLayerBias; }
	return res->err;
}

cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int32_t  pseudoLayer, size_t  weightSpaceSize, const void * weightSpace, int32_t  linLayerID, cudnnTensorDescriptor_t  mDesc, void ** mAddr, cudnnTensorDescriptor_t  bDesc, void ** bAddr)
{
	TALLY_LOG("cudnnGetRNNWeightParams hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNWeightParamsArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNWEIGHTPARAMS;
    
    struct cudnnGetRNNWeightParamsArg *arg_ptr = (struct cudnnGetRNNWeightParamsArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->pseudoLayer = pseudoLayer;
	arg_ptr->weightSpaceSize = weightSpaceSize;
	arg_ptr->weightSpace = const_cast<void *>(weightSpace);
	arg_ptr->linLayerID = linLayerID;
	arg_ptr->mDesc = mDesc;
	arg_ptr->mAddr = mAddr;
	arg_ptr->bDesc = bDesc;
	arg_ptr->bAddr = bAddr;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNWeightParamsResponse *) dat;
	if (mAddr) { *mAddr = res->mAddr; }
	if (bAddr) { *bAddr = res->bAddr; }
	return res->err;
}

cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNForwardInference hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t  rnnDesc, unsigned  paddingMode)
{
	TALLY_LOG("cudnnSetRNNPaddingMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t  rnnDesc, unsigned * paddingMode)
{
	TALLY_LOG("cudnnGetRNNPaddingMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t * rnnDataDesc)
{
	TALLY_LOG("cudnnCreateRNNDataDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateRNNDataDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATERNNDATADESCRIPTOR;
    
    struct cudnnCreateRNNDataDescriptorArg *arg_ptr = (struct cudnnCreateRNNDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->rnnDataDesc = rnnDataDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateRNNDataDescriptorResponse *) dat;
	if (rnnDataDesc) { *rnnDataDesc = res->rnnDataDesc; }
	return res->err;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc)
{
	TALLY_LOG("cudnnDestroyRNNDataDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyRNNDataDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYRNNDATADESCRIPTOR;
    
    struct cudnnDestroyRNNDataDescriptorArg *arg_ptr = (struct cudnnDestroyRNNDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->rnnDataDesc = rnnDataDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t * dataType, cudnnRNNDataLayout_t * layout, int * maxSeqLength, int * batchSize, int * vectorSize, int  arrayLengthRequested, int  seqLengthArray[], void * paddingFill)
{
	TALLY_LOG("cudnnGetRNNDataDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNForwardInferenceEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNForward(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnRNNDataDescriptor_t  yDesc, void * y, cudnnTensorDescriptor_t  hDesc, const void * hx, void * hy, cudnnTensorDescriptor_t  cDesc, const void * cx, void * cy, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	TALLY_LOG("cudnnRNNForward hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNForwardArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNRNNFORWARD;
    
    struct cudnnRNNForwardArg *arg_ptr = (struct cudnnRNNForwardArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->fwdMode = fwdMode;
	arg_ptr->devSeqLengths = const_cast<int32_t *>(devSeqLengths);
	arg_ptr->xDesc = xDesc;
	arg_ptr->x = const_cast<void *>(x);
	arg_ptr->yDesc = yDesc;
	arg_ptr->y = y;
	arg_ptr->hDesc = hDesc;
	arg_ptr->hx = const_cast<void *>(hx);
	arg_ptr->hy = hy;
	arg_ptr->cDesc = cDesc;
	arg_ptr->cx = const_cast<void *>(cx);
	arg_ptr->cy = cy;
	arg_ptr->weightSpaceSize = weightSpaceSize;
	arg_ptr->weightSpace = const_cast<void *>(weightSpace);
	arg_ptr->workSpaceSize = workSpaceSize;
	arg_ptr->workSpace = workSpace;
	arg_ptr->reserveSpaceSize = reserveSpaceSize;
	arg_ptr->reserveSpace = reserveSpace;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_LOG("cudnnSetRNNAlgorithmDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetRNNAlgorithmDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETRNNALGORITHMDESCRIPTOR;
    
    struct cudnnSetRNNAlgorithmDescriptorArg *arg_ptr = (struct cudnnSetRNNAlgorithmDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->algoDesc = algoDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_LOG("cudnnGetRNNForwardInferenceAlgorithmMaxCount hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetRNNForwardInferenceAlgorithmMaxCountArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETRNNFORWARDINFERENCEALGORITHMMAXCOUNT;
    
    struct cudnnGetRNNForwardInferenceAlgorithmMaxCountArg *arg_ptr = (struct cudnnGetRNNForwardInferenceAlgorithmMaxCountArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->count = count;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetRNNForwardInferenceAlgorithmMaxCountResponse *) dat;
	if (count) { *count = res->count; }
	return res->err;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindRNNForwardInferenceAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t * seqDataDesc)
{
	TALLY_LOG("cudnnCreateSeqDataDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateSeqDataDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATESEQDATADESCRIPTOR;
    
    struct cudnnCreateSeqDataDescriptorArg *arg_ptr = (struct cudnnCreateSeqDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->seqDataDesc = seqDataDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateSeqDataDescriptorResponse *) dat;
	if (seqDataDesc) { *seqDataDesc = res->seqDataDesc; }
	return res->err;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t  seqDataDesc)
{
	TALLY_LOG("cudnnDestroySeqDataDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroySeqDataDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYSEQDATADESCRIPTOR;
    
    struct cudnnDestroySeqDataDescriptorArg *arg_ptr = (struct cudnnDestroySeqDataDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->seqDataDesc = seqDataDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t * attnDesc)
{
	TALLY_LOG("cudnnCreateAttnDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateAttnDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATEATTNDESCRIPTOR;
    
    struct cudnnCreateAttnDescriptorArg *arg_ptr = (struct cudnnCreateAttnDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->attnDesc = attnDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateAttnDescriptorResponse *) dat;
	if (attnDesc) { *attnDesc = res->attnDesc; }
	return res->err;
}

cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t  attnDesc)
{
	TALLY_LOG("cudnnDestroyAttnDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyAttnDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYATTNDESCRIPTOR;
    
    struct cudnnDestroyAttnDescriptorArg *arg_ptr = (struct cudnnDestroyAttnDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->attnDesc = attnDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnDescriptor_t  attnDesc, unsigned  attnMode, int  nHeads, double  smScaler, cudnnDataType_t  dataType, cudnnDataType_t  computePrec, cudnnMathType_t  mathType, cudnnDropoutDescriptor_t  attnDropoutDesc, cudnnDropoutDescriptor_t  postDropoutDesc, int  qSize, int  kSize, int  vSize, int  qProjSize, int  kProjSize, int  vProjSize, int  oProjSize, int  qoMaxSeqLength, int  kvMaxSeqLength, int  maxBatchSize, int  maxBeamSize)
{
	TALLY_LOG("cudnnSetAttnDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnSetAttnDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNSETATTNDESCRIPTOR;
    
    struct cudnnSetAttnDescriptorArg *arg_ptr = (struct cudnnSetAttnDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->attnDesc = attnDesc;
	arg_ptr->attnMode = attnMode;
	arg_ptr->nHeads = nHeads;
	arg_ptr->smScaler = smScaler;
	arg_ptr->dataType = dataType;
	arg_ptr->computePrec = computePrec;
	arg_ptr->mathType = mathType;
	arg_ptr->attnDropoutDesc = attnDropoutDesc;
	arg_ptr->postDropoutDesc = postDropoutDesc;
	arg_ptr->qSize = qSize;
	arg_ptr->kSize = kSize;
	arg_ptr->vSize = vSize;
	arg_ptr->qProjSize = qProjSize;
	arg_ptr->kProjSize = kProjSize;
	arg_ptr->vProjSize = vProjSize;
	arg_ptr->oProjSize = oProjSize;
	arg_ptr->qoMaxSeqLength = qoMaxSeqLength;
	arg_ptr->kvMaxSeqLength = kvMaxSeqLength;
	arg_ptr->maxBatchSize = maxBatchSize;
	arg_ptr->maxBeamSize = maxBeamSize;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnDescriptor_t  attnDesc, unsigned * attnMode, int * nHeads, double * smScaler, cudnnDataType_t * dataType, cudnnDataType_t * computePrec, cudnnMathType_t * mathType, cudnnDropoutDescriptor_t * attnDropoutDesc, cudnnDropoutDescriptor_t * postDropoutDesc, int * qSize, int * kSize, int * vSize, int * qProjSize, int * kProjSize, int * vProjSize, int * oProjSize, int * qoMaxSeqLength, int * kvMaxSeqLength, int * maxBatchSize, int * maxBeamSize)
{
	TALLY_LOG("cudnnGetAttnDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, size_t * weightSizeInBytes, size_t * workSpaceSizeInBytes, size_t * reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnGetMultiHeadAttnBuffers hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetMultiHeadAttnBuffersArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETMULTIHEADATTNBUFFERS;
    
    struct cudnnGetMultiHeadAttnBuffersArg *arg_ptr = (struct cudnnGetMultiHeadAttnBuffersArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->attnDesc = attnDesc;
	arg_ptr->weightSizeInBytes = weightSizeInBytes;
	arg_ptr->workSpaceSizeInBytes = workSpaceSizeInBytes;
	arg_ptr->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetMultiHeadAttnBuffersResponse *) dat;
	if (weightSizeInBytes) { *weightSizeInBytes = res->weightSizeInBytes; }
	if (workSpaceSizeInBytes) { *workSpaceSizeInBytes = res->workSpaceSizeInBytes; }
	if (reserveSpaceSizeInBytes) { *reserveSpaceSizeInBytes = res->reserveSpaceSizeInBytes; }
	return res->err;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnMultiHeadAttnWeightKind_t  wKind, size_t  weightSizeInBytes, const void * weights, cudnnTensorDescriptor_t  wDesc, void ** wAddr)
{
	TALLY_LOG("cudnnGetMultiHeadAttnWeights hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnAdvInferVersionCheck()
{
	TALLY_LOG("cudnnAdvInferVersionCheck hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnAdvInferVersionCheckArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNADVINFERVERSIONCHECK;
    
    struct cudnnAdvInferVersionCheckArg *arg_ptr = (struct cudnnAdvInferVersionCheckArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  yDesc, const void * y, const void * dy, cudnnRNNDataDescriptor_t  xDesc, void * dx, cudnnTensorDescriptor_t  hDesc, const void * hx, const void * dhy, void * dhx, cudnnTensorDescriptor_t  cDesc, const void * cx, const void * dcy, void * dcx, size_t  weightSpaceSize, const void * weightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	TALLY_LOG("cudnnRNNBackwardData_v8 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNBackwardData_v8Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDDATA_V8;
    
    struct cudnnRNNBackwardData_v8Arg *arg_ptr = (struct cudnnRNNBackwardData_v8Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->devSeqLengths = const_cast<int32_t *>(devSeqLengths);
	arg_ptr->yDesc = yDesc;
	arg_ptr->y = const_cast<void *>(y);
	arg_ptr->dy = const_cast<void *>(dy);
	arg_ptr->xDesc = xDesc;
	arg_ptr->dx = dx;
	arg_ptr->hDesc = hDesc;
	arg_ptr->hx = const_cast<void *>(hx);
	arg_ptr->dhy = const_cast<void *>(dhy);
	arg_ptr->dhx = dhx;
	arg_ptr->cDesc = cDesc;
	arg_ptr->cx = const_cast<void *>(cx);
	arg_ptr->dcy = const_cast<void *>(dcy);
	arg_ptr->dcx = dcx;
	arg_ptr->weightSpaceSize = weightSpaceSize;
	arg_ptr->weightSpace = const_cast<void *>(weightSpace);
	arg_ptr->workSpaceSize = workSpaceSize;
	arg_ptr->workSpace = workSpace;
	arg_ptr->reserveSpaceSize = reserveSpaceSize;
	arg_ptr->reserveSpace = reserveSpace;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnRNNBackwardWeights_v8(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnWgradMode_t  addGrad, const int32_t  devSeqLengths[], cudnnRNNDataDescriptor_t  xDesc, const void * x, cudnnTensorDescriptor_t  hDesc, const void * hx, cudnnRNNDataDescriptor_t  yDesc, const void * y, size_t  weightSpaceSize, void * dweightSpace, size_t  workSpaceSize, void * workSpace, size_t  reserveSpaceSize, void * reserveSpace)
{
	TALLY_LOG("cudnnRNNBackwardWeights_v8 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnRNNBackwardWeights_v8Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNRNNBACKWARDWEIGHTS_V8;
    
    struct cudnnRNNBackwardWeights_v8Arg *arg_ptr = (struct cudnnRNNBackwardWeights_v8Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->rnnDesc = rnnDesc;
	arg_ptr->addGrad = addGrad;
	arg_ptr->devSeqLengths = const_cast<int32_t *>(devSeqLengths);
	arg_ptr->xDesc = xDesc;
	arg_ptr->x = const_cast<void *>(x);
	arg_ptr->hDesc = hDesc;
	arg_ptr->hx = const_cast<void *>(hx);
	arg_ptr->yDesc = yDesc;
	arg_ptr->y = const_cast<void *>(y);
	arg_ptr->weightSpaceSize = weightSpaceSize;
	arg_ptr->dweightSpace = dweightSpace;
	arg_ptr->workSpaceSize = workSpaceSize;
	arg_ptr->workSpace = workSpace;
	arg_ptr->reserveSpaceSize = reserveSpaceSize;
	arg_ptr->reserveSpace = reserveSpace;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnRNNForwardTrainingEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNForwardTrainingEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNBackwardDataEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  yDesc, const void * y, const cudnnRNNDataDescriptor_t  dyDesc, const void * dy, const cudnnRNNDataDescriptor_t  dcDesc, const void * dcAttn, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnRNNDataDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const cudnnRNNDataDescriptor_t  dkDesc, void * dkeys, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNBackwardDataEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnRNNDataDescriptor_t  yDesc, const void * y, void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNBackwardWeightsEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_LOG("cudnnGetRNNForwardTrainingAlgorithmMaxCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindRNNForwardTrainingAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_LOG("cudnnGetRNNBackwardDataAlgorithmMaxCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindRNNBackwardDataAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_LOG("cudnnGetRNNBackwardWeightsAlgorithmMaxCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, const void * workspace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindRNNBackwardWeightsAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t * ctcLossDesc)
{
	TALLY_LOG("cudnnCreateCTCLossDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType)
{
	TALLY_LOG("cudnnSetCTCLossDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode)
{
	TALLY_LOG("cudnnSetCTCLossDescriptorEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode, int  maxLabelLength)
{
	TALLY_LOG("cudnnSetCTCLossDescriptor_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType)
{
	TALLY_LOG("cudnnGetCTCLossDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode)
{
	TALLY_LOG("cudnnGetCTCLossDescriptorEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode, int * maxLabelLength)
{
	TALLY_LOG("cudnnGetCTCLossDescriptor_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc)
{
	TALLY_LOG("cudnnDestroyCTCLossDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCTCLoss(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  hostLabels[], const int  hostLabelLengths[], const int  hostInputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, void * workspace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnCTCLoss hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  labels[], const int  labelLengths[], const int  inputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, size_t  workSpaceSizeInBytes, void * workspace)
{
	TALLY_LOG("cudnnCTCLoss_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, const int * labels, const int * labelLengths, const int * inputLengths, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetCTCLossWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetCTCLossWorkspaceSize_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnAdvTrainVersionCheck()
{
	TALLY_LOG("cudnnAdvTrainVersionCheck hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnAdvTrainVersionCheckArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNADVTRAINVERSIONCHECK;
    
    struct cudnnAdvTrainVersionCheckArg *arg_ptr = (struct cudnnAdvTrainVersionCheckArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t * convDesc)
{
	TALLY_LOG("cudnnCreateConvolutionDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCreateConvolutionDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCREATECONVOLUTIONDESCRIPTOR;
    
    struct cudnnCreateConvolutionDescriptorArg *arg_ptr = (struct cudnnCreateConvolutionDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->convDesc = convDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnCreateConvolutionDescriptorResponse *) dat;
	if (convDesc) { *convDesc = res->convDesc; }
	return res->err;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t  convDesc)
{
	TALLY_LOG("cudnnDestroyConvolutionDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnDestroyConvolutionDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNDESTROYCONVOLUTIONDESCRIPTOR;
    
    struct cudnnDestroyConvolutionDescriptorArg *arg_ptr = (struct cudnnDestroyConvolutionDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->convDesc = convDesc;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t  mathType)
{
	TALLY_LOG("cudnnSetConvolutionMathType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t * mathType)
{
	TALLY_LOG("cudnnGetConvolutionMathType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t  convDesc, int  groupCount)
{
	TALLY_LOG("cudnnSetConvolutionGroupCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t  convDesc, int * groupCount)
{
	TALLY_LOG("cudnnGetConvolutionGroupCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t  reorderType)
{
	TALLY_LOG("cudnnSetConvolutionReorderType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t * reorderType)
{
	TALLY_LOG("cudnnGetConvolutionReorderType hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t  convDesc, int  pad_h, int  pad_w, int  u, int  v, int  dilation_h, int  dilation_w, cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)
{
	TALLY_LOG("cudnnSetConvolution2dDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t  convDesc, int * pad_h, int * pad_w, int * u, int * v, int * dilation_h, int * dilation_w, cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)
{
	TALLY_LOG("cudnnGetConvolution2dDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t  convDesc, int  arrayLengthRequested, int * arrayLength, int  padA[], int  strideA[], int  dilationA[], cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)
{
	TALLY_LOG("cudnnGetConvolutionNdDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int * n, int * c, int * h, int * w)
{
	TALLY_LOG("cudnnGetConvolution2dForwardOutputDim hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_LOG("cudnnGetConvolutionForwardAlgorithmMaxCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, void * y, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindConvolutionForwardAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnIm2Col(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, void * colBuffer)
{
	TALLY_LOG("cudnnIm2Col hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, cudnnConvolutionFwdAlgo_t  algo, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetConvolutionForwardWorkspaceSize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnGetConvolutionForwardWorkspaceSizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDWORKSPACESIZE;
    
    struct cudnnGetConvolutionForwardWorkspaceSizeArg *arg_ptr = (struct cudnnGetConvolutionForwardWorkspaceSizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->xDesc = xDesc;
	arg_ptr->wDesc = wDesc;
	arg_ptr->convDesc = convDesc;
	arg_ptr->yDesc = yDesc;
	arg_ptr->algo = algo;
	arg_ptr->sizeInBytes = sizeInBytes;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnGetConvolutionForwardWorkspaceSizeResponse *) dat;
	if (sizeInBytes) { *sizeInBytes = res->sizeInBytes; }
	return res->err;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward(cudnnHandle_t  handle, const void * alpha1, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * alpha2, const cudnnTensorDescriptor_t  zDesc, const void * z, const cudnnTensorDescriptor_t  biasDesc, const void * bias, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_LOG("cudnnConvolutionBiasActivationForward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_LOG("cudnnGetConvolutionBackwardDataAlgorithmMaxCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnFindConvolutionBackwardDataAlgorithm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, void * dx, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindConvolutionBackwardDataAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnGetConvolutionBackwardDataAlgorithm_v7 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, cudnnConvolutionBwdDataAlgo_t  algo, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetConvolutionBackwardDataWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t  handle, const void * alpha, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdDataAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_LOG("cudnnConvolutionBackwardData hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const cudnnTensorFormat_t  transformFormat, cudnnFilterDescriptor_t  foldedFilterDesc, cudnnTensorDescriptor_t  paddedDiffDesc, cudnnConvolutionDescriptor_t  foldedConvDesc, cudnnTensorDescriptor_t  foldedGradDesc, cudnnTensorTransformDescriptor_t  filterFoldTransDesc, cudnnTensorTransformDescriptor_t  diffPadTransDesc, cudnnTensorTransformDescriptor_t  gradFoldTransDesc, cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc)
{
	TALLY_LOG("cudnnGetFoldedConvBackwardDataDescriptors hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCnnInferVersionCheck()
{
	TALLY_LOG("cudnnCnnInferVersionCheck hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCnnInferVersionCheckArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCNNINFERVERSIONCHECK;
    
    struct cudnnCnnInferVersionCheckArg *arg_ptr = (struct cudnnCnnInferVersionCheckArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_LOG("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnFindConvolutionBackwardFilterAlgorithm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * y, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, void * dw, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnFindConvolutionBackwardFilterAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)
{
	TALLY_LOG("cudnnGetConvolutionBackwardFilterAlgorithm_v7 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, cudnnConvolutionBwdFilterAlgo_t  algo, size_t * sizeInBytes)
{
	TALLY_LOG("cudnnGetConvolutionBackwardFilterWorkspaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdFilterAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnFilterDescriptor_t  dwDesc, void * dw)
{
	TALLY_LOG("cudnnConvolutionBackwardFilter hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dbDesc, void * db)
{
	TALLY_LOG("cudnnConvolutionBackwardBias hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t * constPack, cudnnFusedOps_t  ops)
{
	TALLY_LOG("cudnnCreateFusedOpsConstParamPack hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t  constPack)
{
	TALLY_LOG("cudnnDestroyFusedOpsConstParamPack hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, const void * param)
{
	TALLY_LOG("cudnnSetFusedOpsConstParamPackAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, void * param, int * isNULL)
{
	TALLY_LOG("cudnnGetFusedOpsConstParamPackAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t * varPack, cudnnFusedOps_t  ops)
{
	TALLY_LOG("cudnnCreateFusedOpsVariantParamPack hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t  varPack)
{
	TALLY_LOG("cudnnDestroyFusedOpsVariantParamPack hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)
{
	TALLY_LOG("cudnnSetFusedOpsVariantParamPackAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)
{
	TALLY_LOG("cudnnGetFusedOpsVariantParamPackAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t * plan, cudnnFusedOps_t  ops)
{
	TALLY_LOG("cudnnCreateFusedOpsPlan hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t  plan)
{
	TALLY_LOG("cudnnDestroyFusedOpsPlan hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnMakeFusedOpsPlan(cudnnHandle_t  handle, cudnnFusedOpsPlan_t  plan, const cudnnFusedOpsConstParamPack_t  constPack, size_t * workspaceSizeInBytes)
{
	TALLY_LOG("cudnnMakeFusedOpsPlan hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t  handle, const cudnnFusedOpsPlan_t  plan, cudnnFusedOpsVariantParamPack_t  varPack)
{
	TALLY_LOG("cudnnFusedOpsExecute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCnnTrainVersionCheck()
{
	TALLY_LOG("cudnnCnnTrainVersionCheck hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnCnnTrainVersionCheckArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNCNNTRAINVERSIONCHECK;
    
    struct cudnnCnnTrainVersionCheckArg *arg_ptr = (struct cudnnCnnTrainVersionCheckArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t  descriptorType, cudnnBackendDescriptor_t * descriptor)
{
	TALLY_LOG("cudnnBackendCreateDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendCreateDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDCREATEDESCRIPTOR;
    
    struct cudnnBackendCreateDescriptorArg *arg_ptr = (struct cudnnBackendCreateDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->descriptorType = descriptorType;
	arg_ptr->descriptor = descriptor;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cudnnBackendCreateDescriptorResponse *) dat;
	if (descriptor) { *descriptor = res->descriptor; }
	return res->err;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t  descriptor)
{
	TALLY_LOG("cudnnBackendDestroyDescriptor hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendDestroyDescriptorArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDDESTROYDESCRIPTOR;
    
    struct cudnnBackendDestroyDescriptorArg *arg_ptr = (struct cudnnBackendDestroyDescriptorArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->descriptor = descriptor;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t  descriptor)
{
	TALLY_LOG("cudnnBackendInitialize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendInitializeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDINITIALIZE;
    
    struct cudnnBackendInitializeArg *arg_ptr = (struct cudnnBackendInitializeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->descriptor = descriptor;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t  descriptor)
{
	TALLY_LOG("cudnnBackendFinalize hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendFinalizeArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDFINALIZE;
    
    struct cudnnBackendFinalizeArg *arg_ptr = (struct cudnnBackendFinalizeArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->descriptor = descriptor;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t  handle, cudnnBackendDescriptor_t  executionPlan, cudnnBackendDescriptor_t  variantPack)
{
	TALLY_LOG("cudnnBackendExecute hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cudnnBackendExecuteArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUDNNBACKENDEXECUTE;
    
    struct cudnnBackendExecuteArg *arg_ptr = (struct cudnnBackendExecuteArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->executionPlan = executionPlan;
	arg_ptr->variantPack = variantPack;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cudnnStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t*  handle)
{
	TALLY_LOG("cublasCreate_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasCreate_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASCREATE_V2;
    
    struct cublasCreate_v2Arg *arg_ptr = (struct cublasCreate_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasCreate_v2Response *) dat;
	if (handle) { *handle = res->handle; }
	return res->err;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t  handle)
{
	TALLY_LOG("cublasDestroy_v2 hooked");

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
	TALLY_LOG("cublasGetVersion_v2 hooked");

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
	if (version) { *version = res->version; }
	return res->err;
}

cublasStatus_t cublasGetProperty(libraryPropertyType  type, int*  value)
{
	TALLY_LOG("cublasGetProperty hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

size_t cublasGetCudartVersion()
{
	TALLY_LOG("cublasGetCudartVersion hooked");

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
	TALLY_LOG("cublasSetWorkspace_v2 hooked");

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
	TALLY_LOG("cublasSetStream_v2 hooked");

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
	TALLY_LOG("cublasGetStream_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasGetStream_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASGETSTREAM_V2;
    
    struct cublasGetStream_v2Arg *arg_ptr = (struct cublasGetStream_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->streamId = streamId;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasGetStream_v2Response *) dat;
	if (streamId) { *streamId = res->streamId; }
	return res->err;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t  handle, cublasPointerMode_t*  mode)
{
	TALLY_LOG("cublasGetPointerMode_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasGetPointerMode_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASGETPOINTERMODE_V2;
    
    struct cublasGetPointerMode_v2Arg *arg_ptr = (struct cublasGetPointerMode_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->mode = mode;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasGetPointerMode_v2Response *) dat;
	if (mode) { *mode = res->mode; }
	return res->err;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t  handle, cublasPointerMode_t  mode)
{
	TALLY_LOG("cublasSetPointerMode_v2 hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasSetPointerMode_v2Arg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASSETPOINTERMODE_V2;
    
    struct cublasSetPointerMode_v2Arg *arg_ptr = (struct cublasSetPointerMode_v2Arg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->handle = handle;
	arg_ptr->mode = mode;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t  handle, cublasAtomicsMode_t*  mode)
{
	TALLY_LOG("cublasGetAtomicsMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t  handle, cublasAtomicsMode_t  mode)
{
	TALLY_LOG("cublasSetAtomicsMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMathMode(cublasHandle_t  handle, cublasMath_t*  mode)
{
	TALLY_LOG("cublasGetMathMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMathMode(cublasHandle_t  handle, cublasMath_t  mode)
{
	TALLY_LOG("cublasSetMathMode hooked");

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
	TALLY_LOG("cublasGetSmCountTarget hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t  handle, int  smCountTarget)
{
	TALLY_LOG("cublasSetSmCountTarget hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

const char* cublasGetStatusName(cublasStatus_t  status)
{
	TALLY_LOG("cublasGetStatusName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

const char* cublasGetStatusString(cublasStatus_t  status)
{
	TALLY_LOG("cublasGetStatusString hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLoggerConfigure(int  logIsOn, int  logToStdOut, int  logToStdErr, const char*  logFileName)
{
	TALLY_LOG("cublasLoggerConfigure hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback  userCallback)
{
	TALLY_LOG("cublasSetLoggerCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback*  userCallback)
{
	TALLY_LOG("cublasGetLoggerCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetVector(int  n, int  elemSize, const void*  x, int  incx, void*  devicePtr, int  incy)
{
	TALLY_LOG("cublasSetVector hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetVector(int  n, int  elemSize, const void*  x, int  incx, void*  y, int  incy)
{
	TALLY_LOG("cublasGetVector hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	TALLY_LOG("cublasSetMatrix hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	TALLY_LOG("cublasGetMatrix hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetVectorAsync(int  n, int  elemSize, const void*  hostPtr, int  incx, void*  devicePtr, int  incy, cudaStream_t  stream)
{
	TALLY_LOG("cublasSetVectorAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetVectorAsync(int  n, int  elemSize, const void*  devicePtr, int  incx, void*  hostPtr, int  incy, cudaStream_t  stream)
{
	TALLY_LOG("cublasGetVectorAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	TALLY_LOG("cublasSetMatrixAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	TALLY_LOG("cublasGetMatrixAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

void cublasXerbla(const char*  srName, int  info)
{
	TALLY_LOG("cublasXerbla hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasNrm2Ex(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasNrm2Ex hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasSnrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDnrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScnrm2_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasScnrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDznrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDotEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasDotEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDotcEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasDotcEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSdot_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, const float*  y, int  incy, float*  result)
{
	TALLY_LOG("cublasSdot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDdot_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, const double*  y, int  incy, double*  result)
{
	TALLY_LOG("cublasDdot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	TALLY_LOG("cublasCdotu_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	TALLY_LOG("cublasCdotc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	TALLY_LOG("cublasZdotu_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	TALLY_LOG("cublasZdotc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScalEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int  incx, cudaDataType  executionType)
{
	TALLY_LOG("cublasScalEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, float*  x, int  incx)
{
	TALLY_LOG("cublasSscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, double*  x, int  incx)
{
	TALLY_LOG("cublasDscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCscal_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCsscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZscal_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZdscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasAxpyEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, cudaDataType  executiontype)
{
	TALLY_LOG("cublasAxpyEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t  handle, int  n, const float*  alpha, const float*  x, int  incx, float*  y, int  incy)
{
	TALLY_LOG("cublasSaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t  handle, int  n, const double*  alpha, const double*  x, int  incx, double*  y, int  incy)
{
	TALLY_LOG("cublasDaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCaxpy_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCopyEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	TALLY_LOG("cublasCopyEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScopy_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  y, int  incy)
{
	TALLY_LOG("cublasScopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  y, int  incy)
{
	TALLY_LOG("cublasDcopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCcopy_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCcopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZcopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSswap_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy)
{
	TALLY_LOG("cublasSswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDswap_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy)
{
	TALLY_LOG("cublasDswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCswap_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZswap_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSwapEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	TALLY_LOG("cublasSwapEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIsamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIdamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIcamax_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIcamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIzamax_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIzamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIamaxEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	TALLY_LOG("cublasIamaxEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIsamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIdamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIcamin_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIcamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIzamin_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIzamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIaminEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	TALLY_LOG("cublasIaminEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasAsumEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasAsumEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSasum_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasSasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDasum_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScasum_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasScasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDzasum_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDzasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSrot_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  c, const float*  s)
{
	TALLY_LOG("cublasSrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrot_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  c, const double*  s)
{
	TALLY_LOG("cublasDrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const cuComplex*  s)
{
	TALLY_LOG("cublasCrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const float*  s)
{
	TALLY_LOG("cublasCsrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const cuDoubleComplex*  s)
{
	TALLY_LOG("cublasZrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const double*  s)
{
	TALLY_LOG("cublasZdrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSrotg_v2(cublasHandle_t  handle, float*  a, float*  b, float*  c, float*  s)
{
	TALLY_LOG("cublasSrotg_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrotg_v2(cublasHandle_t  handle, double*  a, double*  b, double*  c, double*  s)
{
	TALLY_LOG("cublasDrotg_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCrotg_v2(cublasHandle_t  handle, cuComplex*  a, cuComplex*  b, float*  c, cuComplex*  s)
{
	TALLY_LOG("cublasCrotg_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZrotg_v2(cublasHandle_t  handle, cuDoubleComplex*  a, cuDoubleComplex*  b, double*  c, cuDoubleComplex*  s)
{
	TALLY_LOG("cublasZrotg_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotgEx(cublasHandle_t  handle, void*  a, void*  b, cudaDataType  abType, void*  c, void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotgEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSrotm_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  param)
{
	TALLY_LOG("cublasSrotm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrotm_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  param)
{
	TALLY_LOG("cublasDrotm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotmEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSrotmg_v2(cublasHandle_t  handle, float*  d1, float*  d2, float*  x1, const float*  y1, float*  param)
{
	TALLY_LOG("cublasSrotmg_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrotmg_v2(cublasHandle_t  handle, double*  d1, double*  d2, double*  x1, const double*  y1, double*  param)
{
	TALLY_LOG("cublasDrotmg_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotmgEx(cublasHandle_t  handle, void*  d1, cudaDataType  d1Type, void*  d2, cudaDataType  d2Type, void*  x1, cudaDataType  x1Type, const void*  y1, cudaDataType  y1Type, void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotmgEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDgemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCgemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZgemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	TALLY_LOG("cublasStpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	TALLY_LOG("cublasDtpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	TALLY_LOG("cublasStpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	TALLY_LOG("cublasDtpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasChemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZhemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSsbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDsbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasChbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZhbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  AP, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSspmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  AP, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDspmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasChpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZhpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSger_v2(cublasHandle_t  handle, int  m, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	TALLY_LOG("cublasSger_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDger_v2(cublasHandle_t  handle, int  m, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	TALLY_LOG("cublasDger_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCgeru_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCgerc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZgeru_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZgerc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  A, int  lda)
{
	TALLY_LOG("cublasSsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  A, int  lda)
{
	TALLY_LOG("cublasDsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCher_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZher_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  AP)
{
	TALLY_LOG("cublasSspr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  AP)
{
	TALLY_LOG("cublasDspr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  AP)
{
	TALLY_LOG("cublasChpr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZhpr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	TALLY_LOG("cublasSsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	TALLY_LOG("cublasDsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCher2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZher2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  AP)
{
	TALLY_LOG("cublasSspr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  AP)
{
	TALLY_LOG("cublasDspr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  AP)
{
	TALLY_LOG("cublasChpr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZhpr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDgemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCgemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCgemm3m hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCgemm3mEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZgemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZgemm3m hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemm(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, const __half*  B, int  ldb, const __half*  beta, __half*  C, int  ldc)
{
	TALLY_LOG("cublasHgemm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCgemmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasUint8gemmBias(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, cublasOperation_t  transc, int  m, int  n, int  k, const unsigned char*  A, int  A_bias, int  lda, const unsigned char*  B, int  B_bias, int  ldb, unsigned char*  C, int  C_bias, int  ldc, int  C_mult, int  C_shift)
{
	TALLY_LOG("cublasUint8gemmBias hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCsyrkEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCsyrk3mEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const cuComplex*  A, int  lda, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCherk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const cuDoubleComplex*  A, int  lda, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZherk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCherkEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCherk3mEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCher2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZher2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCherkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZherkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasChemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZhemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, float*  B, int  ldb)
{
	TALLY_LOG("cublasStrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, double*  B, int  ldb)
{
	TALLY_LOG("cublasDtrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, cuComplex*  B, int  ldb)
{
	TALLY_LOG("cublasCtrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  B, int  ldb)
{
	TALLY_LOG("cublasZtrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, float*  C, int  ldc)
{
	TALLY_LOG("cublasStrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, double*  C, int  ldc)
{
	TALLY_LOG("cublasDtrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCtrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZtrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half* const  Aarray[], int  lda, const __half* const  Barray[], int  ldb, const __half*  beta, __half* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasHgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  Barray[], int  ldb, const float*  beta, float* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasSgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  Barray[], int  ldb, const double*  beta, double* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasDgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasCgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasCgemm3mBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  Barray[], int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasZgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int  lda, const void* const  Barray[], cudaDataType  Btype, int  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int  ldc, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmBatchedEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, long long int  strideC, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmStridedBatchedEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  B, int  ldb, long long int  strideB, const float*  beta, float*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasSgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  B, int  ldb, long long int  strideB, const double*  beta, double*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasDgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasCgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasCgemm3mStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  B, int  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasZgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  B, int  ldb, long long int  strideB, const __half*  beta, __half*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasHgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  beta, const float*  B, int  ldb, float*  C, int  ldc)
{
	TALLY_LOG("cublasSgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  beta, const double*  B, int  ldb, double*  C, int  ldc)
{
	TALLY_LOG("cublasDgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t  handle, int  n, float* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_LOG("cublasSgetrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t  handle, int  n, double* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_LOG("cublasDgetrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t  handle, int  n, cuComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_LOG("cublasCgetrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t  handle, int  n, cuDoubleComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_LOG("cublasZgetrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t  handle, int  n, const float* const  A[], int  lda, const int*  P, float* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_LOG("cublasSgetriBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t  handle, int  n, const double* const  A[], int  lda, const int*  P, double* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_LOG("cublasDgetriBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, const int*  P, cuComplex* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_LOG("cublasCgetriBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, const int*  P, cuDoubleComplex* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_LOG("cublasZgetriBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const float* const  Aarray[], int  lda, const int*  devIpiv, float* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_LOG("cublasSgetrsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const double* const  Aarray[], int  lda, const int*  devIpiv, double* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_LOG("cublasDgetrsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuComplex* const  Aarray[], int  lda, const int*  devIpiv, cuComplex* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_LOG("cublasCgetrsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuDoubleComplex* const  Aarray[], int  lda, const int*  devIpiv, cuDoubleComplex* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_LOG("cublasZgetrsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float* const  A[], int  lda, float* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasStrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double* const  A[], int  lda, double* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasDtrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  A[], int  lda, cuComplex* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasCtrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasZtrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSmatinvBatched(cublasHandle_t  handle, int  n, const float* const  A[], int  lda, float* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_LOG("cublasSmatinvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDmatinvBatched(cublasHandle_t  handle, int  n, const double* const  A[], int  lda, double* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_LOG("cublasDmatinvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCmatinvBatched(cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, cuComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_LOG("cublasCmatinvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZmatinvBatched(cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_LOG("cublasZmatinvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t  handle, int  m, int  n, float* const  Aarray[], int  lda, float* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_LOG("cublasSgeqrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t  handle, int  m, int  n, double* const  Aarray[], int  lda, double* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_LOG("cublasDgeqrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t  handle, int  m, int  n, cuComplex* const  Aarray[], int  lda, cuComplex* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_LOG("cublasCgeqrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t  handle, int  m, int  n, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_LOG("cublasZgeqrfBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, float* const  Aarray[], int  lda, float* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_LOG("cublasSgelsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, double* const  Aarray[], int  lda, double* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_LOG("cublasDgelsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuComplex* const  Aarray[], int  lda, cuComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_LOG("cublasCgelsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_LOG("cublasZgelsBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const float*  A, int  lda, const float*  x, int  incx, float*  C, int  ldc)
{
	TALLY_LOG("cublasSdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const double*  A, int  lda, const double*  x, int  incx, double*  C, int  ldc)
{
	TALLY_LOG("cublasDdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  AP, float*  A, int  lda)
{
	TALLY_LOG("cublasStpttr hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  AP, double*  A, int  lda)
{
	TALLY_LOG("cublasDtpttr hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  AP, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCtpttr hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZtpttr hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  A, int  lda, float*  AP)
{
	TALLY_LOG("cublasStrttp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  A, int  lda, double*  AP)
{
	TALLY_LOG("cublasDtrttp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  A, int  lda, cuComplex*  AP)
{
	TALLY_LOG("cublasCtrttp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZtrttp hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaProfilerInitialize(const char * configFile, const char * outputFile, cudaOutputMode_t  outputMode)
{
	TALLY_LOG("cudaProfilerInitialize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaProfilerStart()
{
	TALLY_LOG("cudaProfilerStart hooked");

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
	TALLY_LOG("cudaProfilerStop hooked");

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
	TALLY_LOG("nvrtcGetCUBINSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetCUBIN(nvrtcProgram  prog, char * cubin)
{
	TALLY_LOG("nvrtcGetCUBIN hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t*  lightHandle)
{
	TALLY_LOG("cublasLtCreate hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTCREATE;
    
    struct cublasLtCreateArg *arg_ptr = (struct cublasLtCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->lightHandle = lightHandle;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasLtCreateResponse *) dat;
	if (lightHandle) { *lightHandle = res->lightHandle; }
	return res->err;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t  lightHandle)
{
	TALLY_LOG("cublasLtDestroy hooked");

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
	TALLY_LOG("cublasLtGetStatusName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

const char* cublasLtGetStatusString(cublasStatus_t  status)
{
	TALLY_LOG("cublasLtGetStatusString hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

size_t cublasLtGetVersion()
{
	TALLY_LOG("cublasLtGetVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtGetVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTGETVERSION;
    
    struct cublasLtGetVersionArg *arg_ptr = (struct cublasLtGetVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (size_t *) dat;
    return *res;
}

size_t cublasLtGetCudartVersion()
{
	TALLY_LOG("cublasLtGetCudartVersion hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtGetCudartVersionArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTGETCUDARTVERSION;
    
    struct cublasLtGetCudartVersionArg *arg_ptr = (struct cublasLtGetCudartVersionArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (size_t *) dat;
    return *res;
}

cublasStatus_t cublasLtGetProperty(libraryPropertyType  type, int*  value)
{
	TALLY_LOG("cublasLtGetProperty hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t  lightHandle, cublasLtMatrixTransformDesc_t  transformDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  beta, const void*  B, cublasLtMatrixLayout_t  Bdesc, void*  C, cublasLtMatrixLayout_t  Cdesc, cudaStream_t  stream)
{
	TALLY_LOG("cublasLtMatrixTransform hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t  matLayout, size_t  size, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)
{
	TALLY_LOG("cublasLtMatrixLayoutInit_internal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*  matLayout, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)
{
	TALLY_LOG("cublasLtMatrixLayoutCreate hooked");

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
	if (matLayout) { *matLayout = res->matLayout; }
	return res->err;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t  matLayout)
{
	TALLY_LOG("cublasLtMatrixLayoutDestroy hooked");

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
	TALLY_LOG("cublasLtMatrixLayoutGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t  matmulDesc, size_t  size, cublasComputeType_t  computeType, cudaDataType_t  scaleType)
{
	TALLY_LOG("cublasLtMatmulDescInit_internal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*  matmulDesc, cublasComputeType_t  computeType, cudaDataType_t  scaleType)
{
	TALLY_LOG("cublasLtMatmulDescCreate hooked");

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
	if (matmulDesc) { *matmulDesc = res->matmulDesc; }
	return res->err;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t  matmulDesc)
{
	TALLY_LOG("cublasLtMatmulDescDestroy hooked");

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
	TALLY_LOG("cublasLtMatmulDescGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixTransformDescInit_internal(cublasLtMatrixTransformDesc_t  transformDesc, size_t  size, cudaDataType  scaleType)
{
	TALLY_LOG("cublasLtMatrixTransformDescInit_internal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t*  transformDesc, cudaDataType  scaleType)
{
	TALLY_LOG("cublasLtMatrixTransformDescCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t  transformDesc)
{
	TALLY_LOG("cublasLtMatrixTransformDescDestroy hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatrixTransformDescSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_LOG("cublasLtMatrixTransformDescGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t  pref, size_t  size)
{
	TALLY_LOG("cublasLtMatmulPreferenceInit_internal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*  pref)
{
	TALLY_LOG("cublasLtMatmulPreferenceCreate hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtMatmulPreferenceCreateArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCECREATE;
    
    struct cublasLtMatmulPreferenceCreateArg *arg_ptr = (struct cublasLtMatmulPreferenceCreateArg *)(msg + sizeof(CUDA_API_ENUM));
	arg_ptr->pref = pref;
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;
	auto res = (cublasLtMatmulPreferenceCreateResponse *) dat;
	if (pref) { *pref = res->pref; }
	return res->err;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t  pref)
{
	TALLY_LOG("cublasLtMatmulPreferenceDestroy hooked");

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
	TALLY_LOG("cublasLtMatmulPreferenceGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  requestedAlgoCount, int  algoIdsArray[], int*  returnAlgoCount)
{
	TALLY_LOG("cublasLtMatmulAlgoGetIds hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  algoId, cublasLtMatmulAlgo_t*  algo)
{
	TALLY_LOG("cublasLtMatmulAlgoInit hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulHeuristicResult_t*  result)
{
	TALLY_LOG("cublasLtMatmulAlgoCheck hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoCapAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_LOG("cublasLtMatmulAlgoCapGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_LOG("cublasLtMatmulAlgoConfigSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_LOG("cublasLtMatmulAlgoConfigGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t  callback)
{
	TALLY_LOG("cublasLtLoggerSetCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtLoggerSetFile(FILE*  file)
{
	TALLY_LOG("cublasLtLoggerSetFile hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtLoggerOpenFile(const char*  logFile)
{
	TALLY_LOG("cublasLtLoggerOpenFile hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtLoggerSetLevel(int  level)
{
	TALLY_LOG("cublasLtLoggerSetLevel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtLoggerSetMask(int  mask)
{
	TALLY_LOG("cublasLtLoggerSetMask hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtLoggerForceDisable()
{
	TALLY_LOG("cublasLtLoggerForceDisable hooked");

    uint32_t msg_len =  sizeof(CUDA_API_ENUM) + sizeof(struct cublasLtLoggerForceDisableArg);

    uint8_t *msg = (uint8_t *) std::malloc(msg_len);
    MessageHeader_t *msg_header = (MessageHeader_t *) msg;
    msg_header->api_id = CUDA_API_ENUM::CUBLASLTLOGGERFORCEDISABLE;
    
    struct cublasLtLoggerForceDisableArg *arg_ptr = (struct cublasLtLoggerForceDisableArg *)(msg + sizeof(CUDA_API_ENUM));
	CLIENT_SEND_MSG_AND_FREE;
	CLIENT_RECV_MSG;

    auto res = (cublasStatus_t *) dat;
    return *res;
}



}

