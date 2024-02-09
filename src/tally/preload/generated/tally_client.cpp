
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

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>
#include <nccl.h>
#include <curand.h>

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
	TALLY_SPD_LOG("cuGetErrorString hooked");
#if defined(RUN_LOCALLY)
	return lcuGetErrorString(error, pStr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGetErrorName(CUresult  error, const char ** pStr)
{
	TALLY_SPD_LOG("cuGetErrorName hooked");
#if defined(RUN_LOCALLY)
	return lcuGetErrorName(error, pStr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuInit(unsigned int  Flags)
{
	TALLY_SPD_LOG("cuInit hooked");
#if defined(RUN_LOCALLY)
	return lcuInit(Flags);
#else
	return (CUresult) 0;
#endif
}

CUresult cuDriverGetVersion(int * driverVersion)
{
	TALLY_SPD_LOG("cuDriverGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDriverGetVersion(driverVersion);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDriverGetVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDRIVERGETVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDriverGetVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->driverVersion = driverVersion;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDriverGetVersionResponse*>(responsePayload);
			if (driverVersion) { *driverVersion = response->driverVersion; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDriverGetVersion);
	return err;
}

CUresult cuDeviceGet(CUdevice * device, int  ordinal)
{
	TALLY_SPD_LOG("cuDeviceGet hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGet(device, ordinal);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGET;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->device = device;
			request->ordinal = ordinal;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetResponse*>(responsePayload);
			if (device) { *device = response->device; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGet);
	return err;
}

CUresult cuDeviceGetCount(int * count)
{
	TALLY_SPD_LOG("cuDeviceGetCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetCount(count);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetCountArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETCOUNT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetCountArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->count = count;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetCountResponse*>(responsePayload);
			if (count) { *count = response->count; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetCount);
	return err;
}

CUresult cuDeviceGetUuid(CUuuid * uuid, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetUuid hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetUuid(uuid, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetUuidArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETUUID;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetUuidArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->uuid = uuid;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetUuidResponse*>(responsePayload);
			if (uuid) { *uuid = response->uuid; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetUuid);
	return err;
}

CUresult cuDeviceGetUuid_v2(CUuuid * uuid, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetUuid_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetUuid_v2(uuid, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetUuid_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETUUID_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetUuid_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->uuid = uuid;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetUuid_v2Response*>(responsePayload);
			if (uuid) { *uuid = response->uuid; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetUuid_v2);
	return err;
}

CUresult cuDeviceGetLuid(char * luid, unsigned int * deviceNodeMask, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetLuid hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetLuid(luid, deviceNodeMask, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetLuidArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETLUID;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetLuidArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->luid = luid;
			request->deviceNodeMask = deviceNodeMask;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetLuidResponse*>(responsePayload);
			if (luid) { *luid = response->luid; }
			if (deviceNodeMask) { *deviceNodeMask = response->deviceNodeMask; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetLuid);
	return err;
}

CUresult cuDeviceTotalMem_v2(size_t * bytes, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceTotalMem_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceTotalMem_v2(bytes, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceTotalMem_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICETOTALMEM_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceTotalMem_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->bytes = bytes;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceTotalMem_v2Response*>(responsePayload);
			if (bytes) { *bytes = response->bytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceTotalMem_v2);
	return err;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, CUarray_format  format, unsigned  numChannels, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetTexture1DLinearMaxWidth hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetTexture1DLinearMaxWidthArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETTEXTURE1DLINEARMAXWIDTH;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetTexture1DLinearMaxWidthArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->maxWidthInElements = maxWidthInElements;
			request->format = format;
			request->numChannels = numChannels;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetTexture1DLinearMaxWidthResponse*>(responsePayload);
			if (maxWidthInElements) { *maxWidthInElements = response->maxWidthInElements; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetTexture1DLinearMaxWidth);
	return err;
}

CUresult cuDeviceGetAttribute(int * pi, CUdevice_attribute  attrib, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetAttribute(pi, attrib, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetAttributeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pi = pi;
			request->attrib = attrib;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetAttributeResponse*>(responsePayload);
			if (pi) { *pi = response->pi; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetAttribute);
	return err;
}

CUresult cuDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, CUdevice  dev, int  flags)
{
	TALLY_SPD_LOG("cuDeviceGetNvSciSyncAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceSetMemPool(CUdevice  dev, CUmemoryPool  pool)
{
	TALLY_SPD_LOG("cuDeviceSetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceSetMemPool(dev, pool);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceSetMemPoolArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICESETMEMPOOL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceSetMemPoolArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dev = dev;
			request->pool = pool;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceSetMemPool);
	return err;
}

CUresult cuDeviceGetMemPool(CUmemoryPool * pool, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetMemPool(pool, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetMemPoolArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETMEMPOOL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetMemPoolArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pool = pool;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetMemPoolResponse*>(responsePayload);
			if (pool) { *pool = response->pool; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetMemPool);
	return err;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool * pool_out, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetDefaultMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetDefaultMemPool(pool_out, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetDefaultMemPoolArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETDEFAULTMEMPOOL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetDefaultMemPoolArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pool_out = pool_out;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetDefaultMemPoolResponse*>(responsePayload);
			if (pool_out) { *pool_out = response->pool_out; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetDefaultMemPool);
	return err;
}

CUresult cuDeviceGetExecAffinitySupport(int * pi, CUexecAffinityType  type, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetExecAffinitySupport hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetExecAffinitySupport(pi, type, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetExecAffinitySupportArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETEXECAFFINITYSUPPORT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetExecAffinitySupportArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pi = pi;
			request->type = type;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetExecAffinitySupportResponse*>(responsePayload);
			if (pi) { *pi = response->pi; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetExecAffinitySupport);
	return err;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget  target, CUflushGPUDirectRDMAWritesScope  scope)
{
	TALLY_SPD_LOG("cuFlushGPUDirectRDMAWrites hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuFlushGPUDirectRDMAWrites(target, scope);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuFlushGPUDirectRDMAWritesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUFLUSHGPUDIRECTRDMAWRITES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuFlushGPUDirectRDMAWritesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->target = target;
			request->scope = scope;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuFlushGPUDirectRDMAWrites);
	return err;
}

CUresult cuDeviceGetProperties(CUdevprop * prop, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetProperties hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetProperties(prop, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetPropertiesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETPROPERTIES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetPropertiesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->prop = prop;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetPropertiesResponse*>(responsePayload);
			if (prop) { *prop = response->prop; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetProperties);
	return err;
}

CUresult cuDeviceComputeCapability(int * major, int * minor, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceComputeCapability hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceComputeCapability(major, minor, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceComputeCapabilityArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICECOMPUTECAPABILITY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceComputeCapabilityArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->major = major;
			request->minor = minor;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceComputeCapabilityResponse*>(responsePayload);
			if (major) { *major = response->major; }
			if (minor) { *minor = response->minor; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceComputeCapability);
	return err;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext * pctx, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDevicePrimaryCtxRetain hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxRetain(pctx, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxRetainArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXRETAIN;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDevicePrimaryCtxRetainArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pctx = pctx;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDevicePrimaryCtxRetainResponse*>(responsePayload);
			if (pctx) { *pctx = response->pctx; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDevicePrimaryCtxRetain);
	return err;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice  dev)
{
	TALLY_SPD_LOG("cuDevicePrimaryCtxRelease_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxRelease_v2(dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxRelease_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXRELEASE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDevicePrimaryCtxRelease_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDevicePrimaryCtxRelease_v2);
	return err;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice  dev)
{
	TALLY_SPD_LOG("cuDevicePrimaryCtxReset_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxReset_v2(dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxReset_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXRESET_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDevicePrimaryCtxReset_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDevicePrimaryCtxReset_v2);
	return err;
}

CUresult cuCtxCreate_v3(CUcontext * pctx, CUexecAffinityParam * paramsArray, int  numParams, unsigned int  flags, CUdevice  dev)
{
	TALLY_SPD_LOG("cuCtxCreate_v3 hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCtxDestroy_v2(CUcontext  ctx)
{
	TALLY_SPD_LOG("cuCtxDestroy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxDestroy_v2(ctx);
#else
	return (CUresult) 0;
#endif
}

CUresult cuCtxPushCurrent_v2(CUcontext  ctx)
{
	TALLY_SPD_LOG("cuCtxPushCurrent_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxPushCurrent_v2(ctx);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxPushCurrent_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXPUSHCURRENT_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxPushCurrent_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->ctx = ctx;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxPushCurrent_v2);
	return err;
}

CUresult cuCtxPopCurrent_v2(CUcontext * pctx)
{
	TALLY_SPD_LOG("cuCtxPopCurrent_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxPopCurrent_v2(pctx);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxPopCurrent_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXPOPCURRENT_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxPopCurrent_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pctx = pctx;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxPopCurrent_v2Response*>(responsePayload);
			if (pctx) { *pctx = response->pctx; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxPopCurrent_v2);
	return err;
}

CUresult cuCtxSetCurrent(CUcontext  ctx)
{
	TALLY_SPD_LOG("cuCtxSetCurrent hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxSetCurrent(ctx);
#else
	return (CUresult) 0;
#endif
}

CUresult cuCtxGetFlags(unsigned int * flags)
{
	TALLY_SPD_LOG("cuCtxGetFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetFlags(flags);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetFlagsResponse*>(responsePayload);
			if (flags) { *flags = response->flags; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetFlags);
	return err;
}

CUresult cuCtxSetFlags(unsigned int  flags)
{
	TALLY_SPD_LOG("cuCtxSetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxSetFlags(flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCtxGetId(CUcontext  ctx, unsigned long long * ctxId)
{
	TALLY_SPD_LOG("cuCtxGetId hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxGetId(ctx, ctxId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCtxSetLimit(CUlimit  limit, size_t  value)
{
	TALLY_SPD_LOG("cuCtxSetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetLimit(limit, value);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetLimitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXSETLIMIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxSetLimitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->limit = limit;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxSetLimit);
	return err;
}

CUresult cuCtxGetLimit(size_t * pvalue, CUlimit  limit)
{
	TALLY_SPD_LOG("cuCtxGetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetLimit(pvalue, limit);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetLimitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETLIMIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetLimitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pvalue = pvalue;
			request->limit = limit;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetLimitResponse*>(responsePayload);
			if (pvalue) { *pvalue = response->pvalue; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetLimit);
	return err;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache * pconfig)
{
	TALLY_SPD_LOG("cuCtxGetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetCacheConfig(pconfig);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetCacheConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pconfig = pconfig;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetCacheConfigResponse*>(responsePayload);
			if (pconfig) { *pconfig = response->pconfig; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetCacheConfig);
	return err;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache  config)
{
	TALLY_SPD_LOG("cuCtxSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetCacheConfig(config);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetCacheConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXSETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxSetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->config = config;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxSetCacheConfig);
	return err;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig * pConfig)
{
	TALLY_SPD_LOG("cuCtxGetSharedMemConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetSharedMemConfig(pConfig);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetSharedMemConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETSHAREDMEMCONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetSharedMemConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pConfig = pConfig;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetSharedMemConfigResponse*>(responsePayload);
			if (pConfig) { *pConfig = response->pConfig; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetSharedMemConfig);
	return err;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig  config)
{
	TALLY_SPD_LOG("cuCtxSetSharedMemConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetSharedMemConfig(config);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetSharedMemConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXSETSHAREDMEMCONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxSetSharedMemConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->config = config;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxSetSharedMemConfig);
	return err;
}

CUresult cuCtxGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	TALLY_SPD_LOG("cuCtxGetStreamPriorityRange hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetStreamPriorityRangeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETSTREAMPRIORITYRANGE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetStreamPriorityRangeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->leastPriority = leastPriority;
			request->greatestPriority = greatestPriority;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetStreamPriorityRangeResponse*>(responsePayload);
			if (leastPriority) { *leastPriority = response->leastPriority; }
			if (greatestPriority) { *greatestPriority = response->greatestPriority; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetStreamPriorityRange);
	return err;
}

CUresult cuCtxResetPersistingL2Cache()
{
	TALLY_SPD_LOG("cuCtxResetPersistingL2Cache hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxResetPersistingL2Cache();
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxResetPersistingL2CacheArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXRESETPERSISTINGL2CACHE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxResetPersistingL2CacheArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxResetPersistingL2Cache);
	return err;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam * pExecAffinity, CUexecAffinityType  type)
{
	TALLY_SPD_LOG("cuCtxGetExecAffinity hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetExecAffinity(pExecAffinity, type);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetExecAffinityArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETEXECAFFINITY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetExecAffinityArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pExecAffinity = pExecAffinity;
			request->type = type;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetExecAffinityResponse*>(responsePayload);
			if (pExecAffinity) { *pExecAffinity = response->pExecAffinity; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetExecAffinity);
	return err;
}

CUresult cuCtxAttach(CUcontext * pctx, unsigned int  flags)
{
	TALLY_SPD_LOG("cuCtxAttach hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxAttach(pctx, flags);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxAttachArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXATTACH;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxAttachArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pctx = pctx;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxAttachResponse*>(responsePayload);
			if (pctx) { *pctx = response->pctx; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxAttach);
	return err;
}

CUresult cuCtxDetach(CUcontext  ctx)
{
	TALLY_SPD_LOG("cuCtxDetach hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxDetach(ctx);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxDetachArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXDETACH;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxDetachArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->ctx = ctx;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxDetach);
	return err;
}

CUresult cuModuleLoad(CUmodule * module, const char * fname)
{
	TALLY_SPD_LOG("cuModuleLoad hooked");
#if defined(RUN_LOCALLY)
	return lcuModuleLoad(module, fname);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode * mode)
{
	TALLY_SPD_LOG("cuModuleGetLoadingMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuModuleGetLoadingMode(mode);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuModuleGetLoadingModeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULEGETLOADINGMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleGetLoadingModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuModuleGetLoadingModeResponse*>(responsePayload);
			if (mode) { *mode = response->mode; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuModuleGetLoadingMode);
	return err;
}

CUresult cuLinkCreate_v2(unsigned int  numOptions, CUjit_option * options, void ** optionValues, CUlinkState * stateOut)
{
	TALLY_SPD_LOG("cuLinkCreate_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuLinkCreate_v2(numOptions, options, optionValues, stateOut);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLinkAddData_v2(CUlinkState  state, CUjitInputType  type, void * data, size_t  size, const char * name, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	TALLY_SPD_LOG("cuLinkAddData_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLinkAddFile_v2(CUlinkState  state, CUjitInputType  type, const char * path, unsigned int  numOptions, CUjit_option * options, void ** optionValues)
{
	TALLY_SPD_LOG("cuLinkAddFile_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLinkComplete(CUlinkState  state, void ** cubinOut, size_t * sizeOut)
{
	TALLY_SPD_LOG("cuLinkComplete hooked");
#if defined(RUN_LOCALLY)
	return lcuLinkComplete(state, cubinOut, sizeOut);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLinkDestroy(CUlinkState  state)
{
	TALLY_SPD_LOG("cuLinkDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuLinkDestroy(state);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuModuleGetTexRef(CUtexref * pTexRef, CUmodule  hmod, const char * name)
{
	TALLY_SPD_LOG("cuModuleGetTexRef hooked");
#if defined(RUN_LOCALLY)
	return lcuModuleGetTexRef(pTexRef, hmod, name);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuModuleGetSurfRef(CUsurfref * pSurfRef, CUmodule  hmod, const char * name)
{
	TALLY_SPD_LOG("cuModuleGetSurfRef hooked");
#if defined(RUN_LOCALLY)
	return lcuModuleGetSurfRef(pSurfRef, hmod, name);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryLoadData(CUlibrary * library, const void * code, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void**  libraryOptionValues, unsigned int  numLibraryOptions)
{
	TALLY_SPD_LOG("cuLibraryLoadData hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryLoadFromFile(CUlibrary * library, const char * fileName, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int  numLibraryOptions)
{
	TALLY_SPD_LOG("cuLibraryLoadFromFile hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryUnload(CUlibrary  library)
{
	TALLY_SPD_LOG("cuLibraryUnload hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryUnload(library);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryGetKernel(CUkernel * pKernel, CUlibrary  library, const char * name)
{
	TALLY_SPD_LOG("cuLibraryGetKernel hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryGetKernel(pKernel, library, name);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryGetModule(CUmodule * pMod, CUlibrary  library)
{
	TALLY_SPD_LOG("cuLibraryGetModule hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryGetModule(pMod, library);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuKernelGetFunction(CUfunction * pFunc, CUkernel  kernel)
{
	TALLY_SPD_LOG("cuKernelGetFunction hooked");
#if defined(RUN_LOCALLY)
	return lcuKernelGetFunction(pFunc, kernel);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name)
{
	TALLY_SPD_LOG("cuLibraryGetGlobal hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryGetGlobal(dptr, bytes, library, name);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryGetManaged(CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name)
{
	TALLY_SPD_LOG("cuLibraryGetManaged hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryGetManaged(dptr, bytes, library, name);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLibraryGetUnifiedFunction(void ** fptr, CUlibrary  library, const char * symbol)
{
	TALLY_SPD_LOG("cuLibraryGetUnifiedFunction hooked");
#if defined(RUN_LOCALLY)
	return lcuLibraryGetUnifiedFunction(fptr, library, symbol);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuKernelGetAttribute(int * pi, CUfunction_attribute  attrib, CUkernel  kernel, CUdevice  dev)
{
	TALLY_SPD_LOG("cuKernelGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuKernelGetAttribute(pi, attrib, kernel, dev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuKernelSetAttribute(CUfunction_attribute  attrib, int  val, CUkernel  kernel, CUdevice  dev)
{
	TALLY_SPD_LOG("cuKernelSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuKernelSetAttribute(attrib, val, kernel, dev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuKernelSetCacheConfig(CUkernel  kernel, CUfunc_cache  config, CUdevice  dev)
{
	TALLY_SPD_LOG("cuKernelSetCacheConfig hooked");
#if defined(RUN_LOCALLY)
	return lcuKernelSetCacheConfig(kernel, config, dev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemGetInfo_v2(size_t * free, size_t * total)
{
	TALLY_SPD_LOG("cuMemGetInfo_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemGetInfo_v2(free, total);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemGetInfo_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMGETINFO_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemGetInfo_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->free = free;
			request->total = total;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuMemGetInfo_v2Response*>(responsePayload);
			if (free) { *free = response->free; }
			if (total) { *total = response->total; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemGetInfo_v2);
	return err;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr * dptr, size_t * pPitch, size_t  WidthInBytes, size_t  Height, unsigned int  ElementSizeBytes)
{
	TALLY_SPD_LOG("cuMemAllocPitch_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr * pbase, size_t * psize, CUdeviceptr  dptr)
{
	TALLY_SPD_LOG("cuMemGetAddressRange_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemGetAddressRange_v2(pbase, psize, dptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAllocHost_v2(void ** pp, size_t  bytesize)
{
	TALLY_SPD_LOG("cuMemAllocHost_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAllocHost_v2(pp, bytesize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemFreeHost(void * p)
{
	TALLY_SPD_LOG("cuMemFreeHost hooked");
#if defined(RUN_LOCALLY)
	return lcuMemFreeHost(p);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr * pdptr, void * p, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuMemHostGetDevicePointer_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemHostGetDevicePointer_v2(pdptr, p, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemHostGetFlags(unsigned int * pFlags, void * p)
{
	TALLY_SPD_LOG("cuMemHostGetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuMemHostGetFlags(pFlags, p);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAllocManaged(CUdeviceptr * dptr, size_t  bytesize, unsigned int  flags)
{
	TALLY_SPD_LOG("cuMemAllocManaged hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAllocManaged(dptr, bytesize, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceGetByPCIBusId(CUdevice * dev, const char * pciBusId)
{
	TALLY_SPD_LOG("cuDeviceGetByPCIBusId hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceGetByPCIBusId(dev, pciBusId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceGetPCIBusId(char * pciBusId, int  len, CUdevice  dev)
{
	TALLY_SPD_LOG("cuDeviceGetPCIBusId hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetPCIBusId(pciBusId, len, dev);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetPCIBusIdArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEGETPCIBUSID;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDeviceGetPCIBusIdArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pciBusId = pciBusId;
			request->len = len;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDeviceGetPCIBusIdResponse*>(responsePayload);
			if (pciBusId) { *pciBusId = response->pciBusId; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDeviceGetPCIBusId);
	return err;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle * pHandle, CUevent  event)
{
	TALLY_SPD_LOG("cuIpcGetEventHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuIpcGetEventHandle(pHandle, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuIpcOpenEventHandle(CUevent * phEvent, CUipcEventHandle  handle)
{
	TALLY_SPD_LOG("cuIpcOpenEventHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuIpcOpenEventHandle(phEvent, handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuIpcGetMemHandle(CUipcMemHandle * pHandle, CUdeviceptr  dptr)
{
	TALLY_SPD_LOG("cuIpcGetMemHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuIpcGetMemHandle(pHandle, dptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr * pdptr, CUipcMemHandle  handle, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuIpcOpenMemHandle_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuIpcOpenMemHandle_v2(pdptr, handle, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuIpcCloseMemHandle(CUdeviceptr  dptr)
{
	TALLY_SPD_LOG("cuIpcCloseMemHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuIpcCloseMemHandle(dptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemHostRegister_v2(void * p, size_t  bytesize, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuMemHostRegister_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemHostRegister_v2(p, bytesize, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemHostUnregister(void * p)
{
	TALLY_SPD_LOG("cuMemHostUnregister hooked");
#if defined(RUN_LOCALLY)
	return lcuMemHostUnregister(p);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyPeer(CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyPeer hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr  dstDevice, const void * srcHost, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyHtoD_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyDtoH_v2(void * dstHost, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyDtoH_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr  dstDevice, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyDtoD_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyDtoA_v2(CUarray  dstArray, size_t  dstOffset, CUdeviceptr  srcDevice, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyDtoA_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr  dstDevice, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyAtoD_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyHtoA_v2(CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyHtoA_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyAtoH_v2(void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyAtoH_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyAtoA_v2(CUarray  dstArray, size_t  dstOffset, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount)
{
	TALLY_SPD_LOG("cuMemcpyAtoA_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D * pCopy)
{
	TALLY_SPD_LOG("cuMemcpy2D_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy2D_v2(pCopy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * pCopy)
{
	TALLY_SPD_LOG("cuMemcpy2DUnaligned_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy2DUnaligned_v2(pCopy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D * pCopy)
{
	TALLY_SPD_LOG("cuMemcpy3D_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy3D_v2(pCopy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER * pCopy)
{
	TALLY_SPD_LOG("cuMemcpy3DPeer hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy3DPeer(pCopy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyPeerAsync(CUdeviceptr  dstDevice, CUcontext  dstContext, CUdeviceptr  srcDevice, CUcontext  srcContext, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyPeerAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyHtoAAsync_v2(CUarray  dstArray, size_t  dstOffset, const void * srcHost, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyHtoAAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpyAtoHAsync_v2(void * dstHost, CUarray  srcArray, size_t  srcOffset, size_t  ByteCount, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpyAtoHAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * pCopy, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpy2DAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy2DAsync_v2(pCopy, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * pCopy, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpy3DAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy3DAsync_v2(pCopy, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER * pCopy, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemcpy3DPeerAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuMemcpy3DPeerAsync(pCopy, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD16_v2(CUdeviceptr  dstDevice, unsigned short  us, size_t  N)
{
	TALLY_SPD_LOG("cuMemsetD16_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD16_v2(dstDevice, us, N);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD2D8_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height)
{
	TALLY_SPD_LOG("cuMemsetD2D8_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD2D16_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height)
{
	TALLY_SPD_LOG("cuMemsetD2D16_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD2D32_v2(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height)
{
	TALLY_SPD_LOG("cuMemsetD2D32_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD8Async(CUdeviceptr  dstDevice, unsigned char  uc, size_t  N, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemsetD8Async hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD8Async(dstDevice, uc, N, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD16Async(CUdeviceptr  dstDevice, unsigned short  us, size_t  N, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemsetD16Async hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD16Async(dstDevice, us, N, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD2D8Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned char  uc, size_t  Width, size_t  Height, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemsetD2D8Async hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD2D16Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned short  us, size_t  Width, size_t  Height, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemsetD2D16Async hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemsetD2D32Async(CUdeviceptr  dstDevice, size_t  dstPitch, unsigned int  ui, size_t  Width, size_t  Height, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemsetD2D32Async hooked");
#if defined(RUN_LOCALLY)
	return lcuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArrayCreate_v2(CUarray * pHandle, const CUDA_ARRAY_DESCRIPTOR * pAllocateArray)
{
	TALLY_SPD_LOG("cuArrayCreate_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuArrayCreate_v2(pHandle, pAllocateArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)
{
	TALLY_SPD_LOG("cuArrayGetDescriptor_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuArrayGetDescriptor_v2(pArrayDescriptor, hArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUarray  array)
{
	TALLY_SPD_LOG("cuArrayGetSparseProperties hooked");
#if defined(RUN_LOCALLY)
	return lcuArrayGetSparseProperties(sparseProperties, array);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES * sparseProperties, CUmipmappedArray  mipmap)
{
	TALLY_SPD_LOG("cuMipmappedArrayGetSparseProperties hooked");
#if defined(RUN_LOCALLY)
	return lcuMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUarray  array, CUdevice  device)
{
	TALLY_SPD_LOG("cuArrayGetMemoryRequirements hooked");
#if defined(RUN_LOCALLY)
	return lcuArrayGetMemoryRequirements(memoryRequirements, array, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUmipmappedArray  mipmap, CUdevice  device)
{
	TALLY_SPD_LOG("cuMipmappedArrayGetMemoryRequirements hooked");
#if defined(RUN_LOCALLY)
	return lcuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArrayGetPlane(CUarray * pPlaneArray, CUarray  hArray, unsigned int  planeIdx)
{
	TALLY_SPD_LOG("cuArrayGetPlane hooked");
#if defined(RUN_LOCALLY)
	return lcuArrayGetPlane(pPlaneArray, hArray, planeIdx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArrayDestroy(CUarray  hArray)
{
	TALLY_SPD_LOG("cuArrayDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuArrayDestroy(hArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArray3DCreate_v2(CUarray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pAllocateArray)
{
	TALLY_SPD_LOG("cuArray3DCreate_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuArray3DCreate_v2(pHandle, pAllocateArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * pArrayDescriptor, CUarray  hArray)
{
	TALLY_SPD_LOG("cuArray3DGetDescriptor_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray * pHandle, const CUDA_ARRAY3D_DESCRIPTOR * pMipmappedArrayDesc, unsigned int  numMipmapLevels)
{
	TALLY_SPD_LOG("cuMipmappedArrayCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMipmappedArrayGetLevel(CUarray * pLevelArray, CUmipmappedArray  hMipmappedArray, unsigned int  level)
{
	TALLY_SPD_LOG("cuMipmappedArrayGetLevel hooked");
#if defined(RUN_LOCALLY)
	return lcuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray  hMipmappedArray)
{
	TALLY_SPD_LOG("cuMipmappedArrayDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuMipmappedArrayDestroy(hMipmappedArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemGetHandleForAddressRange(void * handle, CUdeviceptr  dptr, size_t  size, CUmemRangeHandleType  handleType, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemGetHandleForAddressRange hooked");
#if defined(RUN_LOCALLY)
	return lcuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAddressReserve(CUdeviceptr * ptr, size_t  size, size_t  alignment, CUdeviceptr  addr, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemAddressReserve hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAddressReserve(ptr, size, alignment, addr, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAddressFree(CUdeviceptr  ptr, size_t  size)
{
	TALLY_SPD_LOG("cuMemAddressFree hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAddressFree(ptr, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemCreate(CUmemGenericAllocationHandle * handle, size_t  size, const CUmemAllocationProp * prop, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuMemCreate(handle, size, prop, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemRelease(CUmemGenericAllocationHandle  handle)
{
	TALLY_SPD_LOG("cuMemRelease hooked");
#if defined(RUN_LOCALLY)
	return lcuMemRelease(handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemMap(CUdeviceptr  ptr, size_t  size, size_t  offset, CUmemGenericAllocationHandle  handle, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemMap hooked");
#if defined(RUN_LOCALLY)
	return lcuMemMap(ptr, size, offset, handle, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo * mapInfoList, unsigned int  count, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemMapArrayAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuMemMapArrayAsync(mapInfoList, count, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemUnmap(CUdeviceptr  ptr, size_t  size)
{
	TALLY_SPD_LOG("cuMemUnmap hooked");
#if defined(RUN_LOCALLY)
	return lcuMemUnmap(ptr, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemSetAccess(CUdeviceptr  ptr, size_t  size, const CUmemAccessDesc * desc, size_t  count)
{
	TALLY_SPD_LOG("cuMemSetAccess hooked");
#if defined(RUN_LOCALLY)
	return lcuMemSetAccess(ptr, size, desc, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemGetAccess(unsigned long long * flags, const CUmemLocation * location, CUdeviceptr  ptr)
{
	TALLY_SPD_LOG("cuMemGetAccess hooked");
#if defined(RUN_LOCALLY)
	return lcuMemGetAccess(flags, location, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemExportToShareableHandle(void * shareableHandle, CUmemGenericAllocationHandle  handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemExportToShareableHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle * handle, void * osHandle, CUmemAllocationHandleType  shHandleType)
{
	TALLY_SPD_LOG("cuMemImportFromShareableHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuMemImportFromShareableHandle(handle, osHandle, shHandleType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemGetAllocationGranularity(size_t * granularity, const CUmemAllocationProp * prop, CUmemAllocationGranularity_flags  option)
{
	TALLY_SPD_LOG("cuMemGetAllocationGranularity hooked");
#if defined(RUN_LOCALLY)
	return lcuMemGetAllocationGranularity(granularity, prop, option);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp * prop, CUmemGenericAllocationHandle  handle)
{
	TALLY_SPD_LOG("cuMemGetAllocationPropertiesFromHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuMemGetAllocationPropertiesFromHandle(prop, handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle * handle, void * addr)
{
	TALLY_SPD_LOG("cuMemRetainAllocationHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuMemRetainAllocationHandle(handle, addr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemFreeAsync(CUdeviceptr  dptr, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemFreeAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuMemFreeAsync(dptr, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolTrimTo(CUmemoryPool  pool, size_t  minBytesToKeep)
{
	TALLY_SPD_LOG("cuMemPoolTrimTo hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolTrimTo(pool, minBytesToKeep);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolSetAttribute(CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)
{
	TALLY_SPD_LOG("cuMemPoolSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolSetAttribute(pool, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolGetAttribute(CUmemoryPool  pool, CUmemPool_attribute  attr, void * value)
{
	TALLY_SPD_LOG("cuMemPoolGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolGetAttribute(pool, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolSetAccess(CUmemoryPool  pool, const CUmemAccessDesc * map, size_t  count)
{
	TALLY_SPD_LOG("cuMemPoolSetAccess hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolSetAccess(pool, map, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags * flags, CUmemoryPool  memPool, CUmemLocation * location)
{
	TALLY_SPD_LOG("cuMemPoolGetAccess hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolGetAccess(flags, memPool, location);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolCreate(CUmemoryPool * pool, const CUmemPoolProps * poolProps)
{
	TALLY_SPD_LOG("cuMemPoolCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolCreate(pool, poolProps);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolDestroy(CUmemoryPool  pool)
{
	TALLY_SPD_LOG("cuMemPoolDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolDestroy(pool);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr * dptr, size_t  bytesize, CUmemoryPool  pool, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemAllocFromPoolAsync hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemAllocFromPoolAsyncArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMALLOCFROMPOOLASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemAllocFromPoolAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dptr = dptr;
			request->bytesize = bytesize;
			request->pool = pool;
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuMemAllocFromPoolAsyncResponse*>(responsePayload);
			if (dptr) { *dptr = response->dptr; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuMemAllocFromPoolAsync);
	return err;
}

CUresult cuMemPoolExportToShareableHandle(void * handle_out, CUmemoryPool  pool, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemPoolExportToShareableHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool * pool_out, void * handle, CUmemAllocationHandleType  handleType, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMemPoolImportFromShareableHandle hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData * shareData_out, CUdeviceptr  ptr)
{
	TALLY_SPD_LOG("cuMemPoolExportPointer hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolExportPointer(shareData_out, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPoolImportPointer(CUdeviceptr * ptr_out, CUmemoryPool  pool, CUmemPoolPtrExportData * shareData)
{
	TALLY_SPD_LOG("cuMemPoolImportPointer hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPoolImportPointer(ptr_out, pool, shareData);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMulticastCreate(CUmemGenericAllocationHandle * mcHandle, const CUmulticastObjectProp * prop)
{
	TALLY_SPD_LOG("cuMulticastCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuMulticastCreate(mcHandle, prop);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle  mcHandle, CUdevice  dev)
{
	TALLY_SPD_LOG("cuMulticastAddDevice hooked");
#if defined(RUN_LOCALLY)
	return lcuMulticastAddDevice(mcHandle, dev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUmemGenericAllocationHandle  memHandle, size_t  memOffset, size_t  size, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMulticastBindMem hooked");
#if defined(RUN_LOCALLY)
	return lcuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUdeviceptr  memptr, size_t  size, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuMulticastBindAddr hooked");
#if defined(RUN_LOCALLY)
	return lcuMulticastBindAddr(mcHandle, mcOffset, memptr, size, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMulticastUnbind(CUmemGenericAllocationHandle  mcHandle, CUdevice  dev, size_t  mcOffset, size_t  size)
{
	TALLY_SPD_LOG("cuMulticastUnbind hooked");
#if defined(RUN_LOCALLY)
	return lcuMulticastUnbind(mcHandle, dev, mcOffset, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMulticastGetGranularity(size_t * granularity, const CUmulticastObjectProp * prop, CUmulticastGranularity_flags  option)
{
	TALLY_SPD_LOG("cuMulticastGetGranularity hooked");
#if defined(RUN_LOCALLY)
	return lcuMulticastGetGranularity(granularity, prop, option);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPrefetchAsync(CUdeviceptr  devPtr, size_t  count, CUdevice  dstDevice, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemPrefetchAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemPrefetchAsync_v2(CUdeviceptr  devPtr, size_t  count, CUmemLocation  location, unsigned int  flags, CUstream  hStream)
{
	TALLY_SPD_LOG("cuMemPrefetchAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAdvise(CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUdevice  device)
{
	TALLY_SPD_LOG("cuMemAdvise hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAdvise(devPtr, count, advice, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemAdvise_v2(CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUmemLocation  location)
{
	TALLY_SPD_LOG("cuMemAdvise_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuMemAdvise_v2(devPtr, count, advice, location);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemRangeGetAttribute(void * data, size_t  dataSize, CUmem_range_attribute  attribute, CUdeviceptr  devPtr, size_t  count)
{
	TALLY_SPD_LOG("cuMemRangeGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuMemRangeGetAttributes(void ** data, size_t * dataSizes, CUmem_range_attribute * attributes, size_t  numAttributes, CUdeviceptr  devPtr, size_t  count)
{
	TALLY_SPD_LOG("cuMemRangeGetAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuPointerSetAttribute(const void * value, CUpointer_attribute  attribute, CUdeviceptr  ptr)
{
	TALLY_SPD_LOG("cuPointerSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuPointerSetAttribute(value, attribute, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuPointerGetAttributes(unsigned int  numAttributes, CUpointer_attribute * attributes, void ** data, CUdeviceptr  ptr)
{
	TALLY_SPD_LOG("cuPointerGetAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcuPointerGetAttributes(numAttributes, attributes, data, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamGetPriority(CUstream  hStream, int * priority)
{
	TALLY_SPD_LOG("cuStreamGetPriority hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamGetPriority(hStream, priority);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamGetFlags(CUstream  hStream, unsigned int * flags)
{
	TALLY_SPD_LOG("cuStreamGetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamGetFlags(hStream, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamGetId(CUstream  hStream, unsigned long long * streamId)
{
	TALLY_SPD_LOG("cuStreamGetId hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamGetId(hStream, streamId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamGetCtx(CUstream  hStream, CUcontext * pctx)
{
	TALLY_SPD_LOG("cuStreamGetCtx hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamGetCtx(hStream, pctx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamWaitEvent(CUstream  hStream, CUevent  hEvent, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuStreamWaitEvent hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamWaitEvent(hStream, hEvent, Flags);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamWaitEventArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMWAITEVENT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamWaitEventArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;
			request->hEvent = hEvent;
			request->Flags = Flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamWaitEvent);
	return err;
}

CUresult cuStreamAddCallback(CUstream  hStream, CUstreamCallback  callback, void * userData, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamAddCallback hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamAddCallback(hStream, callback, userData, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamBeginCapture_v2(CUstream  hStream, CUstreamCaptureMode  mode)
{
	TALLY_SPD_LOG("cuStreamBeginCapture_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamBeginCapture_v2(hStream, mode);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamBeginCapture_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMBEGINCAPTURE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamBeginCapture_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamBeginCapture_v2);
	return err;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode * mode)
{
	TALLY_SPD_LOG("cuThreadExchangeStreamCaptureMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuThreadExchangeStreamCaptureMode(mode);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuThreadExchangeStreamCaptureModeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUTHREADEXCHANGESTREAMCAPTUREMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuThreadExchangeStreamCaptureModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuThreadExchangeStreamCaptureModeResponse*>(responsePayload);
			if (mode) { *mode = response->mode; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuThreadExchangeStreamCaptureMode);
	return err;
}

CUresult cuStreamIsCapturing(CUstream  hStream, CUstreamCaptureStatus * captureStatus)
{
	TALLY_SPD_LOG("cuStreamIsCapturing hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuStreamIsCapturing(hStream, captureStatus);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuStreamIsCapturingArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUSTREAMISCAPTURING;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuStreamIsCapturingArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;
			request->captureStatus = captureStatus;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuStreamIsCapturingResponse*>(responsePayload);
			if (captureStatus) { *captureStatus = response->captureStatus; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuStreamIsCapturing);
	return err;
}

CUresult cuStreamGetCaptureInfo_v2(CUstream  hStream, CUstreamCaptureStatus * captureStatus_out, cuuint64_t * id_out, CUgraph * graph_out, const CUgraphNode ** dependencies_out, size_t * numDependencies_out)
{
	TALLY_SPD_LOG("cuStreamGetCaptureInfo_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamUpdateCaptureDependencies(CUstream  hStream, CUgraphNode * dependencies, size_t  numDependencies, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamUpdateCaptureDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamAttachMemAsync(CUstream  hStream, CUdeviceptr  dptr, size_t  length, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamAttachMemAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamAttachMemAsync(hStream, dptr, length, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamQuery(CUstream  hStream)
{
	TALLY_SPD_LOG("cuStreamQuery hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamQuery(hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamDestroy_v2(CUstream  hStream)
{
	TALLY_SPD_LOG("cuStreamDestroy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamDestroy_v2(hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamCopyAttributes(CUstream  dst, CUstream  src)
{
	TALLY_SPD_LOG("cuStreamCopyAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamCopyAttributes(dst, src);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamGetAttribute(CUstream  hStream, CUstreamAttrID  attr, CUstreamAttrValue * value_out)
{
	TALLY_SPD_LOG("cuStreamGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamGetAttribute(hStream, attr, value_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamSetAttribute(CUstream  hStream, CUstreamAttrID  attr, const CUstreamAttrValue * value)
{
	TALLY_SPD_LOG("cuStreamSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamSetAttribute(hStream, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuEventCreate(CUevent * phEvent, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuEventCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuEventCreate(phEvent, Flags);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuEventCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUEVENTCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuEventCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->phEvent = phEvent;
			request->Flags = Flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuEventCreateResponse*>(responsePayload);
			if (phEvent) { *phEvent = response->phEvent; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuEventCreate);
	return err;
}

CUresult cuEventRecord(CUevent  hEvent, CUstream  hStream)
{
	TALLY_SPD_LOG("cuEventRecord hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuEventRecord(hEvent, hStream);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuEventRecordArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUEVENTRECORD;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuEventRecordArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hEvent = hEvent;
			request->hStream = hStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuEventRecord);
	return err;
}

CUresult cuEventRecordWithFlags(CUevent  hEvent, CUstream  hStream, unsigned int  flags)
{
	TALLY_SPD_LOG("cuEventRecordWithFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuEventRecordWithFlags(hEvent, hStream, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuEventQuery(CUevent  hEvent)
{
	TALLY_SPD_LOG("cuEventQuery hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuEventQuery(hEvent);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuEventQueryArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUEVENTQUERY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuEventQueryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hEvent = hEvent;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuEventQuery);
	return err;
}

CUresult cuEventSynchronize(CUevent  hEvent)
{
	TALLY_SPD_LOG("cuEventSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuEventSynchronize(hEvent);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuEventSynchronizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUEVENTSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuEventSynchronizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hEvent = hEvent;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuEventSynchronize);
	return err;
}

CUresult cuEventDestroy_v2(CUevent  hEvent)
{
	TALLY_SPD_LOG("cuEventDestroy_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuEventDestroy_v2(hEvent);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuEventDestroy_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUEVENTDESTROY_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuEventDestroy_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hEvent = hEvent;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuEventDestroy_v2);
	return err;
}

CUresult cuEventElapsedTime(float * pMilliseconds, CUevent  hStart, CUevent  hEnd)
{
	TALLY_SPD_LOG("cuEventElapsedTime hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuEventElapsedTime(pMilliseconds, hStart, hEnd);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuEventElapsedTimeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUEVENTELAPSEDTIME;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuEventElapsedTimeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pMilliseconds = pMilliseconds;
			request->hStart = hStart;
			request->hEnd = hEnd;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuEventElapsedTimeResponse*>(responsePayload);
			if (pMilliseconds) { *pMilliseconds = response->pMilliseconds; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuEventElapsedTime);
	return err;
}

CUresult cuImportExternalMemory(CUexternalMemory * extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC * memHandleDesc)
{
	TALLY_SPD_LOG("cuImportExternalMemory hooked");
#if defined(RUN_LOCALLY)
	return lcuImportExternalMemory(extMem_out, memHandleDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr * devPtr, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC * bufferDesc)
{
	TALLY_SPD_LOG("cuExternalMemoryGetMappedBuffer hooked");
#if defined(RUN_LOCALLY)
	return lcuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray * mipmap, CUexternalMemory  extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC * mipmapDesc)
{
	TALLY_SPD_LOG("cuExternalMemoryGetMappedMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDestroyExternalMemory(CUexternalMemory  extMem)
{
	TALLY_SPD_LOG("cuDestroyExternalMemory hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDestroyExternalMemory(extMem);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDestroyExternalMemoryArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDESTROYEXTERNALMEMORY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDestroyExternalMemoryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->extMem = extMem;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDestroyExternalMemory);
	return err;
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore * extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC * semHandleDesc)
{
	TALLY_SPD_LOG("cuImportExternalSemaphore hooked");
#if defined(RUN_LOCALLY)
	return lcuImportExternalSemaphore(extSem_out, semHandleDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)
{
	TALLY_SPD_LOG("cuSignalExternalSemaphoresAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore * extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS * paramsArray, unsigned int  numExtSems, CUstream  stream)
{
	TALLY_SPD_LOG("cuWaitExternalSemaphoresAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore  extSem)
{
	TALLY_SPD_LOG("cuDestroyExternalSemaphore hooked");
#if defined(RUN_LOCALLY)
	return lcuDestroyExternalSemaphore(extSem);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamWaitValue32_v2(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamWaitValue32_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamWaitValue32_v2(stream, addr, value, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamWaitValue64_v2(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamWaitValue64_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamWaitValue64_v2(stream, addr, value, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamWriteValue32_v2(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamWriteValue32_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamWriteValue32_v2(stream, addr, value, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamWriteValue64_v2(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamWriteValue64_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamWriteValue64_v2(stream, addr, value, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuStreamBatchMemOp_v2(CUstream  stream, unsigned int  count, CUstreamBatchMemOpParams * paramArray, unsigned int  flags)
{
	TALLY_SPD_LOG("cuStreamBatchMemOp_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuStreamBatchMemOp_v2(stream, count, paramArray, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuFuncSetSharedMemConfig(CUfunction  hfunc, CUsharedconfig  config)
{
	TALLY_SPD_LOG("cuFuncSetSharedMemConfig hooked");
#if defined(RUN_LOCALLY)
	return lcuFuncSetSharedMemConfig(hfunc, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuFuncGetModule(CUmodule * hmod, CUfunction  hfunc)
{
	TALLY_SPD_LOG("cuFuncGetModule hooked");
#if defined(RUN_LOCALLY)
	return lcuFuncGetModule(hmod, hfunc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunchKernelEx(const CUlaunchConfig * config, CUfunction  f, void ** kernelParams, void ** extra)
{
	TALLY_SPD_LOG("cuLaunchKernelEx hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunchKernelEx(config, f, kernelParams, extra);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunchCooperativeKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams)
{
	TALLY_SPD_LOG("cuLaunchCooperativeKernel hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS * launchParamsList, unsigned int  numDevices, unsigned int  flags)
{
	TALLY_SPD_LOG("cuLaunchCooperativeKernelMultiDevice hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunchHostFunc(CUstream  hStream, CUhostFn  fn, void * userData)
{
	TALLY_SPD_LOG("cuLaunchHostFunc hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunchHostFunc(hStream, fn, userData);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuFuncSetBlockShape(CUfunction  hfunc, int  x, int  y, int  z)
{
	TALLY_SPD_LOG("cuFuncSetBlockShape hooked");
#if defined(RUN_LOCALLY)
	return lcuFuncSetBlockShape(hfunc, x, y, z);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuFuncSetSharedSize(CUfunction  hfunc, unsigned int  bytes)
{
	TALLY_SPD_LOG("cuFuncSetSharedSize hooked");
#if defined(RUN_LOCALLY)
	return lcuFuncSetSharedSize(hfunc, bytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuParamSetSize(CUfunction  hfunc, unsigned int  numbytes)
{
	TALLY_SPD_LOG("cuParamSetSize hooked");
#if defined(RUN_LOCALLY)
	return lcuParamSetSize(hfunc, numbytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuParamSeti(CUfunction  hfunc, int  offset, unsigned int  value)
{
	TALLY_SPD_LOG("cuParamSeti hooked");
#if defined(RUN_LOCALLY)
	return lcuParamSeti(hfunc, offset, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuParamSetf(CUfunction  hfunc, int  offset, float  value)
{
	TALLY_SPD_LOG("cuParamSetf hooked");
#if defined(RUN_LOCALLY)
	return lcuParamSetf(hfunc, offset, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuParamSetv(CUfunction  hfunc, int  offset, void * ptr, unsigned int  numbytes)
{
	TALLY_SPD_LOG("cuParamSetv hooked");
#if defined(RUN_LOCALLY)
	return lcuParamSetv(hfunc, offset, ptr, numbytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunch(CUfunction  f)
{
	TALLY_SPD_LOG("cuLaunch hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunch(f);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunchGrid(CUfunction  f, int  grid_width, int  grid_height)
{
	TALLY_SPD_LOG("cuLaunchGrid hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunchGrid(f, grid_width, grid_height);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuLaunchGridAsync(CUfunction  f, int  grid_width, int  grid_height, CUstream  hStream)
{
	TALLY_SPD_LOG("cuLaunchGridAsync hooked");
#if defined(RUN_LOCALLY)
	return lcuLaunchGridAsync(f, grid_width, grid_height, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuParamSetTexRef(CUfunction  hfunc, int  texunit, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuParamSetTexRef hooked");
#if defined(RUN_LOCALLY)
	return lcuParamSetTexRef(hfunc, texunit, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphCreate(CUgraph * phGraph, unsigned int  flags)
{
	TALLY_SPD_LOG("cuGraphCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphCreate(phGraph, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddKernelNode_v2(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddKernelNode_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode  hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphKernelNodeGetParams_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphKernelNodeGetParams_v2(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphKernelNodeSetParams_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphKernelNodeSetParams_v2(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddMemcpyNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)
{
	TALLY_SPD_LOG("cuGraphAddMemcpyNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode  hNode, CUDA_MEMCPY3D * nodeParams)
{
	TALLY_SPD_LOG("cuGraphMemcpyNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphMemcpyNodeGetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode  hNode, const CUDA_MEMCPY3D * nodeParams)
{
	TALLY_SPD_LOG("cuGraphMemcpyNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphMemcpyNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddMemsetNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)
{
	TALLY_SPD_LOG("cuGraphAddMemsetNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode  hNode, CUDA_MEMSET_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphMemsetNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphMemsetNodeGetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphMemsetNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphMemsetNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddHostNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddHostNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphHostNodeGetParams(CUgraphNode  hNode, CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphHostNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphHostNodeGetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphHostNodeSetParams(CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphHostNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphHostNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddChildGraphNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraph  childGraph)
{
	TALLY_SPD_LOG("cuGraphAddChildGraphNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode  hNode, CUgraph * phGraph)
{
	TALLY_SPD_LOG("cuGraphChildGraphNodeGetGraph hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphChildGraphNodeGetGraph(hNode, phGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddEmptyNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies)
{
	TALLY_SPD_LOG("cuGraphAddEmptyNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddEventRecordNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)
{
	TALLY_SPD_LOG("cuGraphAddEventRecordNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode  hNode, CUevent * event_out)
{
	TALLY_SPD_LOG("cuGraphEventRecordNodeGetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphEventRecordNodeGetEvent(hNode, event_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode  hNode, CUevent  event)
{
	TALLY_SPD_LOG("cuGraphEventRecordNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphEventRecordNodeSetEvent(hNode, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddEventWaitNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUevent  event)
{
	TALLY_SPD_LOG("cuGraphAddEventWaitNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode  hNode, CUevent * event_out)
{
	TALLY_SPD_LOG("cuGraphEventWaitNodeGetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphEventWaitNodeGetEvent(hNode, event_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode  hNode, CUevent  event)
{
	TALLY_SPD_LOG("cuGraphEventWaitNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphEventWaitNodeSetEvent(hNode, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddExternalSemaphoresSignalNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode  hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * params_out)
{
	TALLY_SPD_LOG("cuGraphExternalSemaphoresSignalNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExternalSemaphoresSignalNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddExternalSemaphoresWaitNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode  hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS * params_out)
{
	TALLY_SPD_LOG("cuGraphExternalSemaphoresWaitNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExternalSemaphoresWaitNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddBatchMemOpNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddBatchMemOpNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode  hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams_out)
{
	TALLY_SPD_LOG("cuGraphBatchMemOpNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphBatchMemOpNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphBatchMemOpNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExecBatchMemOpNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddMemAllocNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddMemAllocNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode  hNode, CUDA_MEM_ALLOC_NODE_PARAMS * params_out)
{
	TALLY_SPD_LOG("cuGraphMemAllocNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphMemAllocNodeGetParams(hNode, params_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddMemFreeNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUdeviceptr  dptr)
{
	TALLY_SPD_LOG("cuGraphAddMemFreeNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode  hNode, CUdeviceptr * dptr_out)
{
	TALLY_SPD_LOG("cuGraphMemFreeNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphMemFreeNodeGetParams(hNode, dptr_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceGraphMemTrim(CUdevice  device)
{
	TALLY_SPD_LOG("cuDeviceGraphMemTrim hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceGraphMemTrim(device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceGetGraphMemAttribute(CUdevice  device, CUgraphMem_attribute  attr, void*  value)
{
	TALLY_SPD_LOG("cuDeviceGetGraphMemAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceGetGraphMemAttribute(device, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceSetGraphMemAttribute(CUdevice  device, CUgraphMem_attribute  attr, void*  value)
{
	TALLY_SPD_LOG("cuDeviceSetGraphMemAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceSetGraphMemAttribute(device, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphClone(CUgraph * phGraphClone, CUgraph  originalGraph)
{
	TALLY_SPD_LOG("cuGraphClone hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphClone(phGraphClone, originalGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeFindInClone(CUgraphNode * phNode, CUgraphNode  hOriginalNode, CUgraph  hClonedGraph)
{
	TALLY_SPD_LOG("cuGraphNodeFindInClone hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeGetType(CUgraphNode  hNode, CUgraphNodeType * type)
{
	TALLY_SPD_LOG("cuGraphNodeGetType hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeGetType(hNode, type);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphGetNodes(CUgraph  hGraph, CUgraphNode * nodes, size_t * numNodes)
{
	TALLY_SPD_LOG("cuGraphGetNodes hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphGetNodes(hGraph, nodes, numNodes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphGetRootNodes(CUgraph  hGraph, CUgraphNode * rootNodes, size_t * numRootNodes)
{
	TALLY_SPD_LOG("cuGraphGetRootNodes hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphGetRootNodes(hGraph, rootNodes, numRootNodes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphGetEdges(CUgraph  hGraph, CUgraphNode * from, CUgraphNode * to, size_t * numEdges)
{
	TALLY_SPD_LOG("cuGraphGetEdges hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphGetEdges(hGraph, from, to, numEdges);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeGetDependencies(CUgraphNode  hNode, CUgraphNode * dependencies, size_t * numDependencies)
{
	TALLY_SPD_LOG("cuGraphNodeGetDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeGetDependencies(hNode, dependencies, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode  hNode, CUgraphNode * dependentNodes, size_t * numDependentNodes)
{
	TALLY_SPD_LOG("cuGraphNodeGetDependentNodes hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddDependencies(CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)
{
	TALLY_SPD_LOG("cuGraphAddDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddDependencies(hGraph, from, to, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphRemoveDependencies(CUgraph  hGraph, const CUgraphNode * from, const CUgraphNode * to, size_t  numDependencies)
{
	TALLY_SPD_LOG("cuGraphRemoveDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphRemoveDependencies(hGraph, from, to, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphDestroyNode(CUgraphNode  hNode)
{
	TALLY_SPD_LOG("cuGraphDestroyNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphDestroyNode(hNode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec * phGraphExec, CUgraph  hGraph, unsigned long long  flags)
{
	TALLY_SPD_LOG("cuGraphInstantiateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuGraphInstantiateWithFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUGRAPHINSTANTIATEWITHFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuGraphInstantiateWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->phGraphExec = phGraphExec;
			request->hGraph = hGraph;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuGraphInstantiateWithFlagsResponse*>(responsePayload);
			if (phGraphExec) { *phGraphExec = response->phGraphExec; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuGraphInstantiateWithFlags);
	return err;
}

CUresult cuGraphInstantiateWithParams(CUgraphExec * phGraphExec, CUgraph  hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS * instantiateParams)
{
	TALLY_SPD_LOG("cuGraphInstantiateWithParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecGetFlags(CUgraphExec  hGraphExec, cuuint64_t * flags)
{
	TALLY_SPD_LOG("cuGraphExecGetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecGetFlags(hGraphExec, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExecKernelNodeSetParams_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMCPY3D * copyParams, CUcontext  ctx)
{
	TALLY_SPD_LOG("cuGraphExecMemcpyNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_MEMSET_NODE_PARAMS * memsetParams, CUcontext  ctx)
{
	TALLY_SPD_LOG("cuGraphExecMemsetNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_HOST_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExecHostNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraph  childGraph)
{
	TALLY_SPD_LOG("cuGraphExecChildGraphNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)
{
	TALLY_SPD_LOG("cuGraphExecEventRecordNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUevent  event)
{
	TALLY_SPD_LOG("cuGraphExecEventWaitNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExecExternalSemaphoresSignalNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExecExternalSemaphoresWaitNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeSetEnabled(CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int  isEnabled)
{
	TALLY_SPD_LOG("cuGraphNodeSetEnabled hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeGetEnabled(CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int * isEnabled)
{
	TALLY_SPD_LOG("cuGraphNodeGetEnabled hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphUpload(CUgraphExec  hGraphExec, CUstream  hStream)
{
	TALLY_SPD_LOG("cuGraphUpload hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphUpload(hGraphExec, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecDestroy(CUgraphExec  hGraphExec)
{
	TALLY_SPD_LOG("cuGraphExecDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuGraphExecDestroy(hGraphExec);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuGraphExecDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUGRAPHEXECDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuGraphExecDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hGraphExec = hGraphExec;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuGraphExecDestroy);
	return err;
}

CUresult cuGraphDestroy(CUgraph  hGraph)
{
	TALLY_SPD_LOG("cuGraphDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuGraphDestroy(hGraph);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuGraphDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUGRAPHDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuGraphDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hGraph = hGraph;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const CUresult*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuGraphDestroy);
	return err;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec  hGraphExec, CUgraph  hGraph, CUgraphExecUpdateResultInfo * resultInfo)
{
	TALLY_SPD_LOG("cuGraphExecUpdate_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo);
#else

    CUresult err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuGraphExecUpdate_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUGRAPHEXECUPDATE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuGraphExecUpdate_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hGraphExec = hGraphExec;
			request->hGraph = hGraph;
			request->resultInfo = resultInfo;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuGraphExecUpdate_v2Response*>(responsePayload);
			if (resultInfo) { *resultInfo = response->resultInfo; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuGraphExecUpdate_v2);
	return err;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode  dst, CUgraphNode  src)
{
	TALLY_SPD_LOG("cuGraphKernelNodeCopyAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphKernelNodeCopyAttributes(dst, src);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode  hNode, CUkernelNodeAttrID  attr, CUkernelNodeAttrValue * value_out)
{
	TALLY_SPD_LOG("cuGraphKernelNodeGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphKernelNodeGetAttribute(hNode, attr, value_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode  hNode, CUkernelNodeAttrID  attr, const CUkernelNodeAttrValue * value)
{
	TALLY_SPD_LOG("cuGraphKernelNodeSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphKernelNodeSetAttribute(hNode, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphDebugDotPrint(CUgraph  hGraph, const char * path, unsigned int  flags)
{
	TALLY_SPD_LOG("cuGraphDebugDotPrint hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphDebugDotPrint(hGraph, path, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuUserObjectCreate(CUuserObject * object_out, void * ptr, CUhostFn  destroy, unsigned int  initialRefcount, unsigned int  flags)
{
	TALLY_SPD_LOG("cuUserObjectCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuUserObjectRetain(CUuserObject  object, unsigned int  count)
{
	TALLY_SPD_LOG("cuUserObjectRetain hooked");
#if defined(RUN_LOCALLY)
	return lcuUserObjectRetain(object, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuUserObjectRelease(CUuserObject  object, unsigned int  count)
{
	TALLY_SPD_LOG("cuUserObjectRelease hooked");
#if defined(RUN_LOCALLY)
	return lcuUserObjectRelease(object, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphRetainUserObject(CUgraph  graph, CUuserObject  object, unsigned int  count, unsigned int  flags)
{
	TALLY_SPD_LOG("cuGraphRetainUserObject hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphRetainUserObject(graph, object, count, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphReleaseUserObject(CUgraph  graph, CUuserObject  object, unsigned int  count)
{
	TALLY_SPD_LOG("cuGraphReleaseUserObject hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphReleaseUserObject(graph, object, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphAddNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraphNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cuGraphAddNode hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphAddNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphNodeSetParams(CUgraphNode  hNode, CUgraphNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cuGraphNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphExecNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraphNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cuGraphExecNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphExecNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize)
{
	TALLY_SPD_LOG("cuOccupancyMaxActiveBlocksPerMultiprocessor hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, CUfunction  func, int  blockSize, size_t  dynamicSMemSize, unsigned int  flags)
{
	TALLY_SPD_LOG("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyMaxPotentialBlockSize(int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit)
{
	TALLY_SPD_LOG("cuOccupancyMaxPotentialBlockSize hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int * minGridSize, int * blockSize, CUfunction  func, CUoccupancyB2DSize  blockSizeToDynamicSMemSize, size_t  dynamicSMemSize, int  blockSizeLimit, unsigned int  flags)
{
	TALLY_SPD_LOG("cuOccupancyMaxPotentialBlockSizeWithFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, CUfunction  func, int  numBlocks, int  blockSize)
{
	TALLY_SPD_LOG("cuOccupancyAvailableDynamicSMemPerBlock hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyMaxPotentialClusterSize(int * clusterSize, CUfunction  func, const CUlaunchConfig * config)
{
	TALLY_SPD_LOG("cuOccupancyMaxPotentialClusterSize hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyMaxPotentialClusterSize(clusterSize, func, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuOccupancyMaxActiveClusters(int * numClusters, CUfunction  func, const CUlaunchConfig * config)
{
	TALLY_SPD_LOG("cuOccupancyMaxActiveClusters hooked");
#if defined(RUN_LOCALLY)
	return lcuOccupancyMaxActiveClusters(numClusters, func, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetArray(CUtexref  hTexRef, CUarray  hArray, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuTexRefSetArray hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetArray(hTexRef, hArray, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetMipmappedArray(CUtexref  hTexRef, CUmipmappedArray  hMipmappedArray, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuTexRefSetMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetAddress_v2(size_t * ByteOffset, CUtexref  hTexRef, CUdeviceptr  dptr, size_t  bytes)
{
	TALLY_SPD_LOG("cuTexRefSetAddress_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetAddress2D_v3(CUtexref  hTexRef, const CUDA_ARRAY_DESCRIPTOR * desc, CUdeviceptr  dptr, size_t  Pitch)
{
	TALLY_SPD_LOG("cuTexRefSetAddress2D_v3 hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetFormat(CUtexref  hTexRef, CUarray_format  fmt, int  NumPackedComponents)
{
	TALLY_SPD_LOG("cuTexRefSetFormat hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetAddressMode(CUtexref  hTexRef, int  dim, CUaddress_mode  am)
{
	TALLY_SPD_LOG("cuTexRefSetAddressMode hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetAddressMode(hTexRef, dim, am);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetFilterMode(CUtexref  hTexRef, CUfilter_mode  fm)
{
	TALLY_SPD_LOG("cuTexRefSetFilterMode hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetFilterMode(hTexRef, fm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref  hTexRef, CUfilter_mode  fm)
{
	TALLY_SPD_LOG("cuTexRefSetMipmapFilterMode hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetMipmapFilterMode(hTexRef, fm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref  hTexRef, float  bias)
{
	TALLY_SPD_LOG("cuTexRefSetMipmapLevelBias hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetMipmapLevelBias(hTexRef, bias);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref  hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp)
{
	TALLY_SPD_LOG("cuTexRefSetMipmapLevelClamp hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref  hTexRef, unsigned int  maxAniso)
{
	TALLY_SPD_LOG("cuTexRefSetMaxAnisotropy hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetMaxAnisotropy(hTexRef, maxAniso);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetBorderColor(CUtexref  hTexRef, float * pBorderColor)
{
	TALLY_SPD_LOG("cuTexRefSetBorderColor hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetBorderColor(hTexRef, pBorderColor);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefSetFlags(CUtexref  hTexRef, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuTexRefSetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefSetFlags(hTexRef, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr * pdptr, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetAddress_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetAddress_v2(pdptr, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetArray(CUarray * phArray, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetArray hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetArray(phArray, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray * phMipmappedArray, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetMipmappedArray(phMipmappedArray, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetAddressMode(CUaddress_mode * pam, CUtexref  hTexRef, int  dim)
{
	TALLY_SPD_LOG("cuTexRefGetAddressMode hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetAddressMode(pam, hTexRef, dim);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetFilterMode(CUfilter_mode * pfm, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetFilterMode hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetFilterMode(pfm, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetFormat(CUarray_format * pFormat, int * pNumChannels, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetFormat hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetFormat(pFormat, pNumChannels, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode * pfm, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetMipmapFilterMode hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetMipmapFilterMode(pfm, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetMipmapLevelBias(float * pbias, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetMipmapLevelBias hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetMipmapLevelBias(pbias, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetMipmapLevelClamp(float * pminMipmapLevelClamp, float * pmaxMipmapLevelClamp, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetMipmapLevelClamp hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetMaxAnisotropy(int * pmaxAniso, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetMaxAnisotropy hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetBorderColor(float * pBorderColor, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetBorderColor hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetBorderColor(pBorderColor, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefGetFlags(unsigned int * pFlags, CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefGetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefGetFlags(pFlags, hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefCreate(CUtexref * pTexRef)
{
	TALLY_SPD_LOG("cuTexRefCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefCreate(pTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexRefDestroy(CUtexref  hTexRef)
{
	TALLY_SPD_LOG("cuTexRefDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuTexRefDestroy(hTexRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuSurfRefSetArray(CUsurfref  hSurfRef, CUarray  hArray, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuSurfRefSetArray hooked");
#if defined(RUN_LOCALLY)
	return lcuSurfRefSetArray(hSurfRef, hArray, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuSurfRefGetArray(CUarray * phArray, CUsurfref  hSurfRef)
{
	TALLY_SPD_LOG("cuSurfRefGetArray hooked");
#if defined(RUN_LOCALLY)
	return lcuSurfRefGetArray(phArray, hSurfRef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexObjectCreate(CUtexObject * pTexObject, const CUDA_RESOURCE_DESC * pResDesc, const CUDA_TEXTURE_DESC * pTexDesc, const CUDA_RESOURCE_VIEW_DESC * pResViewDesc)
{
	TALLY_SPD_LOG("cuTexObjectCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexObjectDestroy(CUtexObject  texObject)
{
	TALLY_SPD_LOG("cuTexObjectDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuTexObjectDestroy(texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUtexObject  texObject)
{
	TALLY_SPD_LOG("cuTexObjectGetResourceDesc hooked");
#if defined(RUN_LOCALLY)
	return lcuTexObjectGetResourceDesc(pResDesc, texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC * pTexDesc, CUtexObject  texObject)
{
	TALLY_SPD_LOG("cuTexObjectGetTextureDesc hooked");
#if defined(RUN_LOCALLY)
	return lcuTexObjectGetTextureDesc(pTexDesc, texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC * pResViewDesc, CUtexObject  texObject)
{
	TALLY_SPD_LOG("cuTexObjectGetResourceViewDesc hooked");
#if defined(RUN_LOCALLY)
	return lcuTexObjectGetResourceViewDesc(pResViewDesc, texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuSurfObjectCreate(CUsurfObject * pSurfObject, const CUDA_RESOURCE_DESC * pResDesc)
{
	TALLY_SPD_LOG("cuSurfObjectCreate hooked");
#if defined(RUN_LOCALLY)
	return lcuSurfObjectCreate(pSurfObject, pResDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuSurfObjectDestroy(CUsurfObject  surfObject)
{
	TALLY_SPD_LOG("cuSurfObjectDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcuSurfObjectDestroy(surfObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC * pResDesc, CUsurfObject  surfObject)
{
	TALLY_SPD_LOG("cuSurfObjectGetResourceDesc hooked");
#if defined(RUN_LOCALLY)
	return lcuSurfObjectGetResourceDesc(pResDesc, surfObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTensorMapEncodeTiled(CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const cuuint32_t * boxDim, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill)
{
	TALLY_SPD_LOG("cuTensorMapEncodeTiled hooked");
#if defined(RUN_LOCALLY)
	return lcuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTensorMapEncodeIm2col(CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const int * pixelBoxLowerCorner, const int * pixelBoxUpperCorner, cuuint32_t  channelsPerPixel, cuuint32_t  pixelsPerColumn, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill)
{
	TALLY_SPD_LOG("cuTensorMapEncodeIm2col hooked");
#if defined(RUN_LOCALLY)
	return lcuTensorMapEncodeIm2col(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuTensorMapReplaceAddress(CUtensorMap * tensorMap, void * globalAddress)
{
	TALLY_SPD_LOG("cuTensorMapReplaceAddress hooked");
#if defined(RUN_LOCALLY)
	return lcuTensorMapReplaceAddress(tensorMap, globalAddress);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceCanAccessPeer(int * canAccessPeer, CUdevice  dev, CUdevice  peerDev)
{
	TALLY_SPD_LOG("cuDeviceCanAccessPeer hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceCanAccessPeer(canAccessPeer, dev, peerDev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCtxEnablePeerAccess(CUcontext  peerContext, unsigned int  Flags)
{
	TALLY_SPD_LOG("cuCtxEnablePeerAccess hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxEnablePeerAccess(peerContext, Flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCtxDisablePeerAccess(CUcontext  peerContext)
{
	TALLY_SPD_LOG("cuCtxDisablePeerAccess hooked");
#if defined(RUN_LOCALLY)
	return lcuCtxDisablePeerAccess(peerContext);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuDeviceGetP2PAttribute(int*  value, CUdevice_P2PAttribute  attrib, CUdevice  srcDevice, CUdevice  dstDevice)
{
	TALLY_SPD_LOG("cuDeviceGetP2PAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource  resource)
{
	TALLY_SPD_LOG("cuGraphicsUnregisterResource hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsUnregisterResource(resource);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray * pArray, CUgraphicsResource  resource, unsigned int  arrayIndex, unsigned int  mipLevel)
{
	TALLY_SPD_LOG("cuGraphicsSubResourceGetMappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray * pMipmappedArray, CUgraphicsResource  resource)
{
	TALLY_SPD_LOG("cuGraphicsResourceGetMappedMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * pDevPtr, size_t * pSize, CUgraphicsResource  resource)
{
	TALLY_SPD_LOG("cuGraphicsResourceGetMappedPointer_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource  resource, unsigned int  flags)
{
	TALLY_SPD_LOG("cuGraphicsResourceSetMapFlags_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsResourceSetMapFlags_v2(resource, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsMapResources(unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)
{
	TALLY_SPD_LOG("cuGraphicsMapResources hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsMapResources(count, resources, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGraphicsUnmapResources(unsigned int  count, CUgraphicsResource * resources, CUstream  hStream)
{
	TALLY_SPD_LOG("cuGraphicsUnmapResources hooked");
#if defined(RUN_LOCALLY)
	return lcuGraphicsUnmapResources(count, resources, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCoredumpGetAttribute(CUcoredumpSettings  attrib, void*  value, size_t * size)
{
	TALLY_SPD_LOG("cuCoredumpGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuCoredumpGetAttribute(attrib, value, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings  attrib, void * value, size_t * size)
{
	TALLY_SPD_LOG("cuCoredumpGetAttributeGlobal hooked");
#if defined(RUN_LOCALLY)
	return lcuCoredumpGetAttributeGlobal(attrib, value, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCoredumpSetAttribute(CUcoredumpSettings  attrib, void*  value, size_t * size)
{
	TALLY_SPD_LOG("cuCoredumpSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcuCoredumpSetAttribute(attrib, value, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings  attrib, void * value, size_t * size)
{
	TALLY_SPD_LOG("cuCoredumpSetAttributeGlobal hooked");
#if defined(RUN_LOCALLY)
	return lcuCoredumpSetAttributeGlobal(attrib, value, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuGetExportTable(const void ** ppExportTable, const CUuuid * pExportTableId)
{
	TALLY_SPD_LOG("cuGetExportTable hooked");
#if defined(RUN_LOCALLY)
	return lcuGetExportTable(ppExportTable, pExportTableId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceReset()
{
	TALLY_SPD_LOG("cudaDeviceReset hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceReset();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceResetArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICERESET;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceResetArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceReset);
	return err;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit  limit, size_t  value)
{
	TALLY_SPD_LOG("cudaDeviceSetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetLimit(limit, value);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetLimitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICESETLIMIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceSetLimitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->limit = limit;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetLimit);
	return err;
}

cudaError_t cudaDeviceGetLimit(size_t * pValue, enum cudaLimit  limit)
{
	TALLY_SPD_LOG("cudaDeviceGetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetLimit(pValue, limit);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetLimitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETLIMIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetLimitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pValue = pValue;
			request->limit = limit;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetLimitResponse*>(responsePayload);
			if (pValue) { *pValue = response->pValue; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetLimit);
	return err;
}

cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const struct cudaChannelFormatDesc * fmtDesc, int  device)
{
	TALLY_SPD_LOG("cudaDeviceGetTexture1DLinearMaxWidth hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
	TALLY_SPD_LOG("cudaDeviceGetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetCacheConfig(pCacheConfig);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetCacheConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pCacheConfig = pCacheConfig;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetCacheConfigResponse*>(responsePayload);
			if (pCacheConfig) { *pCacheConfig = response->pCacheConfig; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetCacheConfig);
	return err;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	TALLY_SPD_LOG("cudaDeviceGetStreamPriorityRange hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetStreamPriorityRangeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETSTREAMPRIORITYRANGE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetStreamPriorityRangeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->leastPriority = leastPriority;
			request->greatestPriority = greatestPriority;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetStreamPriorityRangeResponse*>(responsePayload);
			if (leastPriority) { *leastPriority = response->leastPriority; }
			if (greatestPriority) { *greatestPriority = response->greatestPriority; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetStreamPriorityRange);
	return err;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache  cacheConfig)
{
	TALLY_SPD_LOG("cudaDeviceSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetCacheConfig(cacheConfig);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetCacheConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICESETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceSetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->cacheConfig = cacheConfig;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetCacheConfig);
	return err;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig)
{
	TALLY_SPD_LOG("cudaDeviceGetSharedMemConfig hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceGetSharedMemConfig(pConfig);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig  config)
{
	TALLY_SPD_LOG("cudaDeviceSetSharedMemConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetSharedMemConfig(config);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetSharedMemConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICESETSHAREDMEMCONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceSetSharedMemConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->config = config;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetSharedMemConfig);
	return err;
}

cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId)
{
	TALLY_SPD_LOG("cudaDeviceGetByPCIBusId hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceGetByPCIBusId(device, pciBusId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int  len, int  device)
{
	TALLY_SPD_LOG("cudaDeviceGetPCIBusId hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetPCIBusId(pciBusId, len, device);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetPCIBusIdArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETPCIBUSID;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetPCIBusIdArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pciBusId = pciBusId;
			request->len = len;
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetPCIBusIdResponse*>(responsePayload);
			if (pciBusId) { *pciBusId = response->pciBusId; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetPCIBusId);
	return err;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaIpcGetEventHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcGetEventHandle(handle, event);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcGetEventHandleArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAIPCGETEVENTHANDLE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaIpcGetEventHandleArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->event = event;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaIpcGetEventHandleResponse*>(responsePayload);
			if (handle) { *handle = response->handle; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcGetEventHandle);
	return err;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t  handle)
{
	TALLY_SPD_LOG("cudaIpcOpenEventHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcOpenEventHandle(event, handle);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcOpenEventHandleArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAIPCOPENEVENTHANDLE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaIpcOpenEventHandleArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;
			request->handle = handle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaIpcOpenEventHandleResponse*>(responsePayload);
			if (event) { *event = response->event; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcOpenEventHandle);
	return err;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr)
{
	TALLY_SPD_LOG("cudaIpcGetMemHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcGetMemHandle(handle, devPtr);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcGetMemHandleArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAIPCGETMEMHANDLE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaIpcGetMemHandleArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->devPtr = devPtr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaIpcGetMemHandleResponse*>(responsePayload);
			if (handle) { *handle = response->handle; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcGetMemHandle);
	return err;
}

cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t  handle, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaIpcOpenMemHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcOpenMemHandle(devPtr, handle, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcOpenMemHandleArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAIPCOPENMEMHANDLE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaIpcOpenMemHandleArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->devPtr = devPtr;
			request->handle = handle;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaIpcOpenMemHandleResponse*>(responsePayload);
			if (devPtr) { *devPtr = response->devPtr; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcOpenMemHandle);
	return err;
}

cudaError_t cudaIpcCloseMemHandle(void * devPtr)
{
	TALLY_SPD_LOG("cudaIpcCloseMemHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcCloseMemHandle(devPtr);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcCloseMemHandleArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAIPCCLOSEMEMHANDLE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaIpcCloseMemHandleArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->devPtr = devPtr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcCloseMemHandle);
	return err;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget  target, enum cudaFlushGPUDirectRDMAWritesScope  scope)
{
	TALLY_SPD_LOG("cudaDeviceFlushGPUDirectRDMAWrites hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceFlushGPUDirectRDMAWrites(target, scope);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceFlushGPUDirectRDMAWritesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEFLUSHGPUDIRECTRDMAWRITES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceFlushGPUDirectRDMAWritesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->target = target;
			request->scope = scope;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceFlushGPUDirectRDMAWrites);
	return err;
}

cudaError_t cudaThreadExit()
{
	TALLY_SPD_LOG("cudaThreadExit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadExit();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadExitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADEXIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaThreadExitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadExit);
	return err;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit  limit, size_t  value)
{
	TALLY_SPD_LOG("cudaThreadSetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadSetLimit(limit, value);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadSetLimitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADSETLIMIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaThreadSetLimitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->limit = limit;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadSetLimit);
	return err;
}

cudaError_t cudaThreadGetLimit(size_t * pValue, enum cudaLimit  limit)
{
	TALLY_SPD_LOG("cudaThreadGetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadGetLimit(pValue, limit);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadGetLimitArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADGETLIMIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaThreadGetLimitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pValue = pValue;
			request->limit = limit;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaThreadGetLimitResponse*>(responsePayload);
			if (pValue) { *pValue = response->pValue; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadGetLimit);
	return err;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
	TALLY_SPD_LOG("cudaThreadGetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadGetCacheConfig(pCacheConfig);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadGetCacheConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADGETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaThreadGetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pCacheConfig = pCacheConfig;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaThreadGetCacheConfigResponse*>(responsePayload);
			if (pCacheConfig) { *pCacheConfig = response->pCacheConfig; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadGetCacheConfig);
	return err;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache  cacheConfig)
{
	TALLY_SPD_LOG("cudaThreadSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadSetCacheConfig(cacheConfig);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadSetCacheConfigArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADSETCACHECONFIG;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaThreadSetCacheConfigArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->cacheConfig = cacheConfig;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadSetCacheConfig);
	return err;
}

const char* cudaGetErrorName(cudaError_t  error)
{
	TALLY_SPD_LOG("cudaGetErrorName hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetErrorName(error);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* cudaGetErrorString(cudaError_t  error)
{
	TALLY_SPD_LOG("cudaGetErrorString hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetErrorString(error);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetDeviceCount(int * count)
{
	TALLY_SPD_LOG("cudaGetDeviceCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGetDeviceCount(count);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGetDeviceCountArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGETDEVICECOUNT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGetDeviceCountArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->count = count;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaGetDeviceCountResponse*>(responsePayload);
			if (count) { *count = response->count; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGetDeviceCount);
	return err;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp * prop, int  device)
{
	TALLY_SPD_LOG("cudaGetDeviceProperties_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGetDeviceProperties_v2(prop, device);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGetDeviceProperties_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGETDEVICEPROPERTIES_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGetDeviceProperties_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->prop = prop;
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaGetDeviceProperties_v2Response*>(responsePayload);
			if (prop) { *prop = response->prop; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGetDeviceProperties_v2);
	return err;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int  device)
{
	TALLY_SPD_LOG("cudaDeviceGetDefaultMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetDefaultMemPool(memPool, device);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetDefaultMemPoolArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETDEFAULTMEMPOOL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetDefaultMemPoolArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->memPool = memPool;
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetDefaultMemPoolResponse*>(responsePayload);
			if (memPool) { *memPool = response->memPool; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetDefaultMemPool);
	return err;
}

cudaError_t cudaDeviceSetMemPool(int  device, cudaMemPool_t  memPool)
{
	TALLY_SPD_LOG("cudaDeviceSetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetMemPool(device, memPool);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetMemPoolArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICESETMEMPOOL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceSetMemPoolArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->device = device;
			request->memPool = memPool;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetMemPool);
	return err;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int  device)
{
	TALLY_SPD_LOG("cudaDeviceGetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetMemPool(memPool, device);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetMemPoolArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETMEMPOOL;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetMemPoolArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->memPool = memPool;
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetMemPoolResponse*>(responsePayload);
			if (memPool) { *memPool = response->memPool; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetMemPool);
	return err;
}

cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int  device, int  flags)
{
	TALLY_SPD_LOG("cudaDeviceGetNvSciSyncAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceGetP2PAttribute(int * value, enum cudaDeviceP2PAttr  attr, int  srcDevice, int  dstDevice)
{
	TALLY_SPD_LOG("cudaDeviceGetP2PAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetP2PAttributeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETP2PATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetP2PAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->value = value;
			request->attr = attr;
			request->srcDevice = srcDevice;
			request->dstDevice = dstDevice;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetP2PAttributeResponse*>(responsePayload);
			if (value) { *value = response->value; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetP2PAttribute);
	return err;
}

cudaError_t cudaInitDevice(int  device, unsigned int  deviceFlags, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaInitDevice hooked");
#if defined(RUN_LOCALLY)
	return lcudaInitDevice(device, deviceFlags, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaSetValidDevices(int * device_arr, int  len)
{
	TALLY_SPD_LOG("cudaSetValidDevices hooked");
#if defined(RUN_LOCALLY)
	return lcudaSetValidDevices(device_arr, len);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags)
{
	TALLY_SPD_LOG("cudaSetDeviceFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaSetDeviceFlags(flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaSetDeviceFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASETDEVICEFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaSetDeviceFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaSetDeviceFlags);
	return err;
}

cudaError_t cudaGetDeviceFlags(unsigned int * flags)
{
	TALLY_SPD_LOG("cudaGetDeviceFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGetDeviceFlags(flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGetDeviceFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGETDEVICEFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGetDeviceFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaGetDeviceFlagsResponse*>(responsePayload);
			if (flags) { *flags = response->flags; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGetDeviceFlags);
	return err;
}

cudaError_t cudaStreamGetPriority(cudaStream_t  hStream, int * priority)
{
	TALLY_SPD_LOG("cudaStreamGetPriority hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamGetPriority(hStream, priority);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamGetPriorityArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMGETPRIORITY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamGetPriorityArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;
			request->priority = priority;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamGetPriorityResponse*>(responsePayload);
			if (priority) { *priority = response->priority; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamGetPriority);
	return err;
}

cudaError_t cudaStreamGetFlags(cudaStream_t  hStream, unsigned int * flags)
{
	TALLY_SPD_LOG("cudaStreamGetFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamGetFlags(hStream, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamGetFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMGETFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamGetFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hStream = hStream;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamGetFlagsResponse*>(responsePayload);
			if (flags) { *flags = response->flags; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamGetFlags);
	return err;
}

cudaError_t cudaStreamGetId(cudaStream_t  hStream, unsigned long long * streamId)
{
	TALLY_SPD_LOG("cudaStreamGetId hooked");
#if defined(RUN_LOCALLY)
	return lcudaStreamGetId(hStream, streamId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaCtxResetPersistingL2Cache()
{
	TALLY_SPD_LOG("cudaCtxResetPersistingL2Cache hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaCtxResetPersistingL2Cache();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaCtxResetPersistingL2CacheArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDACTXRESETPERSISTINGL2CACHE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaCtxResetPersistingL2CacheArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaCtxResetPersistingL2Cache);
	return err;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t  dst, cudaStream_t  src)
{
	TALLY_SPD_LOG("cudaStreamCopyAttributes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCopyAttributes(dst, src);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCopyAttributesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCOPYATTRIBUTES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCopyAttributesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dst = dst;
			request->src = src;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCopyAttributes);
	return err;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t  hStream, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out)
{
	TALLY_SPD_LOG("cudaStreamGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaStreamGetAttribute(hStream, attr, value_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaStreamSetAttribute(cudaStream_t  hStream, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value)
{
	TALLY_SPD_LOG("cudaStreamSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaStreamSetAttribute(hStream, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaStreamDestroy(cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaStreamDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamDestroy(stream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamDestroy);
	return err;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t  stream, cudaEvent_t  event, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaStreamWaitEvent hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamWaitEvent(stream, event, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamWaitEventArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMWAITEVENT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamWaitEventArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->event = event;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamWaitEvent);
	return err;
}

cudaError_t cudaStreamAddCallback(cudaStream_t  stream, cudaStreamCallback_t  callback, void * userData, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaStreamAddCallback hooked");
#if defined(RUN_LOCALLY)
	return lcudaStreamAddCallback(stream, callback, userData, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaStreamQuery(cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaStreamQuery hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamQuery(stream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamQueryArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMQUERY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamQueryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamQuery);
	return err;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t  stream, void * devPtr, size_t  length, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaStreamAttachMemAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaStreamAttachMemAsync(stream, devPtr, length, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode * mode)
{
	TALLY_SPD_LOG("cudaThreadExchangeStreamCaptureMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadExchangeStreamCaptureMode(mode);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadExchangeStreamCaptureModeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDATHREADEXCHANGESTREAMCAPTUREMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaThreadExchangeStreamCaptureModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaThreadExchangeStreamCaptureModeResponse*>(responsePayload);
			if (mode) { *mode = response->mode; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadExchangeStreamCaptureMode);
	return err;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus)
{
	TALLY_SPD_LOG("cudaStreamIsCapturing hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamIsCapturing(stream, pCaptureStatus);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamIsCapturingArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMISCAPTURING;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamIsCapturingArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->pCaptureStatus = pCaptureStatus;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamIsCapturingResponse*>(responsePayload);
			if (pCaptureStatus) { *pCaptureStatus = response->pCaptureStatus; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamIsCapturing);
	return err;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t  stream, cudaGraphNode_t * dependencies, size_t  numDependencies, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaStreamUpdateCaptureDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaEventCreate(cudaEvent_t * event)
{
	TALLY_SPD_LOG("cudaEventCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventCreate(event);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaEventCreateResponse*>(responsePayload);
			if (event) { *event = response->event; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventCreate);
	return err;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaEventCreateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventCreateWithFlags(event, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventCreateWithFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTCREATEWITHFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventCreateWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaEventCreateWithFlagsResponse*>(responsePayload);
			if (event) { *event = response->event; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventCreateWithFlags);
	return err;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t  event, cudaStream_t  stream, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaEventRecordWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventRecordWithFlags(event, stream, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventRecordWithFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTRECORDWITHFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventRecordWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;
			request->stream = stream;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventRecordWithFlags);
	return err;
}

cudaError_t cudaEventQuery(cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaEventQuery hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventQuery(event);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventQueryArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTQUERY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventQueryArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventQuery);
	return err;
}

cudaError_t cudaEventSynchronize(cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaEventSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventSynchronize(event);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventSynchronizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventSynchronizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventSynchronize);
	return err;
}

cudaError_t cudaEventDestroy(cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaEventDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventDestroy(event);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->event = event;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventDestroy);
	return err;
}

cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t  start, cudaEvent_t  end)
{
	TALLY_SPD_LOG("cudaEventElapsedTime hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventElapsedTime(ms, start, end);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventElapsedTimeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAEVENTELAPSEDTIME;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaEventElapsedTimeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->ms = ms;
			request->start = start;
			request->end = end;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaEventElapsedTimeResponse*>(responsePayload);
			if (ms) { *ms = response->ms; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventElapsedTime);
	return err;
}

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const struct cudaExternalMemoryHandleDesc * memHandleDesc)
{
	TALLY_SPD_LOG("cudaImportExternalMemory hooked");
#if defined(RUN_LOCALLY)
	return lcudaImportExternalMemory(extMem_out, memHandleDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryBufferDesc * bufferDesc)
{
	TALLY_SPD_LOG("cudaExternalMemoryGetMappedBuffer hooked");
#if defined(RUN_LOCALLY)
	return lcudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t  extMem, const struct cudaExternalMemoryMipmappedArrayDesc * mipmapDesc)
{
	TALLY_SPD_LOG("cudaExternalMemoryGetMappedMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t  extMem)
{
	TALLY_SPD_LOG("cudaDestroyExternalMemory hooked");
#if defined(RUN_LOCALLY)
	return lcudaDestroyExternalMemory(extMem);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const struct cudaExternalSemaphoreHandleDesc * semHandleDesc)
{
	TALLY_SPD_LOG("cudaImportExternalSemaphore hooked");
#if defined(RUN_LOCALLY)
	return lcudaImportExternalSemaphore(extSem_out, semHandleDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreSignalParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaSignalExternalSemaphoresAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const struct cudaExternalSemaphoreWaitParams * paramsArray, unsigned int  numExtSems, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaWaitExternalSemaphoresAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t  extSem)
{
	TALLY_SPD_LOG("cudaDestroyExternalSemaphore hooked");
#if defined(RUN_LOCALLY)
	return lcudaDestroyExternalSemaphore(extSem);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args)
{
	TALLY_SPD_LOG("cudaLaunchKernelExC hooked");
#if defined(RUN_LOCALLY)
	return lcudaLaunchKernelExC(config, func, args);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaLaunchCooperativeKernel hooked");
#if defined(RUN_LOCALLY)
	return lcudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams * launchParamsList, unsigned int  numDevices, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaLaunchCooperativeKernelMultiDevice hooked");
#if defined(RUN_LOCALLY)
	return lcudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaFuncSetCacheConfig(const void * func, enum cudaFuncCache  cacheConfig)
{
	TALLY_SPD_LOG("cudaFuncSetCacheConfig hooked");
#if defined(RUN_LOCALLY)
	return lcudaFuncSetCacheConfig(func, cacheConfig);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaFuncSetSharedMemConfig(const void * func, enum cudaSharedMemConfig  config)
{
	TALLY_SPD_LOG("cudaFuncSetSharedMemConfig hooked");
#if defined(RUN_LOCALLY)
	return lcudaFuncSetSharedMemConfig(func, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaSetDoubleForDevice(double * d)
{
	TALLY_SPD_LOG("cudaSetDoubleForDevice hooked");
#if defined(RUN_LOCALLY)
	return lcudaSetDoubleForDevice(d);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaSetDoubleForHost(double * d)
{
	TALLY_SPD_LOG("cudaSetDoubleForHost hooked");
#if defined(RUN_LOCALLY)
	return lcudaSetDoubleForHost(d);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaLaunchHostFunc(cudaStream_t  stream, cudaHostFn_t  fn, void * userData)
{
	TALLY_SPD_LOG("cudaLaunchHostFunc hooked");
#if defined(RUN_LOCALLY)
	return lcudaLaunchHostFunc(stream, fn, userData);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int  blockSize, size_t  dynamicSMemSize)
{
	TALLY_SPD_LOG("cudaOccupancyMaxActiveBlocksPerMultiprocessor hooked");
#if defined(RUN_LOCALLY)
	return lcudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int  numBlocks, int  blockSize)
{
	TALLY_SPD_LOG("cudaOccupancyAvailableDynamicSMemPerBlock hooked");
#if defined(RUN_LOCALLY)
	return lcudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig)
{
	TALLY_SPD_LOG("cudaOccupancyMaxPotentialClusterSize hooked");
#if defined(RUN_LOCALLY)
	return lcudaOccupancyMaxPotentialClusterSize(clusterSize, func, launchConfig);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig)
{
	TALLY_SPD_LOG("cudaOccupancyMaxActiveClusters hooked");
#if defined(RUN_LOCALLY)
	return lcudaOccupancyMaxActiveClusters(numClusters, func, launchConfig);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMallocManaged(void ** devPtr, size_t  size, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaMallocManaged hooked");
#if defined(RUN_LOCALLY)
	return lcudaMallocManaged(devPtr, size, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t  width, size_t  height)
{
	TALLY_SPD_LOG("cudaMallocPitch hooked");
#if defined(RUN_LOCALLY)
	return lcudaMallocPitch(devPtr, pitch, width, height);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMallocArray(cudaArray_t * array, const struct cudaChannelFormatDesc * desc, size_t  width, size_t  height, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaMallocArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMallocArray(array, desc, width, height, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaFreeArray(cudaArray_t  array)
{
	TALLY_SPD_LOG("cudaFreeArray hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaFreeArray(array);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaFreeArrayArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAFREEARRAY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaFreeArrayArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->array = array;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaFreeArray);
	return err;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t  mipmappedArray)
{
	TALLY_SPD_LOG("cudaFreeMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaFreeMipmappedArray(mipmappedArray);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaHostRegister(void * ptr, size_t  size, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaHostRegister hooked");
#if defined(RUN_LOCALLY)
	return lcudaHostRegister(ptr, size, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaHostUnregister(void * ptr)
{
	TALLY_SPD_LOG("cudaHostUnregister hooked");
#if defined(RUN_LOCALLY)
	return lcudaHostUnregister(ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaHostGetDevicePointer hooked");
#if defined(RUN_LOCALLY)
	return lcudaHostGetDevicePointer(pDevice, pHost, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost)
{
	TALLY_SPD_LOG("cudaHostGetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcudaHostGetFlags(pFlags, pHost);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr*  pitchedDevPtr, struct cudaExtent  extent)
{
	TALLY_SPD_LOG("cudaMalloc3D hooked");
#if defined(RUN_LOCALLY)
	return lcudaMalloc3D(pitchedDevPtr, extent);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaMalloc3DArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMalloc3DArray(array, desc, extent, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const struct cudaChannelFormatDesc*  desc, struct cudaExtent  extent, unsigned int  numLevels, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaMallocMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t  mipmappedArray, unsigned int  level)
{
	TALLY_SPD_LOG("cudaGetMipmappedArrayLevel hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * p)
{
	TALLY_SPD_LOG("cudaMemcpy3D hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy3D(p);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms * p)
{
	TALLY_SPD_LOG("cudaMemcpy3DPeer hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy3DPeer(p);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * p, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpy3DAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy3DAsync(p, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms * p, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpy3DPeerAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy3DPeerAsync(p, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemGetInfo(size_t * free, size_t * total)
{
	TALLY_SPD_LOG("cudaMemGetInfo hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaMemGetInfo(free, total);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaMemGetInfoArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMGETINFO;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemGetInfoArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->free = free;
			request->total = total;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaMemGetInfoResponse*>(responsePayload);
			if (free) { *free = response->free; }
			if (total) { *total = response->total; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaMemGetInfo);
	return err;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc * desc, struct cudaExtent * extent, unsigned int * flags, cudaArray_t  array)
{
	TALLY_SPD_LOG("cudaArrayGetInfo hooked");
#if defined(RUN_LOCALLY)
	return lcudaArrayGetInfo(desc, extent, flags, array);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t  hArray, unsigned int  planeIdx)
{
	TALLY_SPD_LOG("cudaArrayGetPlane hooked");
#if defined(RUN_LOCALLY)
	return lcudaArrayGetPlane(pPlaneArray, hArray, planeIdx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t  array, int  device)
{
	TALLY_SPD_LOG("cudaArrayGetMemoryRequirements hooked");
#if defined(RUN_LOCALLY)
	return lcudaArrayGetMemoryRequirements(memoryRequirements, array, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t  mipmap, int  device)
{
	TALLY_SPD_LOG("cudaMipmappedArrayGetMemoryRequirements hooked");
#if defined(RUN_LOCALLY)
	return lcudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties * sparseProperties, cudaArray_t  array)
{
	TALLY_SPD_LOG("cudaArrayGetSparseProperties hooked");
#if defined(RUN_LOCALLY)
	return lcudaArrayGetSparseProperties(sparseProperties, array);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t  mipmap)
{
	TALLY_SPD_LOG("cudaMipmappedArrayGetSparseProperties hooked");
#if defined(RUN_LOCALLY)
	return lcudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyPeer(void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count)
{
	TALLY_SPD_LOG("cudaMemcpyPeer hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2D(void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpy2D hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpy2DToArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2DFromArray(void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpy2DFromArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  width, size_t  height, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpy2DArrayToArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpyToSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyToSymbol(symbol, src, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpyFromSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyPeerAsync(void * dst, int  dstDevice, const void * src, int  srcDevice, size_t  count, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpyPeerAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2DAsync(void * dst, size_t  dpitch, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpy2DAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  spitch, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpy2DToArrayAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t  dpitch, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  width, size_t  height, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpy2DFromArrayAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpyToSymbolAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpyFromSymbolAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemset2D(void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height)
{
	TALLY_SPD_LOG("cudaMemset2D hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemset2D(devPtr, pitch, value, width, height);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent)
{
	TALLY_SPD_LOG("cudaMemset3D hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemset3D(pitchedDevPtr, value, extent);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemset2DAsync(void * devPtr, size_t  pitch, int  value, size_t  width, size_t  height, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemset2DAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr  pitchedDevPtr, int  value, struct cudaExtent  extent, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemset3DAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol)
{
	TALLY_SPD_LOG("cudaGetSymbolAddress hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetSymbolAddress(devPtr, symbol);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol)
{
	TALLY_SPD_LOG("cudaGetSymbolSize hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetSymbolSize(size, symbol);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t  count, int  dstDevice, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemPrefetchAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPrefetchAsync_v2(const void * devPtr, size_t  count, struct cudaMemLocation  location, unsigned int  flags, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemPrefetchAsync_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPrefetchAsync_v2(devPtr, count, location, flags, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemAdvise(const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, int  device)
{
	TALLY_SPD_LOG("cudaMemAdvise hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemAdvise(devPtr, count, advice, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemAdvise_v2(const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, struct cudaMemLocation  location)
{
	TALLY_SPD_LOG("cudaMemAdvise_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemAdvise_v2(devPtr, count, advice, location);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemRangeGetAttribute(void * data, size_t  dataSize, enum cudaMemRangeAttribute  attribute, const void * devPtr, size_t  count)
{
	TALLY_SPD_LOG("cudaMemRangeGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, enum cudaMemRangeAttribute * attributes, size_t  numAttributes, const void * devPtr, size_t  count)
{
	TALLY_SPD_LOG("cudaMemRangeGetAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyToArray(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpyToArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpyFromArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t  dst, size_t  wOffsetDst, size_t  hOffsetDst, cudaArray_const_t  src, size_t  wOffsetSrc, size_t  hOffsetSrc, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaMemcpyArrayToArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t  dst, size_t  wOffset, size_t  hOffset, const void * src, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpyToArrayAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t  src, size_t  wOffset, size_t  hOffset, size_t  count, enum cudaMemcpyKind  kind, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMemcpyFromArrayAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMallocAsync(void ** devPtr, size_t  size, cudaStream_t  hStream)
{
	TALLY_SPD_LOG("cudaMallocAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMallocAsync(devPtr, size, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaFreeAsync(void * devPtr, cudaStream_t  hStream)
{
	TALLY_SPD_LOG("cudaFreeAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaFreeAsync(devPtr, hStream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t  memPool, size_t  minBytesToKeep)
{
	TALLY_SPD_LOG("cudaMemPoolTrimTo hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaMemPoolTrimTo(memPool, minBytesToKeep);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaMemPoolTrimToArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMPOOLTRIMTO;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemPoolTrimToArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->memPool = memPool;
			request->minBytesToKeep = minBytesToKeep;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaMemPoolTrimTo);
	return err;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)
{
	TALLY_SPD_LOG("cudaMemPoolSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolSetAttribute(memPool, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t  memPool, enum cudaMemPoolAttr  attr, void * value)
{
	TALLY_SPD_LOG("cudaMemPoolGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolGetAttribute(memPool, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t  memPool, const struct cudaMemAccessDesc * descList, size_t  count)
{
	TALLY_SPD_LOG("cudaMemPoolSetAccess hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolSetAccess(memPool, descList, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags * flags, cudaMemPool_t  memPool, struct cudaMemLocation * location)
{
	TALLY_SPD_LOG("cudaMemPoolGetAccess hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolGetAccess(flags, memPool, location);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t * memPool, const struct cudaMemPoolProps * poolProps)
{
	TALLY_SPD_LOG("cudaMemPoolCreate hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolCreate(memPool, poolProps);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t  memPool)
{
	TALLY_SPD_LOG("cudaMemPoolDestroy hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolDestroy(memPool);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMallocFromPoolAsync(void ** ptr, size_t  size, cudaMemPool_t  memPool, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaMallocFromPoolAsync hooked");
#if defined(RUN_LOCALLY)
	return lcudaMallocFromPoolAsync(ptr, size, memPool, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t  memPool, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaMemPoolExportToShareableHandle hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, enum cudaMemAllocationHandleType  handleType, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaMemPoolImportFromShareableHandle hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData * exportData, void * ptr)
{
	TALLY_SPD_LOG("cudaMemPoolExportPointer hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolExportPointer(exportData, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t  memPool, struct cudaMemPoolPtrExportData * exportData)
{
	TALLY_SPD_LOG("cudaMemPoolImportPointer hooked");
#if defined(RUN_LOCALLY)
	return lcudaMemPoolImportPointer(ptr, memPool, exportData);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int  device, int  peerDevice)
{
	TALLY_SPD_LOG("cudaDeviceCanAccessPeer hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceEnablePeerAccess(int  peerDevice, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaDeviceEnablePeerAccess hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceEnablePeerAccess(peerDevice, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceDisablePeerAccess(int  peerDevice)
{
	TALLY_SPD_LOG("cudaDeviceDisablePeerAccess hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceDisablePeerAccess(peerDevice);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t  resource)
{
	TALLY_SPD_LOG("cudaGraphicsUnregisterResource hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsUnregisterResource(resource);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t  resource, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaGraphicsResourceSetMapFlags hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsResourceSetMapFlags(resource, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsMapResources(int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaGraphicsMapResources hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsMapResources(count, resources, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsUnmapResources(int  count, cudaGraphicsResource_t * resources, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaGraphicsUnmapResources hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsUnmapResources(count, resources, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t  resource)
{
	TALLY_SPD_LOG("cudaGraphicsResourceGetMappedPointer hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t  resource, unsigned int  arrayIndex, unsigned int  mipLevel)
{
	TALLY_SPD_LOG("cudaGraphicsSubResourceGetMappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t  resource)
{
	TALLY_SPD_LOG("cudaGraphicsResourceGetMappedMipmappedArray hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * desc, cudaArray_const_t  array)
{
	TALLY_SPD_LOG("cudaGetChannelDesc hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetChannelDesc(desc, array);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int  x, int  y, int  z, int  w, enum cudaChannelFormatKind  f)
{
	TALLY_SPD_LOG("cudaCreateChannelDesc hooked");
#if defined(RUN_LOCALLY)
	return lcudaCreateChannelDesc(x, y, z, w, f);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc)
{
	TALLY_SPD_LOG("cudaCreateTextureObject hooked");
#if defined(RUN_LOCALLY)
	return lcudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t  texObject)
{
	TALLY_SPD_LOG("cudaDestroyTextureObject hooked");
#if defined(RUN_LOCALLY)
	return lcudaDestroyTextureObject(texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaTextureObject_t  texObject)
{
	TALLY_SPD_LOG("cudaGetTextureObjectResourceDesc hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetTextureObjectResourceDesc(pResDesc, texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc * pTexDesc, cudaTextureObject_t  texObject)
{
	TALLY_SPD_LOG("cudaGetTextureObjectTextureDesc hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetTextureObjectTextureDesc(pTexDesc, texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t  texObject)
{
	TALLY_SPD_LOG("cudaGetTextureObjectResourceViewDesc hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const struct cudaResourceDesc * pResDesc)
{
	TALLY_SPD_LOG("cudaCreateSurfaceObject hooked");
#if defined(RUN_LOCALLY)
	return lcudaCreateSurfaceObject(pSurfObject, pResDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t  surfObject)
{
	TALLY_SPD_LOG("cudaDestroySurfaceObject hooked");
#if defined(RUN_LOCALLY)
	return lcudaDestroySurfaceObject(surfObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc * pResDesc, cudaSurfaceObject_t  surfObject)
{
	TALLY_SPD_LOG("cudaGetSurfaceObjectResourceDesc hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDriverGetVersion(int * driverVersion)
{
	TALLY_SPD_LOG("cudaDriverGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDriverGetVersion(driverVersion);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDriverGetVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADRIVERGETVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDriverGetVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->driverVersion = driverVersion;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDriverGetVersionResponse*>(responsePayload);
			if (driverVersion) { *driverVersion = response->driverVersion; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDriverGetVersion);
	return err;
}

cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)
{
	TALLY_SPD_LOG("cudaRuntimeGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaRuntimeGetVersion(runtimeVersion);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaRuntimeGetVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDARUNTIMEGETVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaRuntimeGetVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->runtimeVersion = runtimeVersion;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaRuntimeGetVersionResponse*>(responsePayload);
			if (runtimeVersion) { *runtimeVersion = response->runtimeVersion; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaRuntimeGetVersion);
	return err;
}

cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaGraphCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphCreate(pGraph, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pGraph = pGraph;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaGraphCreateResponse*>(responsePayload);
			if (pGraph) { *pGraph = response->pGraph; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphCreate);
	return err;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphAddKernelNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t  node, struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphKernelNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphKernelNodeGetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphKernelNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphKernelNodeSetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t  hSrc, cudaGraphNode_t  hDst)
{
	TALLY_SPD_LOG("cudaGraphKernelNodeCopyAttributes hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphKernelNodeCopyAttributes(hSrc, hDst);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out)
{
	TALLY_SPD_LOG("cudaGraphKernelNodeGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphKernelNodeGetAttribute(hNode, attr, value_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value)
{
	TALLY_SPD_LOG("cudaGraphKernelNodeSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphKernelNodeSetAttribute(hNode, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemcpy3DParms * pCopyParams)
{
	TALLY_SPD_LOG("cudaGraphAddMemcpyNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphAddMemcpyNodeToSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t*  pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t*  pDependencies, size_t  numDependencies, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphAddMemcpyNodeFromSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphAddMemcpyNode1D hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t  node, struct cudaMemcpy3DParms * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphMemcpyNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemcpyNodeGetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphMemcpyNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemcpyNodeSetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphMemcpyNodeSetParamsToSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphMemcpyNodeSetParamsFromSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphMemcpyNodeSetParams1D hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaMemsetParams * pMemsetParams)
{
	TALLY_SPD_LOG("cudaGraphAddMemsetNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t  node, struct cudaMemsetParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphMemsetNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemsetNodeGetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphMemsetNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemsetNodeSetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaHostNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphAddHostNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t  node, struct cudaHostNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphHostNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphHostNodeGetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphHostNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphHostNodeSetParams(node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaGraph_t  childGraph)
{
	TALLY_SPD_LOG("cudaGraphAddChildGraphNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t  node, cudaGraph_t * pGraph)
{
	TALLY_SPD_LOG("cudaGraphChildGraphNodeGetGraph hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphChildGraphNodeGetGraph(node, pGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies)
{
	TALLY_SPD_LOG("cudaGraphAddEmptyNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaGraphAddEventRecordNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t  node, cudaEvent_t * event_out)
{
	TALLY_SPD_LOG("cudaGraphEventRecordNodeGetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphEventRecordNodeGetEvent(node, event_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t  node, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaGraphEventRecordNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphEventRecordNodeSetEvent(node, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaGraphAddEventWaitNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t  node, cudaEvent_t * event_out)
{
	TALLY_SPD_LOG("cudaGraphEventWaitNodeGetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphEventWaitNodeGetEvent(node, event_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t  node, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaGraphEventWaitNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphEventWaitNodeSetEvent(node, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphAddExternalSemaphoresSignalNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t  hNode, struct cudaExternalSemaphoreSignalNodeParams * params_out)
{
	TALLY_SPD_LOG("cudaGraphExternalSemaphoresSignalNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphExternalSemaphoresSignalNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphAddExternalSemaphoresWaitNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t  hNode, struct cudaExternalSemaphoreWaitNodeParams * params_out)
{
	TALLY_SPD_LOG("cudaGraphExternalSemaphoresWaitNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphExternalSemaphoresWaitNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaMemAllocNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphAddMemAllocNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t  node, struct cudaMemAllocNodeParams * params_out)
{
	TALLY_SPD_LOG("cudaGraphMemAllocNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemAllocNodeGetParams(node, params_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, void * dptr)
{
	TALLY_SPD_LOG("cudaGraphAddMemFreeNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t  node, void * dptr_out)
{
	TALLY_SPD_LOG("cudaGraphMemFreeNodeGetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphMemFreeNodeGetParams(node, dptr_out);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceGraphMemTrim(int  device)
{
	TALLY_SPD_LOG("cudaDeviceGraphMemTrim hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceGraphMemTrim(device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceGetGraphMemAttribute(int  device, enum cudaGraphMemAttributeType  attr, void*  value)
{
	TALLY_SPD_LOG("cudaDeviceGetGraphMemAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceGetGraphMemAttribute(device, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaDeviceSetGraphMemAttribute(int  device, enum cudaGraphMemAttributeType  attr, void*  value)
{
	TALLY_SPD_LOG("cudaDeviceSetGraphMemAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudaDeviceSetGraphMemAttribute(device, attr, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t  originalGraph)
{
	TALLY_SPD_LOG("cudaGraphClone hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphClone(pGraphClone, originalGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t  originalNode, cudaGraph_t  clonedGraph)
{
	TALLY_SPD_LOG("cudaGraphNodeFindInClone hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t  node, enum cudaGraphNodeType * pType)
{
	TALLY_SPD_LOG("cudaGraphNodeGetType hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeGetType(node, pType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t  graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes)
{
	TALLY_SPD_LOG("cudaGraphGetRootNodes hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphGetEdges(cudaGraph_t  graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges)
{
	TALLY_SPD_LOG("cudaGraphGetEdges hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphGetEdges(graph, from, to, numEdges);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t  node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies)
{
	TALLY_SPD_LOG("cudaGraphNodeGetDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t  node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes)
{
	TALLY_SPD_LOG("cudaGraphNodeGetDependentNodes hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)
{
	TALLY_SPD_LOG("cudaGraphAddDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddDependencies(graph, from, to, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t  graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t  numDependencies)
{
	TALLY_SPD_LOG("cudaGraphRemoveDependencies hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphRemoveDependencies(graph, from, to, numDependencies);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t  node)
{
	TALLY_SPD_LOG("cudaGraphDestroyNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphDestroyNode(node);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)
{
	TALLY_SPD_LOG("cudaGraphInstantiate hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphInstantiate(pGraphExec, graph, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)
{
	TALLY_SPD_LOG("cudaGraphInstantiateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphInstantiateWithFlags(pGraphExec, graph, flags);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphInstantiateWithFlagsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHINSTANTIATEWITHFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphInstantiateWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pGraphExec = pGraphExec;
			request->graph = graph;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaGraphInstantiateWithFlagsResponse*>(responsePayload);
			if (pGraphExec) { *pGraphExec = response->pGraphExec; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphInstantiateWithFlags);
	return err;
}

cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, cudaGraphInstantiateParams * instantiateParams)
{
	TALLY_SPD_LOG("cudaGraphInstantiateWithParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t  graphExec, unsigned long long * flags)
{
	TALLY_SPD_LOG("cudaGraphExecGetFlags hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecGetFlags(graphExec, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaKernelNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecKernelNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemcpy3DParms * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecMemcpyNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const void*  symbol, const void*  src, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphExecMemcpyNodeSetParamsToSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  symbol, size_t  count, size_t  offset, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphExecMemcpyNodeSetParamsFromSymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, void*  dst, const void*  src, size_t  count, enum cudaMemcpyKind  kind)
{
	TALLY_SPD_LOG("cudaGraphExecMemcpyNodeSetParams1D hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaMemsetParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecMemsetNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, const struct cudaHostNodeParams * pNodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecHostNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  node, cudaGraph_t  childGraph)
{
	TALLY_SPD_LOG("cudaGraphExecChildGraphNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaGraphExecEventRecordNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, cudaEvent_t  event)
{
	TALLY_SPD_LOG("cudaGraphExecEventWaitNodeSetEvent hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreSignalNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecExternalSemaphoresSignalNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, const struct cudaExternalSemaphoreWaitNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecExternalSemaphoresWaitNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int  isEnabled)
{
	TALLY_SPD_LOG("cudaGraphNodeSetEnabled hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int * isEnabled)
{
	TALLY_SPD_LOG("cudaGraphNodeGetEnabled hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t  hGraphExec, cudaGraph_t  hGraph, cudaGraphExecUpdateResultInfo * resultInfo)
{
	TALLY_SPD_LOG("cudaGraphExecUpdate hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecUpdate(hGraphExec, hGraph, resultInfo);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphUpload(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaGraphUpload hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphUpload(graphExec, stream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphUploadArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHUPLOAD;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphUploadArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->graphExec = graphExec;
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphUpload);
	return err;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cudaGraphLaunch hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphLaunch(graphExec, stream);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphLaunchArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHLAUNCH;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphLaunchArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->graphExec = graphExec;
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphLaunch);
	return err;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t  graphExec)
{
	TALLY_SPD_LOG("cudaGraphExecDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphExecDestroy(graphExec);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphExecDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHEXECDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphExecDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->graphExec = graphExec;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphExecDestroy);
	return err;
}

cudaError_t cudaGraphDestroy(cudaGraph_t  graph)
{
	TALLY_SPD_LOG("cudaGraphDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphDestroy(graph);
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGRAPHDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGraphDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->graph = graph;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphDestroy);
	return err;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t  graph, const char * path, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaGraphDebugDotPrint hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphDebugDotPrint(graph, path, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t  destroy, unsigned int  initialRefcount, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaUserObjectCreate hooked");
#if defined(RUN_LOCALLY)
	return lcudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t  object, unsigned int  count)
{
	TALLY_SPD_LOG("cudaUserObjectRetain hooked");
#if defined(RUN_LOCALLY)
	return lcudaUserObjectRetain(object, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t  object, unsigned int  count)
{
	TALLY_SPD_LOG("cudaUserObjectRelease hooked");
#if defined(RUN_LOCALLY)
	return lcudaUserObjectRelease(object, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count, unsigned int  flags)
{
	TALLY_SPD_LOG("cudaGraphRetainUserObject hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphRetainUserObject(graph, object, count, flags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t  graph, cudaUserObject_t  object, unsigned int  count)
{
	TALLY_SPD_LOG("cudaGraphReleaseUserObject hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphReleaseUserObject(graph, object, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphAddNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaGraphNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphAddNode hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphNodeSetParams(node, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t  graphExec, cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams)
{
	TALLY_SPD_LOG("cudaGraphExecNodeSetParams hooked");
#if defined(RUN_LOCALLY)
	return lcudaGraphExecNodeSetParams(graphExec, node, nodeParams);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long  flags, enum cudaDriverEntryPointQueryResult * driverStatus)
{
	TALLY_SPD_LOG("cudaGetDriverEntryPoint hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId)
{
	TALLY_SPD_LOG("cudaGetExportTable hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetExportTable(ppExportTable, pExportTableId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t*  functionPtr, const void*  symbolPtr)
{
	TALLY_SPD_LOG("cudaGetFuncBySymbol hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetFuncBySymbol(functionPtr, symbolPtr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaGetKernel(cudaKernel_t * kernelPtr, const void * entryFuncAddr)
{
	TALLY_SPD_LOG("cudaGetKernel hooked");
#if defined(RUN_LOCALLY)
	return lcudaGetKernel(kernelPtr, entryFuncAddr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

size_t cudnnGetVersion()
{
	TALLY_SPD_LOG("cudnnGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetVersion();
#else

    size_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const size_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetVersion);
	return err;
}

size_t cudnnGetMaxDeviceVersion()
{
	TALLY_SPD_LOG("cudnnGetMaxDeviceVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetMaxDeviceVersion();
#else

    size_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetMaxDeviceVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETMAXDEVICEVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetMaxDeviceVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const size_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetMaxDeviceVersion);
	return err;
}

size_t cudnnGetCudartVersion()
{
	TALLY_SPD_LOG("cudnnGetCudartVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetCudartVersion();
#else

    size_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetCudartVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETCUDARTVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetCudartVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const size_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetCudartVersion);
	return err;
}

cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t  handle, cudnnStatus_t * rstatus, cudnnErrQueryMode_t  mode, cudnnRuntimeTag_t * tag)
{
	TALLY_SPD_LOG("cudnnQueryRuntimeError hooked");
#if defined(RUN_LOCALLY)
	return lcudnnQueryRuntimeError(handle, rstatus, mode, tag);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType  type, int * value)
{
	TALLY_SPD_LOG("cudnnGetProperty hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetProperty(type, value);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetPropertyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETPROPERTY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetPropertyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->type = type;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetPropertyResponse*>(responsePayload);
			if (value) { *value = response->value; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetProperty);
	return err;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t  handle)
{
	TALLY_SPD_LOG("cudnnDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroy(handle);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroy);
	return err;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t  handle, cudaStream_t  streamId)
{
	TALLY_SPD_LOG("cudnnSetStream hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetStream(handle, streamId);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetStreamArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETSTREAM;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetStreamArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->streamId = streamId;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetStream);
	return err;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t  handle, cudaStream_t * streamId)
{
	TALLY_SPD_LOG("cudnnGetStream hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetStream(handle, streamId);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetStreamArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETSTREAM;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetStreamArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->streamId = streamId;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetStreamResponse*>(responsePayload);
			if (streamId) { *streamId = response->streamId; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetStream);
	return err;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t * tensorDesc)
{
	TALLY_SPD_LOG("cudnnCreateTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateTensorDescriptor(tensorDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateTensorDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATETENSORDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateTensorDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->tensorDesc = tensorDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateTensorDescriptorResponse*>(responsePayload);
			if (tensorDesc) { *tensorDesc = response->tensorDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateTensorDescriptor);
	return err;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w)
{
	TALLY_SPD_LOG("cudnnSetTensor4dDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetTensor4dDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetTensor4dDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->tensorDesc = tensorDesc;
			request->format = format;
			request->dataType = dataType;
			request->n = n;
			request->c = c;
			request->h = h;
			request->w = w;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetTensor4dDescriptor);
	return err;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t  dataType, int  n, int  c, int  h, int  w, int  nStride, int  cStride, int  hStride, int  wStride)
{
	TALLY_SPD_LOG("cudnnSetTensor4dDescriptorEx hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetTensor4dDescriptorExArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETTENSOR4DDESCRIPTOREX;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetTensor4dDescriptorExArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->tensorDesc = tensorDesc;
			request->dataType = dataType;
			request->n = n;
			request->c = c;
			request->h = h;
			request->w = w;
			request->nStride = nStride;
			request->cStride = cStride;
			request->hStride = hStride;
			request->wStride = wStride;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetTensor4dDescriptorEx);
	return err;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t  tensorDesc, cudnnDataType_t * dataType, int * n, int * c, int * h, int * w, int * nStride, int * cStride, int * hStride, int * wStride)
{
	TALLY_SPD_LOG("cudnnGetTensor4dDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetTensor4dDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETTENSOR4DDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetTensor4dDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->tensorDesc = tensorDesc;
			request->dataType = dataType;
			request->n = n;
			request->c = c;
			request->h = h;
			request->w = w;
			request->nStride = nStride;
			request->cStride = cStride;
			request->hStride = hStride;
			request->wStride = wStride;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetTensor4dDescriptorResponse*>(responsePayload);
			if (dataType) { *dataType = response->dataType; }
			if (n) { *n = response->n; }
			if (c) { *c = response->c; }
			if (h) { *h = response->h; }
			if (w) { *w = response->w; }
			if (nStride) { *nStride = response->nStride; }
			if (cStride) { *cStride = response->cStride; }
			if (hStride) { *hStride = response->hStride; }
			if (wStride) { *wStride = response->wStride; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetTensor4dDescriptor);
	return err;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t  tensorDesc, cudnnTensorFormat_t  format, cudnnDataType_t  dataType, int  nbDims, const int  dimA[])
{
	TALLY_SPD_LOG("cudnnSetTensorNdDescriptorEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t  tensorDesc, size_t * size)
{
	TALLY_SPD_LOG("cudnnGetTensorSizeInBytes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetTensorSizeInBytes(tensorDesc, size);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetTensorSizeInBytesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETTENSORSIZEINBYTES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetTensorSizeInBytesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->tensorDesc = tensorDesc;
			request->size = size;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetTensorSizeInBytesResponse*>(responsePayload);
			if (size) { *size = response->size; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetTensorSizeInBytes);
	return err;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t  tensorDesc)
{
	TALLY_SPD_LOG("cudnnDestroyTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyTensorDescriptor(tensorDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyTensorDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYTENSORDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyTensorDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->tensorDesc = tensorDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyTensorDescriptor);
	return err;
}

cudnnStatus_t cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t  transformDesc, const cudnnTensorDescriptor_t  srcDesc, cudnnTensorDescriptor_t  destDesc, size_t * destSizeInBytes)
{
	TALLY_SPD_LOG("cudnnInitTransformDest hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnInitTransformDestArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNINITTRANSFORMDEST;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnInitTransformDestArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->transformDesc = transformDesc;
			request->srcDesc = srcDesc;
			request->destDesc = destDesc;
			request->destSizeInBytes = destSizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnInitTransformDestResponse*>(responsePayload);
			if (destSizeInBytes) { *destSizeInBytes = response->destSizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnInitTransformDest);
	return err;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t * transformDesc)
{
	TALLY_SPD_LOG("cudnnCreateTensorTransformDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateTensorTransformDescriptor(transformDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateTensorTransformDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATETENSORTRANSFORMDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateTensorTransformDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->transformDesc = transformDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateTensorTransformDescriptorResponse*>(responsePayload);
			if (transformDesc) { *transformDesc = response->transformDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateTensorTransformDescriptor);
	return err;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc, const uint32_t  nbDims, const cudnnTensorFormat_t  destFormat, const int32_t  padBeforeA[], const int32_t  padAfterA[], const uint32_t  foldA[], const cudnnFoldingDirection_t  direction)
{
	TALLY_SPD_LOG("cudnnSetTensorTransformDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc, uint32_t  nbDimsRequested, cudnnTensorFormat_t * destFormat, int32_t  padBeforeA[], int32_t  padAfterA[], uint32_t  foldA[], cudnnFoldingDirection_t * direction)
{
	TALLY_SPD_LOG("cudnnGetTensorTransformDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t  transformDesc)
{
	TALLY_SPD_LOG("cudnnDestroyTensorTransformDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyTensorTransformDescriptor(transformDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyTensorTransformDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYTENSORTRANSFORMDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyTensorTransformDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->transformDesc = transformDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyTensorTransformDescriptor);
	return err;
}

cudnnStatus_t cudnnTransformTensorEx(cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnTensorDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnTensorDescriptor_t  destDesc, void * destData)
{
	TALLY_SPD_LOG("cudnnTransformTensorEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t * opTensorDesc)
{
	TALLY_SPD_LOG("cudnnCreateOpTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateOpTensorDescriptor(opTensorDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateOpTensorDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATEOPTENSORDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateOpTensorDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->opTensorDesc = opTensorDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateOpTensorDescriptorResponse*>(responsePayload);
			if (opTensorDesc) { *opTensorDesc = response->opTensorDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateOpTensorDescriptor);
	return err;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t  opTensorOp, cudnnDataType_t  opTensorCompType, cudnnNanPropagation_t  opTensorNanOpt)
{
	TALLY_SPD_LOG("cudnnSetOpTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetOpTensorDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETOPTENSORDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetOpTensorDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->opTensorDesc = opTensorDesc;
			request->opTensorOp = opTensorOp;
			request->opTensorCompType = opTensorCompType;
			request->opTensorNanOpt = opTensorNanOpt;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetOpTensorDescriptor);
	return err;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t  opTensorDesc, cudnnOpTensorOp_t * opTensorOp, cudnnDataType_t * opTensorCompType, cudnnNanPropagation_t * opTensorNanOpt)
{
	TALLY_SPD_LOG("cudnnGetOpTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetOpTensorDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETOPTENSORDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetOpTensorDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->opTensorDesc = opTensorDesc;
			request->opTensorOp = opTensorOp;
			request->opTensorCompType = opTensorCompType;
			request->opTensorNanOpt = opTensorNanOpt;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetOpTensorDescriptorResponse*>(responsePayload);
			if (opTensorOp) { *opTensorOp = response->opTensorOp; }
			if (opTensorCompType) { *opTensorCompType = response->opTensorCompType; }
			if (opTensorNanOpt) { *opTensorNanOpt = response->opTensorNanOpt; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetOpTensorDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t  opTensorDesc)
{
	TALLY_SPD_LOG("cudnnDestroyOpTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyOpTensorDescriptor(opTensorDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnOpTensor(cudnnHandle_t  handle, const cudnnOpTensorDescriptor_t  opTensorDesc, const void * alpha1, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * alpha2, const cudnnTensorDescriptor_t  bDesc, const void * B, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	TALLY_SPD_LOG("cudnnOpTensor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t * reduceTensorDesc)
{
	TALLY_SPD_LOG("cudnnCreateReduceTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateReduceTensorDescriptor(reduceTensorDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t  reduceTensorOp, cudnnDataType_t  reduceTensorCompType, cudnnNanPropagation_t  reduceTensorNanOpt, cudnnReduceTensorIndices_t  reduceTensorIndices, cudnnIndicesType_t  reduceTensorIndicesType)
{
	TALLY_SPD_LOG("cudnnSetReduceTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t  reduceTensorDesc, cudnnReduceTensorOp_t * reduceTensorOp, cudnnDataType_t * reduceTensorCompType, cudnnNanPropagation_t * reduceTensorNanOpt, cudnnReduceTensorIndices_t * reduceTensorIndices, cudnnIndicesType_t * reduceTensorIndicesType)
{
	TALLY_SPD_LOG("cudnnGetReduceTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t  reduceTensorDesc)
{
	TALLY_SPD_LOG("cudnnDestroyReduceTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetReductionIndicesSize(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetReductionIndicesSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, const cudnnTensorDescriptor_t  aDesc, const cudnnTensorDescriptor_t  cDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetReductionWorkspaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnReduceTensor(cudnnHandle_t  handle, const cudnnReduceTensorDescriptor_t  reduceTensorDesc, void * indices, size_t  indicesSizeInBytes, void * workspace, size_t  workspaceSizeInBytes, const void * alpha, const cudnnTensorDescriptor_t  aDesc, const void * A, const void * beta, const cudnnTensorDescriptor_t  cDesc, void * C)
{
	TALLY_SPD_LOG("cudnnReduceTensor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetTensor(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * valuePtr)
{
	TALLY_SPD_LOG("cudnnSetTensor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetTensor(handle, yDesc, y, valuePtr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  yDesc, void * y, const void * alpha)
{
	TALLY_SPD_LOG("cudnnScaleTensor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnScaleTensor(handle, yDesc, y, alpha);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t * filterDesc)
{
	TALLY_SPD_LOG("cudnnCreateFilterDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateFilterDescriptor(filterDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateFilterDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATEFILTERDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateFilterDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->filterDesc = filterDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateFilterDescriptorResponse*>(responsePayload);
			if (filterDesc) { *filterDesc = response->filterDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateFilterDescriptor);
	return err;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t  dataType, cudnnTensorFormat_t  format, int  k, int  c, int  h, int  w)
{
	TALLY_SPD_LOG("cudnnSetFilter4dDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetFilter4dDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETFILTER4DDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetFilter4dDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->filterDesc = filterDesc;
			request->dataType = dataType;
			request->format = format;
			request->k = k;
			request->c = c;
			request->h = h;
			request->w = w;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetFilter4dDescriptor);
	return err;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t  filterDesc, cudnnDataType_t * dataType, cudnnTensorFormat_t * format, int * k, int * c, int * h, int * w)
{
	TALLY_SPD_LOG("cudnnGetFilter4dDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t  filterDesc, size_t * size)
{
	TALLY_SPD_LOG("cudnnGetFilterSizeInBytes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetFilterSizeInBytes(filterDesc, size);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetFilterSizeInBytesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETFILTERSIZEINBYTES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetFilterSizeInBytesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->filterDesc = filterDesc;
			request->size = size;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetFilterSizeInBytesResponse*>(responsePayload);
			if (size) { *size = response->size; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetFilterSizeInBytes);
	return err;
}

cudnnStatus_t cudnnTransformFilter(cudnnHandle_t  handle, const cudnnTensorTransformDescriptor_t  transDesc, const void * alpha, const cudnnFilterDescriptor_t  srcDesc, const void * srcData, const void * beta, const cudnnFilterDescriptor_t  destDesc, void * destData)
{
	TALLY_SPD_LOG("cudnnTransformFilter hooked");
#if defined(RUN_LOCALLY)
	return lcudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t  filterDesc)
{
	TALLY_SPD_LOG("cudnnDestroyFilterDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyFilterDescriptor(filterDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyFilterDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYFILTERDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyFilterDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->filterDesc = filterDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyFilterDescriptor);
	return err;
}

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t * poolingDesc)
{
	TALLY_SPD_LOG("cudnnCreatePoolingDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreatePoolingDescriptor(poolingDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreatePoolingDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATEPOOLINGDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreatePoolingDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->poolingDesc = poolingDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreatePoolingDescriptorResponse*>(responsePayload);
			if (poolingDesc) { *poolingDesc = response->poolingDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreatePoolingDescriptor);
	return err;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t  mode, cudnnNanPropagation_t  maxpoolingNanOpt, int  windowHeight, int  windowWidth, int  verticalPadding, int  horizontalPadding, int  verticalStride, int  horizontalStride)
{
	TALLY_SPD_LOG("cudnnSetPooling2dDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t  poolingDesc, cudnnPoolingMode_t * mode, cudnnNanPropagation_t * maxpoolingNanOpt, int * windowHeight, int * windowWidth, int * verticalPadding, int * horizontalPadding, int * verticalStride, int * horizontalStride)
{
	TALLY_SPD_LOG("cudnnGetPooling2dDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t  poolingDesc, const cudnnTensorDescriptor_t  inputTensorDesc, int * n, int * c, int * h, int * w)
{
	TALLY_SPD_LOG("cudnnGetPooling2dForwardOutputDim hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, n, c, h, w);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t  poolingDesc)
{
	TALLY_SPD_LOG("cudnnDestroyPoolingDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyPoolingDescriptor(poolingDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyPoolingDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYPOOLINGDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyPoolingDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->poolingDesc = poolingDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyPoolingDescriptor);
	return err;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t * activationDesc)
{
	TALLY_SPD_LOG("cudnnCreateActivationDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateActivationDescriptor(activationDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateActivationDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATEACTIVATIONDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateActivationDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->activationDesc = activationDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateActivationDescriptorResponse*>(responsePayload);
			if (activationDesc) { *activationDesc = response->activationDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateActivationDescriptor);
	return err;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t  mode, cudnnNanPropagation_t  reluNanOpt, double  coef)
{
	TALLY_SPD_LOG("cudnnSetActivationDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetActivationDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETACTIVATIONDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetActivationDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->activationDesc = activationDesc;
			request->mode = mode;
			request->reluNanOpt = reluNanOpt;
			request->coef = coef;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetActivationDescriptor);
	return err;
}

cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t  activationDesc, cudnnActivationMode_t * mode, cudnnNanPropagation_t * reluNanOpt, double * coef)
{
	TALLY_SPD_LOG("cudnnGetActivationDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t  activationDesc, double  swish_beta)
{
	TALLY_SPD_LOG("cudnnSetActivationDescriptorSwishBeta hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetActivationDescriptorSwishBeta(activationDesc, swish_beta);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t  activationDesc, double * swish_beta)
{
	TALLY_SPD_LOG("cudnnGetActivationDescriptorSwishBeta hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetActivationDescriptorSwishBeta(activationDesc, swish_beta);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t  activationDesc)
{
	TALLY_SPD_LOG("cudnnDestroyActivationDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyActivationDescriptor(activationDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyActivationDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYACTIVATIONDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyActivationDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->activationDesc = activationDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyActivationDescriptor);
	return err;
}

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t * normDesc)
{
	TALLY_SPD_LOG("cudnnCreateLRNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateLRNDescriptor(normDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateLRNDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATELRNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateLRNDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->normDesc = normDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateLRNDescriptorResponse*>(responsePayload);
			if (normDesc) { *normDesc = response->normDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateLRNDescriptor);
	return err;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t  normDesc, unsigned  lrnN, double  lrnAlpha, double  lrnBeta, double  lrnK)
{
	TALLY_SPD_LOG("cudnnSetLRNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetLRNDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETLRNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetLRNDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->normDesc = normDesc;
			request->lrnN = lrnN;
			request->lrnAlpha = lrnAlpha;
			request->lrnBeta = lrnBeta;
			request->lrnK = lrnK;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetLRNDescriptor);
	return err;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t  normDesc, unsigned * lrnN, double * lrnAlpha, double * lrnBeta, double * lrnK)
{
	TALLY_SPD_LOG("cudnnGetLRNDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t  lrnDesc)
{
	TALLY_SPD_LOG("cudnnDestroyLRNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyLRNDescriptor(lrnDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyLRNDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYLRNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyLRNDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->lrnDesc = lrnDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyLRNDescriptor);
	return err;
}

cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_SPD_LOG("cudnnDivisiveNormalizationForward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t  derivedBnDesc, const cudnnTensorDescriptor_t  xDesc, cudnnBatchNormMode_t  mode)
{
	TALLY_SPD_LOG("cudnnDeriveBNTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, const void * estimatedMean, const void * estimatedVariance, double  epsilon)
{
	TALLY_SPD_LOG("cudnnBatchNormalizationForwardInference hooked");
#if defined(RUN_LOCALLY)
	return lcudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDeriveNormTensorDescriptor(cudnnTensorDescriptor_t  derivedNormScaleBiasDesc, cudnnTensorDescriptor_t  derivedNormMeanVarDesc, const cudnnTensorDescriptor_t  xDesc, cudnnNormMode_t  mode, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnDeriveNormTensorDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDeriveNormTensorDescriptor(derivedNormScaleBiasDesc, derivedNormMeanVarDesc, xDesc, mode, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnNormalizationForwardInference(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * estimatedMean, const void * estimatedVariance, const cudnnTensorDescriptor_t  zDesc, const void * z, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y, double  epsilon, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnNormalizationForwardInference hooked");
#if defined(RUN_LOCALLY)
	return lcudnnNormalizationForwardInference(handle, mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, normScale, normBias, normMeanVarDesc, estimatedMean, estimatedVariance, zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t * stDesc)
{
	TALLY_SPD_LOG("cudnnCreateSpatialTransformerDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateSpatialTransformerDescriptor(stDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t  stDesc, cudnnSamplerType_t  samplerType, cudnnDataType_t  dataType, const int  nbDims, const int  dimA[])
{
	TALLY_SPD_LOG("cudnnSetSpatialTransformerNdDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t  stDesc)
{
	TALLY_SPD_LOG("cudnnDestroySpatialTransformerDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroySpatialTransformerDescriptor(stDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * theta, void * grid)
{
	TALLY_SPD_LOG("cudnnSpatialTfGridGeneratorForward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * grid, const void * beta, cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_SPD_LOG("cudnnSpatialTfSamplerForward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t * dropoutDesc)
{
	TALLY_SPD_LOG("cudnnCreateDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateDropoutDescriptor(dropoutDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateDropoutDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATEDROPOUTDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateDropoutDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dropoutDesc = dropoutDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateDropoutDescriptorResponse*>(responsePayload);
			if (dropoutDesc) { *dropoutDesc = response->dropoutDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateDropoutDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc)
{
	TALLY_SPD_LOG("cudnnDestroyDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyDropoutDescriptor(dropoutDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyDropoutDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYDROPOUTDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyDropoutDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dropoutDesc = dropoutDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyDropoutDescriptor);
	return err;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t  handle, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnDropoutGetStatesSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDropoutGetStatesSize(handle, sizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDropoutGetStatesSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDROPOUTGETSTATESSIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDropoutGetStatesSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->sizeInBytes = sizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnDropoutGetStatesSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDropoutGetStatesSize);
	return err;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t  xdesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnDropoutGetReserveSpaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDropoutGetReserveSpaceSize(xdesc, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	TALLY_SPD_LOG("cudnnSetDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetDropoutDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETDROPOUTDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetDropoutDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dropoutDesc = dropoutDesc;
			request->handle = handle;
			request->dropout = dropout;
			request->states = states;
			request->stateSizeInBytes = stateSizeInBytes;
			request->seed = seed;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetDropoutDescriptor);
	return err;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	TALLY_SPD_LOG("cudnnRestoreDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRestoreDropoutDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNRESTOREDROPOUTDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnRestoreDropoutDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dropoutDesc = dropoutDesc;
			request->handle = handle;
			request->dropout = dropout;
			request->states = states;
			request->stateSizeInBytes = stateSizeInBytes;
			request->seed = seed;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnRestoreDropoutDescriptor);
	return err;
}

cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float * dropout, void ** states, unsigned long long * seed)
{
	TALLY_SPD_LOG("cudnnGetDropoutDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetDropoutDescriptor(dropoutDesc, handle, dropout, states, seed);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDropoutForward(cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  xdesc, const void * x, const cudnnTensorDescriptor_t  ydesc, void * y, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnDropoutForward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t * algoDesc)
{
	TALLY_SPD_LOG("cudnnCreateAlgorithmDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateAlgorithmDescriptor(algoDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t  algorithm)
{
	TALLY_SPD_LOG("cudnnSetAlgorithmDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetAlgorithmDescriptor(algoDesc, algorithm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t  algoDesc, cudnnAlgorithm_t * algorithm)
{
	TALLY_SPD_LOG("cudnnGetAlgorithmDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetAlgorithmDescriptor(algoDesc, algorithm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t  src, cudnnAlgorithmDescriptor_t  dest)
{
	TALLY_SPD_LOG("cudnnCopyAlgorithmDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCopyAlgorithmDescriptor(src, dest);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_SPD_LOG("cudnnDestroyAlgorithmDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyAlgorithmDescriptor(algoDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t * algoPerf, int  numberToCreate)
{
	TALLY_SPD_LOG("cudnnCreateAlgorithmPerformance hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateAlgorithmPerformance(algoPerf, numberToCreate);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t  algoDesc, cudnnStatus_t  status, float  time, size_t  memory)
{
	TALLY_SPD_LOG("cudnnSetAlgorithmPerformance hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t  algoPerf, cudnnAlgorithmDescriptor_t * algoDesc, cudnnStatus_t * status, float * time, size_t * memory)
{
	TALLY_SPD_LOG("cudnnGetAlgorithmPerformance hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t * algoPerf, int  numberToDestroy)
{
	TALLY_SPD_LOG("cudnnDestroyAlgorithmPerformance hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyAlgorithmPerformance(algoPerf, numberToDestroy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, size_t * algoSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetAlgorithmSpaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetAlgorithmSpaceSize(handle, algoDesc, algoSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSaveAlgorithm(cudnnHandle_t  handle, cudnnAlgorithmDescriptor_t  algoDesc, void * algoSpace, size_t  algoSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnSaveAlgorithm hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnRestoreAlgorithm(cudnnHandle_t  handle, void * algoSpace, size_t  algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_SPD_LOG("cudnnRestoreAlgorithm hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetCallback(unsigned  mask, void * udata, cudnnCallback_t  fptr)
{
	TALLY_SPD_LOG("cudnnSetCallback hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetCallback(mask, udata, fptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetCallback(unsigned * mask, void ** udata, cudnnCallback_t * fptr)
{
	TALLY_SPD_LOG("cudnnGetCallback hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetCallback(mask, udata, fptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnOpsInferVersionCheck()
{
	TALLY_SPD_LOG("cudnnOpsInferVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnOpsInferVersionCheck();
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnOpsInferVersionCheckArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNOPSINFERVERSIONCHECK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnOpsInferVersionCheckArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnOpsInferVersionCheck);
	return err;
}

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t  handle, cudnnSoftmaxAlgorithm_t  algo, cudnnSoftmaxMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_SPD_LOG("cudnnSoftmaxBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t  handle, const cudnnPoolingDescriptor_t  poolingDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_SPD_LOG("cudnnPoolingBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t  handle, cudnnActivationDescriptor_t  activationDesc, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_SPD_LOG("cudnnActivationBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnLRNCrossChannelBackward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnLRNMode_t  lrnMode, const void * alpha, const cudnnTensorDescriptor_t  yDesc, const void * y, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_SPD_LOG("cudnnLRNCrossChannelBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnHandle_t  handle, cudnnLRNDescriptor_t  normDesc, cudnnDivNormMode_t  mode, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * means, const void * dy, void * temp, void * temp2, const void * beta, const cudnnTensorDescriptor_t  dXdMeansDesc, void * dx, void * dMeans)
{
	TALLY_SPD_LOG("cudnnDivisiveNormalizationBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONFORWARDTRAININGEXWORKSPACESIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->mode = mode;
			request->bnOps = bnOps;
			request->xDesc = xDesc;
			request->zDesc = zDesc;
			request->yDesc = yDesc;
			request->bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc;
			request->activationDesc = activationDesc;
			request->sizeInBytes = sizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
	return err;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetBatchNormalizationBackwardExWorkspaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetBatchNormalizationBackwardExWorkspaceSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONBACKWARDEXWORKSPACESIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetBatchNormalizationBackwardExWorkspaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->mode = mode;
			request->bnOps = bnOps;
			request->xDesc = xDesc;
			request->yDesc = yDesc;
			request->dyDesc = dyDesc;
			request->dzDesc = dzDesc;
			request->dxDesc = dxDesc;
			request->dBnScaleBiasDesc = dBnScaleBiasDesc;
			request->activationDesc = activationDesc;
			request->sizeInBytes = sizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetBatchNormalizationBackwardExWorkspaceSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetBatchNormalizationBackwardExWorkspaceSize);
	return err;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, cudnnBatchNormOps_t  bnOps, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetBatchNormalizationTrainingExReserveSpaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetBatchNormalizationTrainingExReserveSpaceSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETBATCHNORMALIZATIONTRAININGEXRESERVESPACESIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetBatchNormalizationTrainingExReserveSpaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->mode = mode;
			request->bnOps = bnOps;
			request->activationDesc = activationDesc;
			request->xDesc = xDesc;
			request->sizeInBytes = sizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetBatchNormalizationTrainingExReserveSpaceSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
	return err;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc, const void * bnScale, const void * bnBias, double  exponentialAverageFactor, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance)
{
	TALLY_SPD_LOG("cudnnBatchNormalizationForwardTraining hooked");
#if defined(RUN_LOCALLY)
	return lcudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScale, void * dBnScaleResult, void * dBnBiasResult, double  epsilon, const void * savedMean, const void * savedInvVariance)
{
	TALLY_SPD_LOG("cudnnBatchNormalizationBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  zDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  normScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnGetNormalizationForwardTrainingWorkspaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetNormalizationForwardTrainingWorkspaceSize(handle, mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  yDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnTensorDescriptor_t  dzDesc, const cudnnTensorDescriptor_t  dxDesc, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  normMeanVarDesc, size_t * sizeInBytes, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnGetNormalizationBackwardWorkspaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetNormalizationBackwardWorkspaceSize(handle, mode, normOps, algo, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dNormScaleBiasDesc, activationDesc, normMeanVarDesc, sizeInBytes, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnGetNormalizationTrainingReserveSpaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetNormalizationTrainingReserveSpaceSize(handle, mode, normOps, algo, activationDesc, xDesc, sizeInBytes, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnNormalizationForwardTraining(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alpha, const void * beta, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  normScaleBiasDesc, const void * normScale, const void * normBias, double  exponentialAverageFactor, const cudnnTensorDescriptor_t  normMeanVarDesc, void * resultRunningMean, void * resultRunningVariance, double  epsilon, void * resultSaveMean, void * resultSaveInvVariance, cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  zDesc, const void * zData, const cudnnTensorDescriptor_t  yDesc, void * yData, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnNormalizationForwardTraining hooked");
#if defined(RUN_LOCALLY)
	return lcudnnNormalizationForwardTraining(handle, mode, normOps, algo, alpha, beta, xDesc, xData, normScaleBiasDesc, normScale, normBias, exponentialAverageFactor, normMeanVarDesc, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, zDesc, zData, yDesc, yData, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnNormalizationBackward(cudnnHandle_t  handle, cudnnNormMode_t  mode, cudnnNormOps_t  normOps, cudnnNormAlgo_t  algo, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * xData, const cudnnTensorDescriptor_t  yDesc, const void * yData, const cudnnTensorDescriptor_t  dyDesc, const void * dyData, const cudnnTensorDescriptor_t  dzDesc, void * dzData, const cudnnTensorDescriptor_t  dxDesc, void * dxData, const cudnnTensorDescriptor_t  dNormScaleBiasDesc, const void * normScaleData, const void * normBiasData, void * dNormScaleData, void * dNormBiasData, double  epsilon, const cudnnTensorDescriptor_t  normMeanVarDesc, const void * savedMean, const void * savedInvVariance, cudnnActivationDescriptor_t  activationDesc, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes, int  groupCnt)
{
	TALLY_SPD_LOG("cudnnNormalizationBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnNormalizationBackward(handle, mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dNormScaleBiasDesc, normScaleData, normBiasData, dNormScaleData, dNormBiasData, epsilon, normMeanVarDesc, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes, groupCnt);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t  handle, const cudnnSpatialTransformerDescriptor_t  stDesc, const void * dgrid, void * dtheta)
{
	TALLY_SPD_LOG("cudnnSpatialTfGridGeneratorBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t  handle, cudnnSpatialTransformerDescriptor_t  stDesc, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx, const void * alphaDgrid, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * grid, const void * betaDgrid, void * dgrid)
{
	TALLY_SPD_LOG("cudnnSpatialTfSamplerBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t  handle, const cudnnDropoutDescriptor_t  dropoutDesc, const cudnnTensorDescriptor_t  dydesc, const void * dy, const cudnnTensorDescriptor_t  dxdesc, void * dx, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnDropoutBackward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnOpsTrainVersionCheck()
{
	TALLY_SPD_LOG("cudnnOpsTrainVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnOpsTrainVersionCheck();
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnOpsTrainVersionCheckArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNOPSTRAINVERSIONCHECK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnOpsTrainVersionCheckArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnOpsTrainVersionCheck);
	return err;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t * rnnDesc)
{
	TALLY_SPD_LOG("cudnnCreateRNNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateRNNDescriptor(rnnDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateRNNDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATERNNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateRNNDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateRNNDescriptorResponse*>(responsePayload);
			if (rnnDesc) { *rnnDesc = response->rnnDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateRNNDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t  rnnDesc)
{
	TALLY_SPD_LOG("cudnnDestroyRNNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyRNNDescriptor(rnnDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyRNNDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYRNNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyRNNDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyRNNDescriptor);
	return err;
}

cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t  algo, cudnnRNNMode_t  cellMode, cudnnRNNBiasMode_t  biasMode, cudnnDirectionMode_t  dirMode, cudnnRNNInputMode_t  inputMode, cudnnDataType_t  dataType, cudnnDataType_t  mathPrec, cudnnMathType_t  mathType, int32_t  inputSize, int32_t  hiddenSize, int32_t  projSize, int32_t  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, uint32_t  auxFlags)
{
	TALLY_SPD_LOG("cudnnSetRNNDescriptor_v8 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNDescriptor_v8Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V8;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetRNNDescriptor_v8Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;
			request->algo = algo;
			request->cellMode = cellMode;
			request->biasMode = biasMode;
			request->dirMode = dirMode;
			request->inputMode = inputMode;
			request->dataType = dataType;
			request->mathPrec = mathPrec;
			request->mathType = mathType;
			request->inputSize = inputSize;
			request->hiddenSize = hiddenSize;
			request->projSize = projSize;
			request->numLayers = numLayers;
			request->dropoutDesc = dropoutDesc;
			request->auxFlags = auxFlags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNDescriptor_v8);
	return err;
}

cudnnStatus_t cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNAlgo_t * algo, cudnnRNNMode_t * cellMode, cudnnRNNBiasMode_t * biasMode, cudnnDirectionMode_t * dirMode, cudnnRNNInputMode_t * inputMode, cudnnDataType_t * dataType, cudnnDataType_t * mathPrec, cudnnMathType_t * mathType, int32_t * inputSize, int32_t * hiddenSize, int32_t * projSize, int32_t * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, uint32_t * auxFlags)
{
	TALLY_SPD_LOG("cudnnGetRNNDescriptor_v8 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  hiddenSize, const int  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, cudnnRNNInputMode_t  inputMode, cudnnDirectionMode_t  direction, cudnnRNNMode_t  cellMode, cudnnRNNAlgo_t  algo, cudnnDataType_t  mathPrec)
{
	TALLY_SPD_LOG("cudnnSetRNNDescriptor_v6 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNDescriptor_v6Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETRNNDESCRIPTOR_V6;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetRNNDescriptor_v6Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->hiddenSize = hiddenSize;
			request->numLayers = numLayers;
			request->dropoutDesc = dropoutDesc;
			request->inputMode = inputMode;
			request->direction = direction;
			request->cellMode = cellMode;
			request->algo = algo;
			request->mathPrec = mathPrec;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNDescriptor_v6);
	return err;
}

cudnnStatus_t cudnnGetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int * hiddenSize, int * numLayers, cudnnDropoutDescriptor_t * dropoutDesc, cudnnRNNInputMode_t * inputMode, cudnnDirectionMode_t * direction, cudnnRNNMode_t * cellMode, cudnnRNNAlgo_t * algo, cudnnDataType_t * mathPrec)
{
	TALLY_SPD_LOG("cudnnGetRNNDescriptor_v6 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t  mType)
{
	TALLY_SPD_LOG("cudnnSetRNNMatrixMathType hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNMatrixMathType(rnnDesc, mType);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNMatrixMathTypeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETRNNMATRIXMATHTYPE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetRNNMatrixMathTypeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;
			request->mType = mType;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNMatrixMathType);
	return err;
}

cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t * mType)
{
	TALLY_SPD_LOG("cudnnGetRNNMatrixMathType hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNMatrixMathType(rnnDesc, mType);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNMatrixMathTypeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNMATRIXMATHTYPE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNMatrixMathTypeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;
			request->mType = mType;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNMatrixMathTypeResponse*>(responsePayload);
			if (mType) { *mType = response->mType; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNMatrixMathType);
	return err;
}

cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t  biasMode)
{
	TALLY_SPD_LOG("cudnnSetRNNBiasMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNBiasMode(rnnDesc, biasMode);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNBiasModeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETRNNBIASMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetRNNBiasModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;
			request->biasMode = biasMode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNBiasMode);
	return err;
}

cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNBiasMode_t * biasMode)
{
	TALLY_SPD_LOG("cudnnGetRNNBiasMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNBiasMode(rnnDesc, biasMode);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNBiasModeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNBIASMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNBiasModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;
			request->biasMode = biasMode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNBiasModeResponse*>(responsePayload);
			if (biasMode) { *biasMode = response->biasMode; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNBiasMode);
	return err;
}

cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	TALLY_SPD_LOG("cudnnRNNSetClip_v8 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNSetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNSetClip_v8Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNRNNSETCLIP_V8;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnRNNSetClip_v8Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDesc = rnnDesc;
			request->clipMode = clipMode;
			request->clipNanOpt = clipNanOpt;
			request->lclip = lclip;
			request->rclip = rclip;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnRNNSetClip_v8);
	return err;
}

cudnnStatus_t cudnnRNNGetClip_v8(cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)
{
	TALLY_SPD_LOG("cudnnRNNGetClip_v8 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNGetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	TALLY_SPD_LOG("cudnnRNNSetClip hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNSetClipArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNRNNSETCLIP;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnRNNSetClipArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->clipMode = clipMode;
			request->clipNanOpt = clipNanOpt;
			request->lclip = lclip;
			request->rclip = rclip;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnRNNSetClip);
	return err;
}

cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t * clipMode, cudnnNanPropagation_t * clipNanOpt, double * lclip, double * rclip)
{
	TALLY_SPD_LOG("cudnnRNNGetClip hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNGetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  recProjSize, const int  outProjSize)
{
	TALLY_SPD_LOG("cudnnSetRNNProjectionLayers hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * recProjSize, int * outProjSize)
{
	TALLY_SPD_LOG("cudnnGetRNNProjectionLayers hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t  rnnDesc, const int  minibatch, const cudnnDataType_t  dataType, cudnnPersistentRNNPlan_t * plan)
{
	TALLY_SPD_LOG("cudnnCreatePersistentRNNPlan hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, plan);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t  plan)
{
	TALLY_SPD_LOG("cudnnDestroyPersistentRNNPlan hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyPersistentRNNPlan(plan);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t  rnnDesc, cudnnPersistentRNNPlan_t  plan)
{
	TALLY_SPD_LOG("cudnnSetPersistentRNNPlan hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetPersistentRNNPlan(rnnDesc, plan);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnBuildRNNDynamic(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int  miniBatch)
{
	TALLY_SPD_LOG("cudnnBuildRNNDynamic hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBuildRNNDynamic(handle, rnnDesc, miniBatch);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBuildRNNDynamicArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNBUILDRNNDYNAMIC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnBuildRNNDynamicArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->miniBatch = miniBatch;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnBuildRNNDynamic);
	return err;
}

cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnForwardMode_t  fwdMode, cudnnRNNDataDescriptor_t  xDesc, size_t * workSpaceSize, size_t * reserveSpaceSize)
{
	TALLY_SPD_LOG("cudnnGetRNNTempSpaceSizes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNTempSpaceSizes(handle, rnnDesc, fwdMode, xDesc, workSpaceSize, reserveSpaceSize);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNTempSpaceSizesArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNTEMPSPACESIZES;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNTempSpaceSizesArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->fwdMode = fwdMode;
			request->xDesc = xDesc;
			request->workSpaceSize = workSpaceSize;
			request->reserveSpaceSize = reserveSpaceSize;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNTempSpaceSizesResponse*>(responsePayload);
			if (workSpaceSize) { *workSpaceSize = response->workSpaceSize; }
			if (reserveSpaceSize) { *reserveSpaceSize = response->reserveSpaceSize; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNTempSpaceSizes);
	return err;
}

cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnTensorDescriptor_t  xDesc, size_t * sizeInBytes, cudnnDataType_t  dataType)
{
	TALLY_SPD_LOG("cudnnGetRNNParamsSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNParamsSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNPARAMSSIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNParamsSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->xDesc = xDesc;
			request->sizeInBytes = sizeInBytes;
			request->dataType = dataType;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNParamsSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNParamsSize);
	return err;
}

cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, size_t * weightSpaceSize)
{
	TALLY_SPD_LOG("cudnnGetRNNWeightSpaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNWeightSpaceSize(handle, rnnDesc, weightSpaceSize);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNWeightSpaceSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNWEIGHTSPACESIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNWeightSpaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->weightSpaceSize = weightSpaceSize;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNWeightSpaceSizeResponse*>(responsePayload);
			if (weightSpaceSize) { *weightSpaceSize = response->weightSpaceSize; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNWeightSpaceSize);
	return err;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerMatDesc, void ** linLayerMat)
{
	TALLY_SPD_LOG("cudnnGetRNNLinLayerMatrixParams hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNLinLayerMatrixParamsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNLINLAYERMATRIXPARAMS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNLinLayerMatrixParamsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->pseudoLayer = pseudoLayer;
			request->xDesc = xDesc;
			request->wDesc = wDesc;
			request->w = const_cast<void *>(w);
			request->linLayerID = linLayerID;
			request->linLayerMatDesc = linLayerMatDesc;
			request->linLayerMat = linLayerMat;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNLinLayerMatrixParamsResponse*>(responsePayload);
			if (linLayerMat) { *linLayerMat = response->linLayerMat; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNLinLayerMatrixParams);
	return err;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  pseudoLayer, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const void * w, const int  linLayerID, cudnnFilterDescriptor_t  linLayerBiasDesc, void ** linLayerBias)
{
	TALLY_SPD_LOG("cudnnGetRNNLinLayerBiasParams hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNLinLayerBiasParamsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNLINLAYERBIASPARAMS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNLinLayerBiasParamsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->pseudoLayer = pseudoLayer;
			request->xDesc = xDesc;
			request->wDesc = wDesc;
			request->w = const_cast<void *>(w);
			request->linLayerID = linLayerID;
			request->linLayerBiasDesc = linLayerBiasDesc;
			request->linLayerBias = linLayerBias;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNLinLayerBiasParamsResponse*>(responsePayload);
			if (linLayerBias) { *linLayerBias = response->linLayerBias; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNLinLayerBiasParams);
	return err;
}

cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, int32_t  pseudoLayer, size_t  weightSpaceSize, const void * weightSpace, int32_t  linLayerID, cudnnTensorDescriptor_t  mDesc, void ** mAddr, cudnnTensorDescriptor_t  bDesc, void ** bAddr)
{
	TALLY_SPD_LOG("cudnnGetRNNWeightParams hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNWeightParams(handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNWeightParamsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNWEIGHTPARAMS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNWeightParamsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->pseudoLayer = pseudoLayer;
			request->weightSpaceSize = weightSpaceSize;
			request->weightSpace = const_cast<void *>(weightSpace);
			request->linLayerID = linLayerID;
			request->mDesc = mDesc;
			request->mAddr = mAddr;
			request->bDesc = bDesc;
			request->bAddr = bAddr;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNWeightParamsResponse*>(responsePayload);
			if (mAddr) { *mAddr = response->mAddr; }
			if (bAddr) { *bAddr = response->bAddr; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNWeightParams);
	return err;
}

cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnRNNForwardInference hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t  rnnDesc, unsigned  paddingMode)
{
	TALLY_SPD_LOG("cudnnSetRNNPaddingMode hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetRNNPaddingMode(rnnDesc, paddingMode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t  rnnDesc, unsigned * paddingMode)
{
	TALLY_SPD_LOG("cudnnGetRNNPaddingMode hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNPaddingMode(rnnDesc, paddingMode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t * rnnDataDesc)
{
	TALLY_SPD_LOG("cudnnCreateRNNDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateRNNDataDescriptor(rnnDataDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateRNNDataDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATERNNDATADESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateRNNDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDataDesc = rnnDataDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateRNNDataDescriptorResponse*>(responsePayload);
			if (rnnDataDesc) { *rnnDataDesc = response->rnnDataDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateRNNDataDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc)
{
	TALLY_SPD_LOG("cudnnDestroyRNNDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyRNNDataDescriptor(rnnDataDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyRNNDataDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYRNNDATADESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyRNNDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->rnnDataDesc = rnnDataDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyRNNDataDescriptor);
	return err;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t  rnnDataDesc, cudnnDataType_t * dataType, cudnnRNNDataLayout_t * layout, int * maxSeqLength, int * batchSize, int * vectorSize, int  arrayLengthRequested, int  seqLengthArray[], void * paddingFill)
{
	TALLY_SPD_LOG("cudnnGetRNNDataDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnRNNForwardInferenceEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_SPD_LOG("cudnnSetRNNAlgorithmDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNAlgorithmDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETRNNALGORITHMDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetRNNAlgorithmDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->algoDesc = algoDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetRNNAlgorithmDescriptor);
	return err;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_SPD_LOG("cudnnGetRNNForwardInferenceAlgorithmMaxCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNForwardInferenceAlgorithmMaxCountArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETRNNFORWARDINFERENCEALGORITHMMAXCOUNT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetRNNForwardInferenceAlgorithmMaxCountArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->rnnDesc = rnnDesc;
			request->count = count;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetRNNForwardInferenceAlgorithmMaxCountResponse*>(responsePayload);
			if (count) { *count = response->count; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetRNNForwardInferenceAlgorithmMaxCount);
	return err;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindRNNForwardInferenceAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t * seqDataDesc)
{
	TALLY_SPD_LOG("cudnnCreateSeqDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateSeqDataDescriptor(seqDataDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateSeqDataDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATESEQDATADESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateSeqDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->seqDataDesc = seqDataDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateSeqDataDescriptorResponse*>(responsePayload);
			if (seqDataDesc) { *seqDataDesc = response->seqDataDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateSeqDataDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t  seqDataDesc)
{
	TALLY_SPD_LOG("cudnnDestroySeqDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroySeqDataDescriptor(seqDataDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroySeqDataDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYSEQDATADESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroySeqDataDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->seqDataDesc = seqDataDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroySeqDataDescriptor);
	return err;
}

cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t * attnDesc)
{
	TALLY_SPD_LOG("cudnnCreateAttnDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateAttnDescriptor(attnDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateAttnDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATEATTNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateAttnDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->attnDesc = attnDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateAttnDescriptorResponse*>(responsePayload);
			if (attnDesc) { *attnDesc = response->attnDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateAttnDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t  attnDesc)
{
	TALLY_SPD_LOG("cudnnDestroyAttnDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyAttnDescriptor(attnDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyAttnDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYATTNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyAttnDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->attnDesc = attnDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyAttnDescriptor);
	return err;
}

cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnDescriptor_t  attnDesc, unsigned  attnMode, int  nHeads, double  smScaler, cudnnDataType_t  dataType, cudnnDataType_t  computePrec, cudnnMathType_t  mathType, cudnnDropoutDescriptor_t  attnDropoutDesc, cudnnDropoutDescriptor_t  postDropoutDesc, int  qSize, int  kSize, int  vSize, int  qProjSize, int  kProjSize, int  vProjSize, int  oProjSize, int  qoMaxSeqLength, int  kvMaxSeqLength, int  maxBatchSize, int  maxBeamSize)
{
	TALLY_SPD_LOG("cudnnSetAttnDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetAttnDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNSETATTNDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnSetAttnDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->attnDesc = attnDesc;
			request->attnMode = attnMode;
			request->nHeads = nHeads;
			request->smScaler = smScaler;
			request->dataType = dataType;
			request->computePrec = computePrec;
			request->mathType = mathType;
			request->attnDropoutDesc = attnDropoutDesc;
			request->postDropoutDesc = postDropoutDesc;
			request->qSize = qSize;
			request->kSize = kSize;
			request->vSize = vSize;
			request->qProjSize = qProjSize;
			request->kProjSize = kProjSize;
			request->vProjSize = vProjSize;
			request->oProjSize = oProjSize;
			request->qoMaxSeqLength = qoMaxSeqLength;
			request->kvMaxSeqLength = kvMaxSeqLength;
			request->maxBatchSize = maxBatchSize;
			request->maxBeamSize = maxBeamSize;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnSetAttnDescriptor);
	return err;
}

cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnDescriptor_t  attnDesc, unsigned * attnMode, int * nHeads, double * smScaler, cudnnDataType_t * dataType, cudnnDataType_t * computePrec, cudnnMathType_t * mathType, cudnnDropoutDescriptor_t * attnDropoutDesc, cudnnDropoutDescriptor_t * postDropoutDesc, int * qSize, int * kSize, int * vSize, int * qProjSize, int * kProjSize, int * vProjSize, int * oProjSize, int * qoMaxSeqLength, int * kvMaxSeqLength, int * maxBatchSize, int * maxBeamSize)
{
	TALLY_SPD_LOG("cudnnGetAttnDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, size_t * weightSizeInBytes, size_t * workSpaceSizeInBytes, size_t * reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetMultiHeadAttnBuffers hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetMultiHeadAttnBuffersArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETMULTIHEADATTNBUFFERS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetMultiHeadAttnBuffersArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->attnDesc = attnDesc;
			request->weightSizeInBytes = weightSizeInBytes;
			request->workSpaceSizeInBytes = workSpaceSizeInBytes;
			request->reserveSpaceSizeInBytes = reserveSpaceSizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetMultiHeadAttnBuffersResponse*>(responsePayload);
			if (weightSizeInBytes) { *weightSizeInBytes = response->weightSizeInBytes; }
			if (workSpaceSizeInBytes) { *workSpaceSizeInBytes = response->workSpaceSizeInBytes; }
			if (reserveSpaceSizeInBytes) { *reserveSpaceSizeInBytes = response->reserveSpaceSizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetMultiHeadAttnBuffers);
	return err;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, cudnnMultiHeadAttnWeightKind_t  wKind, size_t  weightSizeInBytes, const void * weights, cudnnTensorDescriptor_t  wDesc, void ** wAddr)
{
	TALLY_SPD_LOG("cudnnGetMultiHeadAttnWeights hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnAdvInferVersionCheck()
{
	TALLY_SPD_LOG("cudnnAdvInferVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnAdvInferVersionCheck();
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnAdvInferVersionCheckArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNADVINFERVERSIONCHECK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnAdvInferVersionCheckArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnAdvInferVersionCheck);
	return err;
}

cudnnStatus_t cudnnRNNForwardTrainingEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnRNNForwardTrainingEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnRNNBackwardDataEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  yDesc, const void * y, const cudnnRNNDataDescriptor_t  dyDesc, const void * dy, const cudnnRNNDataDescriptor_t  dcDesc, const void * dcAttn, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnRNNDataDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const cudnnRNNDataDescriptor_t  dkDesc, void * dkeys, void * workSpace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnRNNBackwardDataEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnRNNDataDescriptor_t  yDesc, const void * y, void * workSpace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnRNNBackwardWeightsEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_SPD_LOG("cudnnGetRNNForwardTrainingAlgorithmMaxCount hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t * yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindRNNForwardTrainingAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_SPD_LOG("cudnnGetRNNBackwardDataAlgorithmMaxCount hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * yDesc, const void * y, const cudnnTensorDescriptor_t * dyDesc, const void * dy, const cudnnTensorDescriptor_t  dhyDesc, const void * dhy, const cudnnTensorDescriptor_t  dcyDesc, const void * dcy, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnTensorDescriptor_t * dxDesc, void * dx, const cudnnTensorDescriptor_t  dhxDesc, void * dhx, const cudnnTensorDescriptor_t  dcxDesc, void * dcx, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, void * workspace, size_t  workSpaceSizeInBytes, void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindRNNBackwardDataAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, int * count)
{
	TALLY_SPD_LOG("cudnnGetRNNBackwardWeightsAlgorithmMaxCount hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const int  seqLength, const cudnnTensorDescriptor_t * xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t * yDesc, const void * y, const float  findIntensity, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnAlgorithmPerformance_t * perfResults, const void * workspace, size_t  workSpaceSizeInBytes, const cudnnFilterDescriptor_t  dwDesc, void * dw, const void * reserveSpace, size_t  reserveSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindRNNBackwardWeightsAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t * ctcLossDesc)
{
	TALLY_SPD_LOG("cudnnCreateCTCLossDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateCTCLossDescriptor(ctcLossDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType)
{
	TALLY_SPD_LOG("cudnnSetCTCLossDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetCTCLossDescriptor(ctcLossDesc, compType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode)
{
	TALLY_SPD_LOG("cudnnSetCTCLossDescriptorEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t  compType, cudnnLossNormalizationMode_t  normMode, cudnnNanPropagation_t  gradMode, int  maxLabelLength)
{
	TALLY_SPD_LOG("cudnnSetCTCLossDescriptor_v8 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType)
{
	TALLY_SPD_LOG("cudnnGetCTCLossDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetCTCLossDescriptor(ctcLossDesc, compType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode)
{
	TALLY_SPD_LOG("cudnnGetCTCLossDescriptorEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t  ctcLossDesc, cudnnDataType_t * compType, cudnnLossNormalizationMode_t * normMode, cudnnNanPropagation_t * gradMode, int * maxLabelLength)
{
	TALLY_SPD_LOG("cudnnGetCTCLossDescriptor_v8 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetCTCLossDescriptor_v8(ctcLossDesc, compType, normMode, gradMode, maxLabelLength);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t  ctcLossDesc)
{
	TALLY_SPD_LOG("cudnnDestroyCTCLossDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyCTCLossDescriptor(ctcLossDesc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCTCLoss(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  hostLabels[], const int  hostLabelLengths[], const int  hostInputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, void * workspace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnCTCLoss hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCTCLoss(handle, probsDesc, probs, hostLabels, hostLabelLengths, hostInputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const void * probs, const int  labels[], const int  labelLengths[], const int  inputLengths[], void * costs, const cudnnTensorDescriptor_t  gradientsDesc, void * gradients, size_t  workSpaceSizeInBytes, void * workspace)
{
	TALLY_SPD_LOG("cudnnCTCLoss_v8 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCTCLoss_v8(handle, algo, ctcLossDesc, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients, workSpaceSizeInBytes, workspace);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, const int * labels, const int * labelLengths, const int * inputLengths, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetCTCLossWorkspaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t  handle, cudnnCTCLossAlgo_t  algo, cudnnCTCLossDescriptor_t  ctcLossDesc, const cudnnTensorDescriptor_t  probsDesc, const cudnnTensorDescriptor_t  gradientsDesc, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetCTCLossWorkspaceSize_v8 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetCTCLossWorkspaceSize_v8(handle, algo, ctcLossDesc, probsDesc, gradientsDesc, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnAdvTrainVersionCheck()
{
	TALLY_SPD_LOG("cudnnAdvTrainVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnAdvTrainVersionCheck();
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnAdvTrainVersionCheckArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNADVTRAINVERSIONCHECK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnAdvTrainVersionCheckArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnAdvTrainVersionCheck);
	return err;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t * convDesc)
{
	TALLY_SPD_LOG("cudnnCreateConvolutionDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateConvolutionDescriptor(convDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateConvolutionDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCREATECONVOLUTIONDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCreateConvolutionDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->convDesc = convDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnCreateConvolutionDescriptorResponse*>(responsePayload);
			if (convDesc) { *convDesc = response->convDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCreateConvolutionDescriptor);
	return err;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t  convDesc)
{
	TALLY_SPD_LOG("cudnnDestroyConvolutionDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyConvolutionDescriptor(convDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyConvolutionDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNDESTROYCONVOLUTIONDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnDestroyConvolutionDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->convDesc = convDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnDestroyConvolutionDescriptor);
	return err;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t  mathType)
{
	TALLY_SPD_LOG("cudnnSetConvolutionMathType hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetConvolutionMathType(convDesc, mathType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t  convDesc, cudnnMathType_t * mathType)
{
	TALLY_SPD_LOG("cudnnGetConvolutionMathType hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionMathType(convDesc, mathType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t  convDesc, int  groupCount)
{
	TALLY_SPD_LOG("cudnnSetConvolutionGroupCount hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetConvolutionGroupCount(convDesc, groupCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t  convDesc, int * groupCount)
{
	TALLY_SPD_LOG("cudnnGetConvolutionGroupCount hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionGroupCount(convDesc, groupCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t  reorderType)
{
	TALLY_SPD_LOG("cudnnSetConvolutionReorderType hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetConvolutionReorderType(convDesc, reorderType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t  convDesc, cudnnReorderType_t * reorderType)
{
	TALLY_SPD_LOG("cudnnGetConvolutionReorderType hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionReorderType(convDesc, reorderType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t  convDesc, int  pad_h, int  pad_w, int  u, int  v, int  dilation_h, int  dilation_w, cudnnConvolutionMode_t  mode, cudnnDataType_t  computeType)
{
	TALLY_SPD_LOG("cudnnSetConvolution2dDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t  convDesc, int * pad_h, int * pad_w, int * u, int * v, int * dilation_h, int * dilation_w, cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)
{
	TALLY_SPD_LOG("cudnnGetConvolution2dDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t  convDesc, int  arrayLengthRequested, int * arrayLength, int  padA[], int  strideA[], int  dilationA[], cudnnConvolutionMode_t * mode, cudnnDataType_t * computeType)
{
	TALLY_SPD_LOG("cudnnGetConvolutionNdDescriptor hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  inputTensorDesc, const cudnnFilterDescriptor_t  filterDesc, int * n, int * c, int * h, int * w)
{
	TALLY_SPD_LOG("cudnnGetConvolution2dForwardOutputDim hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_SPD_LOG("cudnnGetConvolutionForwardAlgorithmMaxCount hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionForwardAlgorithmMaxCount(handle, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, void * y, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindConvolutionForwardAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnIm2Col(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, void * colBuffer)
{
	TALLY_SPD_LOG("cudnnIm2Col hooked");
#if defined(RUN_LOCALLY)
	return lcudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnFilterDescriptor_t  wDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  yDesc, cudnnConvolutionFwdAlgo_t  algo, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetConvolutionForwardWorkspaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionForwardWorkspaceSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONFORWARDWORKSPACESIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetConvolutionForwardWorkspaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->xDesc = xDesc;
			request->wDesc = wDesc;
			request->convDesc = convDesc;
			request->yDesc = yDesc;
			request->algo = algo;
			request->sizeInBytes = sizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetConvolutionForwardWorkspaceSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionForwardWorkspaceSize);
	return err;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward(cudnnHandle_t  handle, const void * alpha1, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionFwdAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * alpha2, const cudnnTensorDescriptor_t  zDesc, const void * z, const cudnnTensorDescriptor_t  biasDesc, const void * bias, const cudnnActivationDescriptor_t  activationDesc, const cudnnTensorDescriptor_t  yDesc, void * y)
{
	TALLY_SPD_LOG("cudnnConvolutionBiasActivationForward hooked");
#if defined(RUN_LOCALLY)
	return lcudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_SPD_LOG("cudnnGetConvolutionBackwardDataAlgorithmMaxCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionBackwardDataAlgorithmMaxCountArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAALGORITHMMAXCOUNT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetConvolutionBackwardDataAlgorithmMaxCountArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->count = count;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetConvolutionBackwardDataAlgorithmMaxCountResponse*>(responsePayload);
			if (count) { *count = response->count; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
	return err;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)
{
	TALLY_SPD_LOG("cudnnFindConvolutionBackwardDataAlgorithm hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, void * dx, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindConvolutionBackwardDataAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t * perfResults)
{
	TALLY_SPD_LOG("cudnnGetConvolutionBackwardDataAlgorithm_v7 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t  handle, const cudnnFilterDescriptor_t  wDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  dxDesc, cudnnConvolutionBwdDataAlgo_t  algo, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetConvolutionBackwardDataWorkspaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionBackwardDataWorkspaceSizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDDATAWORKSPACESIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetConvolutionBackwardDataWorkspaceSizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->wDesc = wDesc;
			request->dyDesc = dyDesc;
			request->convDesc = convDesc;
			request->dxDesc = dxDesc;
			request->algo = algo;
			request->sizeInBytes = sizeInBytes;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetConvolutionBackwardDataWorkspaceSizeResponse*>(responsePayload);
			if (sizeInBytes) { *sizeInBytes = response->sizeInBytes; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize);
	return err;
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t  handle, const void * alpha, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdDataAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnTensorDescriptor_t  dxDesc, void * dx)
{
	TALLY_SPD_LOG("cudnnConvolutionBackwardData hooked");
#if defined(RUN_LOCALLY)
	return lcudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const cudnnTensorFormat_t  transformFormat, cudnnFilterDescriptor_t  foldedFilterDesc, cudnnTensorDescriptor_t  paddedDiffDesc, cudnnConvolutionDescriptor_t  foldedConvDesc, cudnnTensorDescriptor_t  foldedGradDesc, cudnnTensorTransformDescriptor_t  filterFoldTransDesc, cudnnTensorTransformDescriptor_t  diffPadTransDesc, cudnnTensorTransformDescriptor_t  gradFoldTransDesc, cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc)
{
	TALLY_SPD_LOG("cudnnGetFoldedConvBackwardDataDescriptors hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetFoldedConvBackwardDataDescriptorsArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETFOLDEDCONVBACKWARDDATADESCRIPTORS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetFoldedConvBackwardDataDescriptorsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->filterDesc = filterDesc;
			request->diffDesc = diffDesc;
			request->convDesc = convDesc;
			request->gradDesc = gradDesc;
			request->transformFormat = transformFormat;
			request->foldedFilterDesc = foldedFilterDesc;
			request->paddedDiffDesc = paddedDiffDesc;
			request->foldedConvDesc = foldedConvDesc;
			request->foldedGradDesc = foldedGradDesc;
			request->filterFoldTransDesc = filterFoldTransDesc;
			request->diffPadTransDesc = diffPadTransDesc;
			request->gradFoldTransDesc = gradFoldTransDesc;
			request->gradUnfoldTransDesc = gradUnfoldTransDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetFoldedConvBackwardDataDescriptors);
	return err;
}

cudnnStatus_t cudnnCnnInferVersionCheck()
{
	TALLY_SPD_LOG("cudnnCnnInferVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCnnInferVersionCheck();
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCnnInferVersionCheckArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCNNINFERVERSIONCHECK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCnnInferVersionCheckArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCnnInferVersionCheck);
	return err;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_SPD_LOG("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionBackwardFilterAlgorithmMaxCountArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNGETCONVOLUTIONBACKWARDFILTERALGORITHMMAXCOUNT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnGetConvolutionBackwardFilterAlgorithmMaxCountArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->count = count;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnGetConvolutionBackwardFilterAlgorithmMaxCountResponse*>(responsePayload);
			if (count) { *count = response->count; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
	return err;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)
{
	TALLY_SPD_LOG("cudnnFindConvolutionBackwardFilterAlgorithm hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * y, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  dwDesc, void * dw, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnFindConvolutionBackwardFilterAlgorithmEx hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  srcDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, const int  requestedAlgoCount, int * returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)
{
	TALLY_SPD_LOG("cudnnGetConvolutionBackwardFilterAlgorithm_v7 hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t  handle, const cudnnTensorDescriptor_t  xDesc, const cudnnTensorDescriptor_t  dyDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnFilterDescriptor_t  gradDesc, cudnnConvolutionBwdFilterAlgo_t  algo, size_t * sizeInBytes)
{
	TALLY_SPD_LOG("cudnnGetConvolutionBackwardFilterWorkspaceSize hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnConvolutionDescriptor_t  convDesc, cudnnConvolutionBwdFilterAlgo_t  algo, void * workSpace, size_t  workSpaceSizeInBytes, const void * beta, const cudnnFilterDescriptor_t  dwDesc, void * dw)
{
	TALLY_SPD_LOG("cudnnConvolutionBackwardFilter hooked");
#if defined(RUN_LOCALLY)
	return lcudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t  handle, const void * alpha, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const void * beta, const cudnnTensorDescriptor_t  dbDesc, void * db)
{
	TALLY_SPD_LOG("cudnnConvolutionBackwardBias hooked");
#if defined(RUN_LOCALLY)
	return lcudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t * constPack, cudnnFusedOps_t  ops)
{
	TALLY_SPD_LOG("cudnnCreateFusedOpsConstParamPack hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateFusedOpsConstParamPack(constPack, ops);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t  constPack)
{
	TALLY_SPD_LOG("cudnnDestroyFusedOpsConstParamPack hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyFusedOpsConstParamPack(constPack);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, const void * param)
{
	TALLY_SPD_LOG("cudnnSetFusedOpsConstParamPackAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t  constPack, cudnnFusedOpsConstParamLabel_t  paramLabel, void * param, int * isNULL)
{
	TALLY_SPD_LOG("cudnnGetFusedOpsConstParamPackAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param, isNULL);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t * varPack, cudnnFusedOps_t  ops)
{
	TALLY_SPD_LOG("cudnnCreateFusedOpsVariantParamPack hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateFusedOpsVariantParamPack(varPack, ops);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t  varPack)
{
	TALLY_SPD_LOG("cudnnDestroyFusedOpsVariantParamPack hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyFusedOpsVariantParamPack(varPack);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)
{
	TALLY_SPD_LOG("cudnnSetFusedOpsVariantParamPackAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t  varPack, cudnnFusedOpsVariantParamLabel_t  paramLabel, void * ptr)
{
	TALLY_SPD_LOG("cudnnGetFusedOpsVariantParamPackAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t * plan, cudnnFusedOps_t  ops)
{
	TALLY_SPD_LOG("cudnnCreateFusedOpsPlan hooked");
#if defined(RUN_LOCALLY)
	return lcudnnCreateFusedOpsPlan(plan, ops);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t  plan)
{
	TALLY_SPD_LOG("cudnnDestroyFusedOpsPlan hooked");
#if defined(RUN_LOCALLY)
	return lcudnnDestroyFusedOpsPlan(plan);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnMakeFusedOpsPlan(cudnnHandle_t  handle, cudnnFusedOpsPlan_t  plan, const cudnnFusedOpsConstParamPack_t  constPack, size_t * workspaceSizeInBytes)
{
	TALLY_SPD_LOG("cudnnMakeFusedOpsPlan hooked");
#if defined(RUN_LOCALLY)
	return lcudnnMakeFusedOpsPlan(handle, plan, constPack, workspaceSizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t  handle, const cudnnFusedOpsPlan_t  plan, cudnnFusedOpsVariantParamPack_t  varPack)
{
	TALLY_SPD_LOG("cudnnFusedOpsExecute hooked");
#if defined(RUN_LOCALLY)
	return lcudnnFusedOpsExecute(handle, plan, varPack);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudnnStatus_t cudnnCnnTrainVersionCheck()
{
	TALLY_SPD_LOG("cudnnCnnTrainVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCnnTrainVersionCheck();
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCnnTrainVersionCheckArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNCNNTRAINVERSIONCHECK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnCnnTrainVersionCheckArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnCnnTrainVersionCheck);
	return err;
}

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t  descriptorType, cudnnBackendDescriptor_t * descriptor)
{
	TALLY_SPD_LOG("cudnnBackendCreateDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendCreateDescriptor(descriptorType, descriptor);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendCreateDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNBACKENDCREATEDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnBackendCreateDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->descriptorType = descriptorType;
			request->descriptor = descriptor;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudnnBackendCreateDescriptorResponse*>(responsePayload);
			if (descriptor) { *descriptor = response->descriptor; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnBackendCreateDescriptor);
	return err;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t  descriptor)
{
	TALLY_SPD_LOG("cudnnBackendDestroyDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendDestroyDescriptor(descriptor);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendDestroyDescriptorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNBACKENDDESTROYDESCRIPTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnBackendDestroyDescriptorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->descriptor = descriptor;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnBackendDestroyDescriptor);
	return err;
}

cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t  descriptor)
{
	TALLY_SPD_LOG("cudnnBackendInitialize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendInitialize(descriptor);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendInitializeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNBACKENDINITIALIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnBackendInitializeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->descriptor = descriptor;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnBackendInitialize);
	return err;
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t  descriptor)
{
	TALLY_SPD_LOG("cudnnBackendFinalize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendFinalize(descriptor);
#else

    cudnnStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendFinalizeArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDNNBACKENDFINALIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudnnBackendFinalizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->descriptor = descriptor;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudnnStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudnnBackendFinalize);
	return err;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t  handle, int*  version)
{
	TALLY_SPD_LOG("cublasGetVersion_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetVersion_v2(handle, version);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetVersion_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETVERSION_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetVersion_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->version = version;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasGetVersion_v2Response*>(responsePayload);
			if (version) { *version = response->version; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetVersion_v2);
	return err;
}

cublasStatus_t cublasGetProperty(libraryPropertyType  type, int*  value)
{
	TALLY_SPD_LOG("cublasGetProperty hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetProperty(type, value);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetPropertyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETPROPERTY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetPropertyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->type = type;
			request->value = value;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasGetPropertyResponse*>(responsePayload);
			if (value) { *value = response->value; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetProperty);
	return err;
}

size_t cublasGetCudartVersion()
{
	TALLY_SPD_LOG("cublasGetCudartVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetCudartVersion();
#else

    size_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetCudartVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETCUDARTVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetCudartVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const size_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetCudartVersion);
	return err;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t  handle, cudaStream_t*  streamId)
{
	TALLY_SPD_LOG("cublasGetStream_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetStream_v2(handle, streamId);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetStream_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETSTREAM_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetStream_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->streamId = streamId;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasGetStream_v2Response*>(responsePayload);
			if (streamId) { *streamId = response->streamId; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetStream_v2);
	return err;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t  handle, cublasPointerMode_t*  mode)
{
	TALLY_SPD_LOG("cublasGetPointerMode_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetPointerMode_v2(handle, mode);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetPointerMode_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETPOINTERMODE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetPointerMode_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasGetPointerMode_v2Response*>(responsePayload);
			if (mode) { *mode = response->mode; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetPointerMode_v2);
	return err;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t  handle, cublasPointerMode_t  mode)
{
	TALLY_SPD_LOG("cublasSetPointerMode_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetPointerMode_v2(handle, mode);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	};
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetPointerMode_v2Arg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETPOINTERMODE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetPointerMode_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->mode = mode;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cublasStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetPointerMode_v2);
	return err;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t  handle, cublasAtomicsMode_t*  mode)
{
	TALLY_SPD_LOG("cublasGetAtomicsMode hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetAtomicsMode(handle, mode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t  handle, cublasAtomicsMode_t  mode)
{
	TALLY_SPD_LOG("cublasSetAtomicsMode hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetAtomicsMode(handle, mode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetSmCountTarget(cublasHandle_t  handle, int*  smCountTarget)
{
	TALLY_SPD_LOG("cublasGetSmCountTarget hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetSmCountTarget(handle, smCountTarget);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetSmCountTargetArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETSMCOUNTTARGET;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetSmCountTargetArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->smCountTarget = smCountTarget;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasGetSmCountTargetResponse*>(responsePayload);
			if (smCountTarget) { *smCountTarget = response->smCountTarget; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetSmCountTarget);
	return err;
}

cublasStatus_t cublasSetSmCountTarget(cublasHandle_t  handle, int  smCountTarget)
{
	TALLY_SPD_LOG("cublasSetSmCountTarget hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetSmCountTarget(handle, smCountTarget);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	};
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetSmCountTargetArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETSMCOUNTTARGET;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetSmCountTargetArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->smCountTarget = smCountTarget;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cublasStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetSmCountTarget);
	return err;
}

const char* cublasGetStatusName(cublasStatus_t  status)
{
	TALLY_SPD_LOG("cublasGetStatusName hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetStatusName(status);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* cublasGetStatusString(cublasStatus_t  status)
{
	TALLY_SPD_LOG("cublasGetStatusString hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetStatusString(status);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLoggerConfigure(int  logIsOn, int  logToStdOut, int  logToStdErr, const char*  logFileName)
{
	TALLY_SPD_LOG("cublasLoggerConfigure hooked");
#if defined(RUN_LOCALLY)
	return lcublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback  userCallback)
{
	TALLY_SPD_LOG("cublasSetLoggerCallback hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetLoggerCallback(userCallback);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	};
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetLoggerCallbackArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETLOGGERCALLBACK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetLoggerCallbackArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->userCallback = userCallback;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cublasStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetLoggerCallback);
	return err;
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback*  userCallback)
{
	TALLY_SPD_LOG("cublasGetLoggerCallback hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetLoggerCallback(userCallback);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetLoggerCallbackArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASGETLOGGERCALLBACK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasGetLoggerCallbackArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->userCallback = userCallback;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasGetLoggerCallbackResponse*>(responsePayload);
			if (userCallback) { *userCallback = response->userCallback; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasGetLoggerCallback);
	return err;
}

cublasStatus_t cublasSetVector(int  n, int  elemSize, const void*  x, int  incx, void*  devicePtr, int  incy)
{
	TALLY_SPD_LOG("cublasSetVector hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetVector(n, elemSize, x, incx, devicePtr, incy);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	};
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetVectorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETVECTOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetVectorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->n = n;
			request->elemSize = elemSize;
			request->x = const_cast<void *>(x);
			request->incx = incx;
			request->devicePtr = devicePtr;
			request->incy = incy;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cublasStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasSetVector);
	return err;
}

cublasStatus_t cublasSetVector_64(int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  devicePtr, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSetVector_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetVector_64(n, elemSize, x, incx, devicePtr, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetVector(int  n, int  elemSize, const void*  x, int  incx, void*  y, int  incy)
{
	TALLY_SPD_LOG("cublasGetVector hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetVector(n, elemSize, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetVector_64(int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasGetVector_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetVector_64(n, elemSize, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	TALLY_SPD_LOG("cublasSetMatrix hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetMatrix_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb)
{
	TALLY_SPD_LOG("cublasSetMatrix_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetMatrix_64(rows, cols, elemSize, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	TALLY_SPD_LOG("cublasGetMatrix hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetMatrix_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb)
{
	TALLY_SPD_LOG("cublasGetMatrix_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetMatrix_64(rows, cols, elemSize, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetVectorAsync(int  n, int  elemSize, const void*  hostPtr, int  incx, void*  devicePtr, int  incy, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasSetVectorAsync hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetVectorAsync_64(int64_t  n, int64_t  elemSize, const void*  hostPtr, int64_t  incx, void*  devicePtr, int64_t  incy, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasSetVectorAsync_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetVectorAsync_64(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetVectorAsync(int  n, int  elemSize, const void*  devicePtr, int  incx, void*  hostPtr, int  incy, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasGetVectorAsync hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetVectorAsync_64(int64_t  n, int64_t  elemSize, const void*  devicePtr, int64_t  incx, void*  hostPtr, int64_t  incy, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasGetVectorAsync_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetVectorAsync_64(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasSetMatrixAsync hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSetMatrixAsync_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasSetMatrixAsync_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasGetMatrixAsync hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGetMatrixAsync_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasGetMatrixAsync_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

void cublasXerbla(const char*  srName, int  info)
{
	TALLY_SPD_LOG("cublasXerbla hooked");
#if defined(RUN_LOCALLY)
	return lcublasXerbla(srName, info);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasNrm2Ex(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasNrm2Ex hooked");
#if defined(RUN_LOCALLY)
	return lcublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasNrm2Ex_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasNrm2Ex_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasNrm2Ex_64(handle, n, x, xType, incx, result, resultType, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	TALLY_SPD_LOG("cublasSnrm2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSnrm2_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result)
{
	TALLY_SPD_LOG("cublasSnrm2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSnrm2_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDnrm2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDnrm2_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDnrm2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDnrm2_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScnrm2_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	TALLY_SPD_LOG("cublasScnrm2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScnrm2_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScnrm2_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result)
{
	TALLY_SPD_LOG("cublasScnrm2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScnrm2_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDznrm2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDznrm2_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDznrm2_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDznrm2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDznrm2_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDotEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasDotEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDotEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasDotEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDotEx_64(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDotcEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasDotcEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDotcEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasDotcEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDotcEx_64(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSdot_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, const float*  y, int  incy, float*  result)
{
	TALLY_SPD_LOG("cublasSdot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSdot_v2(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSdot_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  result)
{
	TALLY_SPD_LOG("cublasSdot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSdot_v2_64(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDdot_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, const double*  y, int  incy, double*  result)
{
	TALLY_SPD_LOG("cublasDdot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDdot_v2(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDdot_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  result)
{
	TALLY_SPD_LOG("cublasDdot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDdot_v2_64(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	TALLY_SPD_LOG("cublasCdotu_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCdotu_v2(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCdotu_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result)
{
	TALLY_SPD_LOG("cublasCdotu_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCdotu_v2_64(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	TALLY_SPD_LOG("cublasCdotc_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCdotc_v2(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCdotc_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result)
{
	TALLY_SPD_LOG("cublasCdotc_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCdotc_v2_64(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	TALLY_SPD_LOG("cublasZdotu_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdotu_v2(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdotu_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result)
{
	TALLY_SPD_LOG("cublasZdotu_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdotu_v2_64(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	TALLY_SPD_LOG("cublasZdotc_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdotc_v2(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdotc_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result)
{
	TALLY_SPD_LOG("cublasZdotc_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdotc_v2_64(handle, n, x, incx, y, incy, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScalEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int  incx, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasScalEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScalEx_64(cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int64_t  incx, cudaDataType  executionType)
{
	TALLY_SPD_LOG("cublasScalEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScalEx_64(handle, n, alpha, alphaType, x, xType, incx, executionType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasSscal_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSscal_v2(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSscal_v2_64(cublasHandle_t  handle, int64_t  n, const float*  alpha, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasSscal_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSscal_v2_64(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDscal_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDscal_v2(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDscal_v2_64(cublasHandle_t  handle, int64_t  n, const double*  alpha, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDscal_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDscal_v2_64(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCscal_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCscal_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCscal_v2(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCscal_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCscal_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCscal_v2_64(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCsscal_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsscal_v2(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsscal_v2_64(cublasHandle_t  handle, int64_t  n, const float*  alpha, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCsscal_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsscal_v2_64(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZscal_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZscal_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZscal_v2(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZscal_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZscal_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZscal_v2_64(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZdscal_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdscal_v2(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdscal_v2_64(cublasHandle_t  handle, int64_t  n, const double*  alpha, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZdscal_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdscal_v2_64(handle, n, alpha, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasAxpyEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasAxpyEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasAxpyEx_64(cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasAxpyEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasAxpyEx_64(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t  handle, int  n, const float*  alpha, const float*  x, int  incx, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasSaxpy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSaxpy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSaxpy_v2_64(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t  handle, int  n, const double*  alpha, const double*  x, int  incx, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDaxpy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDaxpy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDaxpy_v2_64(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCaxpy_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasCaxpy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCaxpy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCaxpy_v2_64(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZaxpy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZaxpy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZaxpy_v2_64(handle, n, alpha, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCopyEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	TALLY_SPD_LOG("cublasCopyEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCopyEx(handle, n, x, xType, incx, y, yType, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCopyEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCopyEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCopyEx_64(handle, n, x, xType, incx, y, yType, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScopy_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasScopy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScopy_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScopy_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasScopy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScopy_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDcopy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDcopy_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDcopy_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDcopy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDcopy_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCcopy_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasCcopy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCcopy_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCcopy_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCcopy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCcopy_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZcopy_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZcopy_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZcopy_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZcopy_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZcopy_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSswap_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasSswap_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSswap_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSswap_v2_64(cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSswap_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSswap_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDswap_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDswap_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDswap_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDswap_v2_64(cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDswap_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDswap_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCswap_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasCswap_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCswap_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCswap_v2_64(cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCswap_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCswap_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZswap_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZswap_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZswap_v2(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZswap_v2_64(cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZswap_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZswap_v2_64(handle, n, x, incx, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSwapEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	TALLY_SPD_LOG("cublasSwapEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasSwapEx(handle, n, x, xType, incx, y, yType, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSwapEx_64(cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSwapEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSwapEx_64(handle, n, x, xType, incx, y, yType, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIsamax_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIsamax_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIsamax_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIsamax_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIsamax_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIdamax_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIdamax_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIdamax_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIdamax_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIdamax_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIcamax_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIcamax_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIcamax_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIcamax_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIcamax_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIcamax_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIzamax_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIzamax_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIzamax_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIzamax_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIzamax_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIzamax_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIamaxEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIamaxEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasIamaxEx(handle, n, x, xType, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIamaxEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIamaxEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIamaxEx_64(handle, n, x, xType, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIsamin_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIsamin_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIsamin_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIsamin_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIsamin_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIdamin_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIdamin_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIdamin_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIdamin_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIdamin_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIcamin_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIcamin_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIcamin_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIcamin_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIcamin_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIcamin_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIzamin_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIzamin_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIzamin_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIzamin_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIzamin_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIzamin_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIaminEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	TALLY_SPD_LOG("cublasIaminEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasIaminEx(handle, n, x, xType, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasIaminEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result)
{
	TALLY_SPD_LOG("cublasIaminEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasIaminEx_64(handle, n, x, xType, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasAsumEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasAsumEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasAsumEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasAsumEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasAsumEx_64(handle, n, x, xType, incx, result, resultType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSasum_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	TALLY_SPD_LOG("cublasSasum_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSasum_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSasum_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result)
{
	TALLY_SPD_LOG("cublasSasum_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSasum_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDasum_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDasum_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDasum_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDasum_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDasum_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDasum_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScasum_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	TALLY_SPD_LOG("cublasScasum_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScasum_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasScasum_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result)
{
	TALLY_SPD_LOG("cublasScasum_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasScasum_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDzasum_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDzasum_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDzasum_v2(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDzasum_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result)
{
	TALLY_SPD_LOG("cublasDzasum_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDzasum_v2_64(handle, n, x, incx, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSrot_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  c, const float*  s)
{
	TALLY_SPD_LOG("cublasSrot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSrot_v2(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSrot_v2_64(cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  c, const float*  s)
{
	TALLY_SPD_LOG("cublasSrot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSrot_v2_64(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDrot_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  c, const double*  s)
{
	TALLY_SPD_LOG("cublasDrot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDrot_v2(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDrot_v2_64(cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  c, const double*  s)
{
	TALLY_SPD_LOG("cublasDrot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDrot_v2_64(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const cuComplex*  s)
{
	TALLY_SPD_LOG("cublasCrot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCrot_v2(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCrot_v2_64(cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const cuComplex*  s)
{
	TALLY_SPD_LOG("cublasCrot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCrot_v2_64(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const float*  s)
{
	TALLY_SPD_LOG("cublasCsrot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsrot_v2_64(cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const float*  s)
{
	TALLY_SPD_LOG("cublasCsrot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsrot_v2_64(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const cuDoubleComplex*  s)
{
	TALLY_SPD_LOG("cublasZrot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZrot_v2(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZrot_v2_64(cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const cuDoubleComplex*  s)
{
	TALLY_SPD_LOG("cublasZrot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZrot_v2_64(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const double*  s)
{
	TALLY_SPD_LOG("cublasZdrot_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdrot_v2_64(cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const double*  s)
{
	TALLY_SPD_LOG("cublasZdrot_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdrot_v2_64(handle, n, x, incx, y, incy, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasRotEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasRotEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasRotEx_64(cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasRotEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasRotEx_64(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSrotg_v2(cublasHandle_t  handle, float*  a, float*  b, float*  c, float*  s)
{
	TALLY_SPD_LOG("cublasSrotg_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSrotg_v2(handle, a, b, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDrotg_v2(cublasHandle_t  handle, double*  a, double*  b, double*  c, double*  s)
{
	TALLY_SPD_LOG("cublasDrotg_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDrotg_v2(handle, a, b, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCrotg_v2(cublasHandle_t  handle, cuComplex*  a, cuComplex*  b, float*  c, cuComplex*  s)
{
	TALLY_SPD_LOG("cublasCrotg_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCrotg_v2(handle, a, b, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZrotg_v2(cublasHandle_t  handle, cuDoubleComplex*  a, cuDoubleComplex*  b, double*  c, cuDoubleComplex*  s)
{
	TALLY_SPD_LOG("cublasZrotg_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZrotg_v2(handle, a, b, c, s);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasRotgEx(cublasHandle_t  handle, void*  a, void*  b, cudaDataType  abType, void*  c, void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasRotgEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasRotgEx(handle, a, b, abType, c, s, csType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSrotm_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  param)
{
	TALLY_SPD_LOG("cublasSrotm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSrotm_v2(handle, n, x, incx, y, incy, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSrotm_v2_64(cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  param)
{
	TALLY_SPD_LOG("cublasSrotm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSrotm_v2_64(handle, n, x, incx, y, incy, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDrotm_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  param)
{
	TALLY_SPD_LOG("cublasDrotm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDrotm_v2(handle, n, x, incx, y, incy, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDrotm_v2_64(cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  param)
{
	TALLY_SPD_LOG("cublasDrotm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDrotm_v2_64(handle, n, x, incx, y, incy, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasRotmEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasRotmEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasRotmEx_64(cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasRotmEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasRotmEx_64(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSrotmg_v2(cublasHandle_t  handle, float*  d1, float*  d2, float*  x1, const float*  y1, float*  param)
{
	TALLY_SPD_LOG("cublasSrotmg_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSrotmg_v2(handle, d1, d2, x1, y1, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDrotmg_v2(cublasHandle_t  handle, double*  d1, double*  d2, double*  x1, const double*  y1, double*  param)
{
	TALLY_SPD_LOG("cublasDrotmg_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDrotmg_v2(handle, d1, d2, x1, y1, param);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasRotmgEx(cublasHandle_t  handle, void*  d1, cudaDataType  d1Type, void*  d2, cudaDataType  d2Type, void*  x1, cudaDataType  x1Type, const void*  y1, cudaDataType  y1Type, void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_SPD_LOG("cublasRotmgEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSgemv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDgemv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDgemv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasCgemv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCgemv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZgemv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZgemv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasSgbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSgbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDgbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDgbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasCgbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCgbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZgbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZgbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasStrmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasStrmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDtrmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDtrmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCtrmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCtrmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZtrmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZtrmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasStbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasStbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDtbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDtbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCtbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCtbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZtbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZtbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasStpmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasStpmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDtpmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDtpmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCtpmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCtpmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZtpmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZtpmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasStrsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasStrsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDtrsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDtrsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCtrsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCtrsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZtrsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZtrsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasStpsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasStpsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDtpsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDtpsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCtpsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCtpsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZtpsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZtpsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_SPD_LOG("cublasStbsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasStbsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_SPD_LOG("cublasDtbsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasDtbsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasCtbsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasCtbsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_SPD_LOG("cublasZtbsv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_SPD_LOG("cublasZtbsv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasSsymv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSsymv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDsymv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDsymv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasCsymv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasCsymv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZsymv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZsymv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasChemv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChemv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasChemv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChemv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZhemv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhemv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZhemv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhemv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasSsbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSsbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDsbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDsbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasChbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasChbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZhbmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZhbmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  AP, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_SPD_LOG("cublasSspmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSspmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  AP, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasSspmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSspmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  AP, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_SPD_LOG("cublasDspmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDspmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  AP, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasDspmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDspmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasChpmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasChpmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChpmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_SPD_LOG("cublasZhpmv_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_SPD_LOG("cublasZhpmv_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhpmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSger_v2(cublasHandle_t  handle, int  m, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	TALLY_SPD_LOG("cublasSger_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSger_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasSger_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSger_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDger_v2(cublasHandle_t  handle, int  m, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	TALLY_SPD_LOG("cublasDger_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDger_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasDger_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDger_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCgeru_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgeru_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasCgeru_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgeru_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCgerc_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgerc_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasCgerc_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgerc_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZgeru_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgeru_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasZgeru_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgeru_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZgerc_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgerc_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasZgerc_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgerc_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  A, int  lda)
{
	TALLY_SPD_LOG("cublasSsyr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasSsyr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  A, int  lda)
{
	TALLY_SPD_LOG("cublasDsyr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasDsyr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCsyr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasCsyr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZsyr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasZsyr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCher_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCher_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasCher_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCher_v2_64(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZher_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZher_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasZher_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZher_v2_64(handle, uplo, n, alpha, x, incx, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  AP)
{
	TALLY_SPD_LOG("cublasSspr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSspr_v2(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSspr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  AP)
{
	TALLY_SPD_LOG("cublasSspr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSspr_v2_64(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  AP)
{
	TALLY_SPD_LOG("cublasDspr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDspr_v2(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDspr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  AP)
{
	TALLY_SPD_LOG("cublasDspr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDspr_v2_64(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  AP)
{
	TALLY_SPD_LOG("cublasChpr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChpr_v2(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChpr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  AP)
{
	TALLY_SPD_LOG("cublasChpr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChpr_v2_64(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  AP)
{
	TALLY_SPD_LOG("cublasZhpr_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhpr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  AP)
{
	TALLY_SPD_LOG("cublasZhpr_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhpr_v2_64(handle, uplo, n, alpha, x, incx, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	TALLY_SPD_LOG("cublasSsyr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasSsyr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	TALLY_SPD_LOG("cublasDsyr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasDsyr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCsyr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasCsyr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZsyr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasZsyr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCher2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCher2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasCher2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCher2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZher2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZher2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_SPD_LOG("cublasZher2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZher2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  AP)
{
	TALLY_SPD_LOG("cublasSspr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSspr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  AP)
{
	TALLY_SPD_LOG("cublasSspr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSspr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  AP)
{
	TALLY_SPD_LOG("cublasDspr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDspr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  AP)
{
	TALLY_SPD_LOG("cublasDspr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDspr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  AP)
{
	TALLY_SPD_LOG("cublasChpr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChpr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  AP)
{
	TALLY_SPD_LOG("cublasChpr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChpr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  AP)
{
	TALLY_SPD_LOG("cublasZhpr2_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhpr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  AP)
{
	TALLY_SPD_LOG("cublasZhpr2_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhpr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasSgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasSgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  xarray[], int  incx, const double*  beta, double* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasDgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  xarray[], int64_t  incx, const double*  beta, double* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasDgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  xarray[], int  incx, const cuComplex*  beta, cuComplex* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasCgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  xarray[], int64_t  incx, const cuComplex*  beta, cuComplex* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  xarray[], int  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasZgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  xarray[], int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasZgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSHgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, __half* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasHSHgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSHgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSHgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, __half* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasHSHgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSHgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSSgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasHSSgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSSgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasHSSgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSTgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, __nv_bfloat16* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasTSTgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSTgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSTgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, __nv_bfloat16* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasTSTgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSTgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSSgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)
{
	TALLY_SPD_LOG("cublasTSSgemvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSSgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasTSSgemvBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasSgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasSgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  x, int  incx, long long int  stridex, const double*  beta, double*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasDgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  x, int64_t  incx, long long int  stridex, const double*  beta, double*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasDgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  x, int  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasCgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  x, int64_t  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  x, int  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasZgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  x, int64_t  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasZgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSHgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, __half*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasHSHgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSHgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSHgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, __half*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasHSHgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSHgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSSgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasHSSgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHSSgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasHSSgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHSSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSTgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasTSTgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSTgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSTgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasTSTgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSTgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSSgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_SPD_LOG("cublasTSSgemvStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasTSSgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasTSSgemvStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasTSSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSgemm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDgemm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDgemm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCgemm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCgemm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCgemm3m hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3m_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCgemm3m_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3m_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3mEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_SPD_LOG("cublasCgemm3mEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3mEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCgemm3mEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3mEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZgemm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZgemm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZgemm3m hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemm3m_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZgemm3m_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemm3m_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHgemm(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, const __half*  B, int  ldb, const __half*  beta, __half*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasHgemm hooked");
#if defined(RUN_LOCALLY)
	return lcublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHgemm_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, const __half*  B, int64_t  ldb, const __half*  beta, __half*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasHgemm_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHgemm_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemmEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSgemmEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGemmEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_SPD_LOG("cublasGemmEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_SPD_LOG("cublasCgemmEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemmEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCgemmEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  beta, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSsyrk_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSsyrk_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  beta, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDsyrk_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDsyrk_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCsyrk_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCsyrk_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZsyrk_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZsyrk_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_SPD_LOG("cublasCsyrkEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrkEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCsyrkEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrkEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_SPD_LOG("cublasCsyrk3mEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrk3mEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCsyrk3mEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrk3mEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const cuComplex*  A, int  lda, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCherk_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const cuComplex*  A, int64_t  lda, const float*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCherk_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const cuDoubleComplex*  A, int  lda, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZherk_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZherk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const cuDoubleComplex*  A, int64_t  lda, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZherk_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZherk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_SPD_LOG("cublasCherkEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherkEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCherkEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherkEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_SPD_LOG("cublasCherk3mEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherk3mEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCherk3mEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherk3mEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSsyr2k_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSsyr2k_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDsyr2k_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDsyr2k_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCsyr2k_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCsyr2k_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZsyr2k_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZsyr2k_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCher2k_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCher2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCher2k_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCher2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZher2k_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZher2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZher2k_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZher2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSsyrkx hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSsyrkx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDsyrkx hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDsyrkx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCsyrkx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCsyrkx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZsyrkx hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZsyrkx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCherkx hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCherkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCherkx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCherkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZherkx hooked");
#if defined(RUN_LOCALLY)
	return lcublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZherkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZherkx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZherkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSsymm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSsymm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDsymm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDsymm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCsymm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCsymm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZsymm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZsymm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasChemm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasChemm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasChemm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasChemm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZhemm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZhemm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZhemm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZhemm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, float*  B, int  ldb)
{
	TALLY_SPD_LOG("cublasStrsm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, float*  B, int64_t  ldb)
{
	TALLY_SPD_LOG("cublasStrsm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, double*  B, int  ldb)
{
	TALLY_SPD_LOG("cublasDtrsm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, double*  B, int64_t  ldb)
{
	TALLY_SPD_LOG("cublasDtrsm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, cuComplex*  B, int  ldb)
{
	TALLY_SPD_LOG("cublasCtrsm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, cuComplex*  B, int64_t  ldb)
{
	TALLY_SPD_LOG("cublasCtrsm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  B, int  ldb)
{
	TALLY_SPD_LOG("cublasZtrsm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  B, int64_t  ldb)
{
	TALLY_SPD_LOG("cublasZtrsm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasStrmm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasStrmm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDtrmm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDtrmm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCtrmm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCtrmm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZtrmm_v2 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZtrmm_v2_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half* const  Aarray[], int  lda, const __half* const  Barray[], int  ldb, const __half*  beta, __half* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_SPD_LOG("cublasHgemmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  Barray[], int64_t  ldb, const __half*  beta, __half* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasHgemmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  Barray[], int  ldb, const float*  beta, float* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_SPD_LOG("cublasSgemmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  Barray[], int64_t  ldb, const float*  beta, float* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasSgemmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  Barray[], int  ldb, const double*  beta, double* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_SPD_LOG("cublasDgemmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  Barray[], int64_t  ldb, const double*  beta, double* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasDgemmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_SPD_LOG("cublasCgemmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCgemmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3mBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_SPD_LOG("cublasCgemm3mBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3mBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCgemm3mBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3mBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  Barray[], int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_SPD_LOG("cublasZgemmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  Barray[], int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasZgemmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  B, int  ldb, long long int  strideB, const __half*  beta, __half*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_SPD_LOG("cublasHgemmStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasHgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  B, int64_t  ldb, long long int  strideB, const __half*  beta, __half*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasHgemmStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasHgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  B, int64_t  ldb, long long int  strideB, const float*  beta, float*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasSgemmStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  B, int  ldb, long long int  strideB, const double*  beta, double*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_SPD_LOG("cublasDgemmStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  B, int64_t  ldb, long long int  strideB, const double*  beta, double*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasDgemmStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_SPD_LOG("cublasCgemmStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCgemmStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_SPD_LOG("cublasCgemm3mStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgemm3mStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCgemm3mStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgemm3mStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  B, int  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_SPD_LOG("cublasZgemmStridedBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  B, int64_t  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasZgemmStridedBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int  lda, const void* const  Barray[], cudaDataType  Btype, int  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int  ldc, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_SPD_LOG("cublasGemmBatchedEx hooked");
#if defined(RUN_LOCALLY)
	return lcublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int64_t  lda, const void* const  Barray[], cudaDataType  Btype, int64_t  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int64_t  ldc, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_SPD_LOG("cublasGemmBatchedEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGemmBatchedEx_64(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int64_t  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, long long int  strideC, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_SPD_LOG("cublasGemmStridedBatchedEx_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasGemmStridedBatchedEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  beta, const float*  B, int  ldb, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSgeam hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, const float*  B, int64_t  ldb, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSgeam_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  beta, const double*  B, int  ldb, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDgeam hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, const double*  B, int64_t  ldb, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDgeam_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCgeam hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCgeam_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZgeam hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZgeam_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float* const  A[], int  lda, float* const  B[], int  ldb, int  batchCount)
{
	TALLY_SPD_LOG("cublasStrsmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float* const  A[], int64_t  lda, float* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasStrsmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double* const  A[], int  lda, double* const  B[], int  ldb, int  batchCount)
{
	TALLY_SPD_LOG("cublasDtrsmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double* const  A[], int64_t  lda, double* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasDtrsmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  A[], int  lda, cuComplex* const  B[], int  ldb, int  batchCount)
{
	TALLY_SPD_LOG("cublasCtrsmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  A[], int64_t  lda, cuComplex* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasCtrsmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  B[], int  ldb, int  batchCount)
{
	TALLY_SPD_LOG("cublasZtrsmBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int64_t  lda, cuDoubleComplex* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_SPD_LOG("cublasZtrsmBatched_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const float*  A, int  lda, const float*  x, int  incx, float*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasSdgmm hooked");
#if defined(RUN_LOCALLY)
	return lcublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const float*  A, int64_t  lda, const float*  x, int64_t  incx, float*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasSdgmm_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasSdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const double*  A, int  lda, const double*  x, int  incx, double*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasDdgmm hooked");
#if defined(RUN_LOCALLY)
	return lcublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const double*  A, int64_t  lda, const double*  x, int64_t  incx, double*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasDdgmm_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasDdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, cuComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasCdgmm hooked");
#if defined(RUN_LOCALLY)
	return lcublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, cuComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasCdgmm_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasCdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  C, int  ldc)
{
	TALLY_SPD_LOG("cublasZdgmm hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_SPD_LOG("cublasZdgmm_64 hooked");
#if defined(RUN_LOCALLY)
	return lcublasZdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSmatinvBatched(cublasHandle_t  handle, int  n, const float* const  A[], int  lda, float* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasSmatinvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDmatinvBatched(cublasHandle_t  handle, int  n, const double* const  A[], int  lda, double* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasDmatinvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCmatinvBatched(cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, cuComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasCmatinvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZmatinvBatched(cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  Ainv[], int  lda_inv, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasZmatinvBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t  handle, int  m, int  n, float* const  Aarray[], int  lda, float* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasSgeqrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t  handle, int  m, int  n, double* const  Aarray[], int  lda, double* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasDgeqrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t  handle, int  m, int  n, cuComplex* const  Aarray[], int  lda, cuComplex* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasCgeqrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t  handle, int  m, int  n, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  TauArray[], int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasZgeqrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, float* const  Aarray[], int  lda, float* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_SPD_LOG("cublasSgelsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, double* const  Aarray[], int  lda, double* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_SPD_LOG("cublasDgelsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuComplex* const  Aarray[], int  lda, cuComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_SPD_LOG("cublasCgelsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  nrhs, cuDoubleComplex* const  Aarray[], int  lda, cuDoubleComplex* const  Carray[], int  ldc, int*  info, int*  devInfoArray, int  batchSize)
{
	TALLY_SPD_LOG("cublasZgelsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  AP, float*  A, int  lda)
{
	TALLY_SPD_LOG("cublasStpttr hooked");
#if defined(RUN_LOCALLY)
	return lcublasStpttr(handle, uplo, n, AP, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  AP, double*  A, int  lda)
{
	TALLY_SPD_LOG("cublasDtpttr hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtpttr(handle, uplo, n, AP, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  AP, cuComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasCtpttr hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtpttr(handle, uplo, n, AP, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtpttr(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  A, int  lda)
{
	TALLY_SPD_LOG("cublasZtpttr hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtpttr(handle, uplo, n, AP, A, lda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasStrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  A, int  lda, float*  AP)
{
	TALLY_SPD_LOG("cublasStrttp hooked");
#if defined(RUN_LOCALLY)
	return lcublasStrttp(handle, uplo, n, A, lda, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  A, int  lda, double*  AP)
{
	TALLY_SPD_LOG("cublasDtrttp hooked");
#if defined(RUN_LOCALLY)
	return lcublasDtrttp(handle, uplo, n, A, lda, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  A, int  lda, cuComplex*  AP)
{
	TALLY_SPD_LOG("cublasCtrttp hooked");
#if defined(RUN_LOCALLY)
	return lcublasCtrttp(handle, uplo, n, A, lda, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZtrttp(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  AP)
{
	TALLY_SPD_LOG("cublasZtrttp hooked");
#if defined(RUN_LOCALLY)
	return lcublasZtrttp(handle, uplo, n, A, lda, AP);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t  handle, int  n, float* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasSgetrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t  handle, int  n, double* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasDgetrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t  handle, int  n, cuComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasCgetrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t  handle, int  n, cuDoubleComplex* const  A[], int  lda, int*  P, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasZgetrfBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t  handle, int  n, const float* const  A[], int  lda, const int*  P, float* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasSgetriBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t  handle, int  n, const double* const  A[], int  lda, const int*  P, double* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasDgetriBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t  handle, int  n, const cuComplex* const  A[], int  lda, const int*  P, cuComplex* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasCgetriBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t  handle, int  n, const cuDoubleComplex* const  A[], int  lda, const int*  P, cuDoubleComplex* const  C[], int  ldc, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasZgetriBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const float* const  Aarray[], int  lda, const int*  devIpiv, float* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasSgetrsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const double* const  Aarray[], int  lda, const int*  devIpiv, double* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasDgetrsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuComplex* const  Aarray[], int  lda, const int*  devIpiv, cuComplex* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasCgetrsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  n, int  nrhs, const cuDoubleComplex* const  Aarray[], int  lda, const int*  devIpiv, cuDoubleComplex* const  Barray[], int  ldb, int*  info, int  batchSize)
{
	TALLY_SPD_LOG("cublasZgetrsBatched hooked");
#if defined(RUN_LOCALLY)
	return lcublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasUint8gemmBias(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, cublasOperation_t  transc, int  m, int  n, int  k, const unsigned char*  A, int  A_bias, int  lda, const unsigned char*  B, int  B_bias, int  ldb, unsigned char*  C, int  C_bias, int  ldc, int  C_mult, int  C_shift)
{
	TALLY_SPD_LOG("cublasUint8gemmBias hooked");
#if defined(RUN_LOCALLY)
	return lcublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cudaError_t cudaProfilerStart()
{
	TALLY_SPD_LOG("cudaProfilerStart hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaProfilerStart();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaProfilerStartArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAPROFILERSTART;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaProfilerStartArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaProfilerStart);
	return err;
}

cudaError_t cudaProfilerStop()
{
	TALLY_SPD_LOG("cudaProfilerStop hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaProfilerStop();
#else

    cudaError_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaProfilerStopArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAPROFILERSTOP;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaProfilerStopArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cudaError_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	last_err = err;
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaProfilerStop);
	return err;
}

CUresult cuProfilerInitialize(const char * configFile, const char * outputFile, CUoutput_mode  outputMode)
{
	TALLY_SPD_LOG("cuProfilerInitialize hooked");
#if defined(RUN_LOCALLY)
	return lcuProfilerInitialize(configFile, outputFile, outputMode);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuProfilerStart()
{
	TALLY_SPD_LOG("cuProfilerStart hooked");
#if defined(RUN_LOCALLY)
	return lcuProfilerStart();
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

CUresult cuProfilerStop()
{
	TALLY_SPD_LOG("cuProfilerStop hooked");
#if defined(RUN_LOCALLY)
	return lcuProfilerStop();
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char * nvrtcGetErrorString(nvrtcResult  result)
{
	TALLY_SPD_LOG("nvrtcGetErrorString hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetErrorString(result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcVersion(int * major, int * minor)
{
	TALLY_SPD_LOG("nvrtcVersion hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcVersion(major, minor);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetNumSupportedArchs(int*  numArchs)
{
	TALLY_SPD_LOG("nvrtcGetNumSupportedArchs hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetNumSupportedArchs(numArchs);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetSupportedArchs(int*  supportedArchs)
{
	TALLY_SPD_LOG("nvrtcGetSupportedArchs hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetSupportedArchs(supportedArchs);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram * prog, const char * src, const char * name, int  numHeaders, const char * const * headers, const char * const * includeNames)
{
	TALLY_SPD_LOG("nvrtcCreateProgram hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram * prog)
{
	TALLY_SPD_LOG("nvrtcDestroyProgram hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcDestroyProgram(prog);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram  prog, int  numOptions, const char * const * options)
{
	TALLY_SPD_LOG("nvrtcCompileProgram hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcCompileProgram(prog, numOptions, options);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram  prog, size_t * ptxSizeRet)
{
	TALLY_SPD_LOG("nvrtcGetPTXSize hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetPTXSize(prog, ptxSizeRet);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetPTX(nvrtcProgram  prog, char * ptx)
{
	TALLY_SPD_LOG("nvrtcGetPTX hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetPTX(prog, ptx);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetCUBINSize(nvrtcProgram  prog, size_t * cubinSizeRet)
{
	TALLY_SPD_LOG("nvrtcGetCUBINSize hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetCUBINSize(prog, cubinSizeRet);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetCUBIN(nvrtcProgram  prog, char * cubin)
{
	TALLY_SPD_LOG("nvrtcGetCUBIN hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetCUBIN(prog, cubin);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram  prog, size_t * LTOIRSizeRet)
{
	TALLY_SPD_LOG("nvrtcGetLTOIRSize hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetLTOIRSize(prog, LTOIRSizeRet);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetLTOIR(nvrtcProgram  prog, char * LTOIR)
{
	TALLY_SPD_LOG("nvrtcGetLTOIR hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetLTOIR(prog, LTOIR);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram  prog, size_t * optixirSizeRet)
{
	TALLY_SPD_LOG("nvrtcGetOptiXIRSize hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetOptiXIRSize(prog, optixirSizeRet);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetOptiXIR(nvrtcProgram  prog, char * optixir)
{
	TALLY_SPD_LOG("nvrtcGetOptiXIR hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetOptiXIR(prog, optixir);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram  prog, size_t * logSizeRet)
{
	TALLY_SPD_LOG("nvrtcGetProgramLogSize hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetProgramLogSize(prog, logSizeRet);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram  prog, char * log)
{
	TALLY_SPD_LOG("nvrtcGetProgramLog hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetProgramLog(prog, log);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcAddNameExpression(nvrtcProgram  prog, const char * const  name_expression)
{
	TALLY_SPD_LOG("nvrtcAddNameExpression hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcAddNameExpression(prog, name_expression);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

nvrtcResult nvrtcGetLoweredName(nvrtcProgram  prog, const char *const  name_expression, const char**  lowered_name)
{
	TALLY_SPD_LOG("nvrtcGetLoweredName hooked");
#if defined(RUN_LOCALLY)
	return lnvrtcGetLoweredName(prog, name_expression, lowered_name);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* cublasLtGetStatusName(cublasStatus_t  status)
{
	TALLY_SPD_LOG("cublasLtGetStatusName hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtGetStatusName(status);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* cublasLtGetStatusString(cublasStatus_t  status)
{
	TALLY_SPD_LOG("cublasLtGetStatusString hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtGetStatusString(status);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

size_t cublasLtGetVersion()
{
	TALLY_SPD_LOG("cublasLtGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtGetVersion();
#else

    size_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtGetVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTGETVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtGetVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const size_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtGetVersion);
	return err;
}

size_t cublasLtGetCudartVersion()
{
	TALLY_SPD_LOG("cublasLtGetCudartVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtGetCudartVersion();
#else

    size_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtGetCudartVersionArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTGETCUDARTVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtGetCudartVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const size_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtGetCudartVersion);
	return err;
}

cublasStatus_t cublasLtGetProperty(libraryPropertyType  type, int*  value)
{
	TALLY_SPD_LOG("cublasLtGetProperty hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtGetProperty(type, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtHeuristicsCacheGetCapacity(size_t*  capacity)
{
	TALLY_SPD_LOG("cublasLtHeuristicsCacheGetCapacity hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtHeuristicsCacheGetCapacity(capacity);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtHeuristicsCacheSetCapacity(size_t  capacity)
{
	TALLY_SPD_LOG("cublasLtHeuristicsCacheSetCapacity hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtHeuristicsCacheSetCapacity(capacity);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

unsigned cublasLtDisableCpuInstructionsSetMask(unsigned  mask)
{
	TALLY_SPD_LOG("cublasLtDisableCpuInstructionsSetMask hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtDisableCpuInstructionsSetMask(mask);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t  lightHandle, cublasLtMatrixTransformDesc_t  transformDesc, const void*  alpha, const void*  A, cublasLtMatrixLayout_t  Adesc, const void*  beta, const void*  B, cublasLtMatrixLayout_t  Bdesc, void*  C, cublasLtMatrixLayout_t  Cdesc, cudaStream_t  stream)
{
	TALLY_SPD_LOG("cublasLtMatrixTransform hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatrixTransform(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t  matLayout, size_t  size, cudaDataType  type, uint64_t  rows, uint64_t  cols, int64_t  ld)
{
	TALLY_SPD_LOG("cublasLtMatrixLayoutInit_internal hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatrixLayoutInit_internal(matLayout, size, type, rows, cols, ld);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t  matLayout, cublasLtMatrixLayoutAttribute_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_SPD_LOG("cublasLtMatrixLayoutGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatrixLayoutGetAttribute(matLayout, attr, buf, sizeInBytes, sizeWritten);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t  matmulDesc, size_t  size, cublasComputeType_t  computeType, cudaDataType_t  scaleType)
{
	TALLY_SPD_LOG("cublasLtMatmulDescInit_internal hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulDescInit_internal(matmulDesc, size, computeType, scaleType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t  matmulDesc, cublasLtMatmulDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_SPD_LOG("cublasLtMatmulDescGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulDescGetAttribute(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatrixTransformDescInit_internal(cublasLtMatrixTransformDesc_t  transformDesc, size_t  size, cudaDataType  scaleType)
{
	TALLY_SPD_LOG("cublasLtMatrixTransformDescInit_internal hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatrixTransformDescInit_internal(transformDesc, size, scaleType);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t*  transformDesc, cudaDataType  scaleType)
{
	TALLY_SPD_LOG("cublasLtMatrixTransformDescCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatrixTransformDescCreate(transformDesc, scaleType);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	}

    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatrixTransformDescCreateArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatrixTransformDescCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->transformDesc = transformDesc;
			request->scaleType = scaleType;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatrixTransformDescCreateResponse*>(responsePayload);
			if (transformDesc) { *transformDesc = response->transformDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixTransformDescCreate);
	return err;
}

cublasStatus_t cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t  transformDesc)
{
	TALLY_SPD_LOG("cublasLtMatrixTransformDescDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatrixTransformDescDestroy(transformDesc);
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	};
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatrixTransformDescDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXTRANSFORMDESCDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatrixTransformDescDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->transformDesc = transformDesc;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cublasStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixTransformDescDestroy);
	return err;
}

cublasStatus_t cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_SPD_LOG("cublasLtMatrixTransformDescSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatrixTransformDescSetAttribute(transformDesc, attr, buf, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t  transformDesc, cublasLtMatrixTransformDescAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_SPD_LOG("cublasLtMatrixTransformDescGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatrixTransformDescGetAttribute(transformDesc, attr, buf, sizeInBytes, sizeWritten);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t  pref, size_t  size)
{
	TALLY_SPD_LOG("cublasLtMatmulPreferenceInit_internal hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulPreferenceInit_internal(pref, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t  pref, cublasLtMatmulPreferenceAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_SPD_LOG("cublasLtMatmulPreferenceGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulPreferenceGetAttribute(pref, attr, buf, sizeInBytes, sizeWritten);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  requestedAlgoCount, int  algoIdsArray[], int*  returnAlgoCount)
{
	TALLY_SPD_LOG("cublasLtMatmulAlgoGetIds hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulAlgoGetIds(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, requestedAlgoCount, algoIdsArray, returnAlgoCount);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t  lightHandle, cublasComputeType_t  computeType, cudaDataType_t  scaleType, cudaDataType_t  Atype, cudaDataType_t  Btype, cudaDataType_t  Ctype, cudaDataType_t  Dtype, int  algoId, cublasLtMatmulAlgo_t*  algo)
{
	TALLY_SPD_LOG("cublasLtMatmulAlgoInit hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulAlgoInit(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t  lightHandle, cublasLtMatmulDesc_t  operationDesc, cublasLtMatrixLayout_t  Adesc, cublasLtMatrixLayout_t  Bdesc, cublasLtMatrixLayout_t  Cdesc, cublasLtMatrixLayout_t  Ddesc, const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulHeuristicResult_t*  result)
{
	TALLY_SPD_LOG("cublasLtMatmulAlgoCheck hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulAlgoCheck(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoCapAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_SPD_LOG("cublasLtMatmulAlgoCapGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulAlgoCapGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, const void*  buf, size_t  sizeInBytes)
{
	TALLY_SPD_LOG("cublasLtMatmulAlgoConfigSetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulAlgoConfigSetAttribute(algo, attr, buf, sizeInBytes);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t*  algo, cublasLtMatmulAlgoConfigAttributes_t  attr, void*  buf, size_t  sizeInBytes, size_t*  sizeWritten)
{
	TALLY_SPD_LOG("cublasLtMatmulAlgoConfigGetAttribute hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtMatmulAlgoConfigGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t  callback)
{
	TALLY_SPD_LOG("cublasLtLoggerSetCallback hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtLoggerSetCallback(callback);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtLoggerSetFile(FILE*  file)
{
	TALLY_SPD_LOG("cublasLtLoggerSetFile hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtLoggerSetFile(file);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtLoggerOpenFile(const char*  logFile)
{
	TALLY_SPD_LOG("cublasLtLoggerOpenFile hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtLoggerOpenFile(logFile);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtLoggerSetLevel(int  level)
{
	TALLY_SPD_LOG("cublasLtLoggerSetLevel hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtLoggerSetLevel(level);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtLoggerSetMask(int  mask)
{
	TALLY_SPD_LOG("cublasLtLoggerSetMask hooked");
#if defined(RUN_LOCALLY)
	return lcublasLtLoggerSetMask(mask);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

cublasStatus_t cublasLtLoggerForceDisable()
{
	TALLY_SPD_LOG("cublasLtLoggerForceDisable hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtLoggerForceDisable();
#else
	if (replace_cublas) {
		auto err = CUBLAS_STATUS_SUCCESS;
		throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cublas function is not handled when REPLACE_CUBLAS is set.");
	};
    cublasStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtLoggerForceDisableArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTLOGGERFORCEDISABLE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtLoggerForceDisableArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const cublasStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtLoggerForceDisable);
	return err;
}

curandStatus_t curandCreateGenerator(curandGenerator_t * generator, curandRngType_t  rng_type)
{
	TALLY_SPD_LOG("curandCreateGenerator hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcurandCreateGenerator(generator, rng_type);
#else

    curandStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(curandCreateGeneratorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CURANDCREATEGENERATOR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (curandCreateGeneratorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->generator = generator;
			request->rng_type = rng_type;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const curandCreateGeneratorResponse*>(responsePayload);
			if (generator) { *generator = response->generator; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(curandCreateGenerator);
	return err;
}

curandStatus_t curandCreateGeneratorHost(curandGenerator_t * generator, curandRngType_t  rng_type)
{
	TALLY_SPD_LOG("curandCreateGeneratorHost hooked");
#if defined(RUN_LOCALLY)
	return lcurandCreateGeneratorHost(generator, rng_type);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandDestroyGenerator(curandGenerator_t  generator)
{
	TALLY_SPD_LOG("curandDestroyGenerator hooked");
#if defined(RUN_LOCALLY)
	return lcurandDestroyGenerator(generator);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGetVersion(int * version)
{
	TALLY_SPD_LOG("curandGetVersion hooked");
#if defined(RUN_LOCALLY)
	return lcurandGetVersion(version);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGetProperty(libraryPropertyType  type, int * value)
{
	TALLY_SPD_LOG("curandGetProperty hooked");
#if defined(RUN_LOCALLY)
	return lcurandGetProperty(type, value);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandSetStream(curandGenerator_t  generator, cudaStream_t  stream)
{
	TALLY_SPD_LOG("curandSetStream hooked");
#if defined(RUN_LOCALLY)
	return lcurandSetStream(generator, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t  generator, unsigned long long  seed)
{
	TALLY_SPD_LOG("curandSetPseudoRandomGeneratorSeed hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcurandSetPseudoRandomGeneratorSeed(generator, seed);
#else

    curandStatus_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(curandSetPseudoRandomGeneratorSeedArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CURANDSETPSEUDORANDOMGENERATORSEED;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (curandSetPseudoRandomGeneratorSeedArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->generator = generator;
			request->seed = seed;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const curandStatus_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(curandSetPseudoRandomGeneratorSeed);
	return err;
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t  generator, unsigned long long  offset)
{
	TALLY_SPD_LOG("curandSetGeneratorOffset hooked");
#if defined(RUN_LOCALLY)
	return lcurandSetGeneratorOffset(generator, offset);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandSetGeneratorOrdering(curandGenerator_t  generator, curandOrdering_t  order)
{
	TALLY_SPD_LOG("curandSetGeneratorOrdering hooked");
#if defined(RUN_LOCALLY)
	return lcurandSetGeneratorOrdering(generator, order);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t  generator, unsigned int  num_dimensions)
{
	TALLY_SPD_LOG("curandSetQuasiRandomGeneratorDimensions hooked");
#if defined(RUN_LOCALLY)
	return lcurandSetQuasiRandomGeneratorDimensions(generator, num_dimensions);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerate(curandGenerator_t  generator, unsigned int * outputPtr, size_t  num)
{
	TALLY_SPD_LOG("curandGenerate hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerate(generator, outputPtr, num);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateLongLong(curandGenerator_t  generator, unsigned long long * outputPtr, size_t  num)
{
	TALLY_SPD_LOG("curandGenerateLongLong hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateLongLong(generator, outputPtr, num);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateUniform(curandGenerator_t  generator, float * outputPtr, size_t  num)
{
	TALLY_SPD_LOG("curandGenerateUniform hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateUniform(generator, outputPtr, num);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t  generator, double * outputPtr, size_t  num)
{
	TALLY_SPD_LOG("curandGenerateUniformDouble hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateUniformDouble(generator, outputPtr, num);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateNormal(curandGenerator_t  generator, float * outputPtr, size_t  n, float  mean, float  stddev)
{
	TALLY_SPD_LOG("curandGenerateNormal hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateNormal(generator, outputPtr, n, mean, stddev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t  generator, double * outputPtr, size_t  n, double  mean, double  stddev)
{
	TALLY_SPD_LOG("curandGenerateNormalDouble hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t  generator, float * outputPtr, size_t  n, float  mean, float  stddev)
{
	TALLY_SPD_LOG("curandGenerateLogNormal hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateLogNormal(generator, outputPtr, n, mean, stddev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t  generator, double * outputPtr, size_t  n, double  mean, double  stddev)
{
	TALLY_SPD_LOG("curandGenerateLogNormalDouble hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandCreatePoissonDistribution(double  lambda, curandDiscreteDistribution_t * discrete_distribution)
{
	TALLY_SPD_LOG("curandCreatePoissonDistribution hooked");
#if defined(RUN_LOCALLY)
	return lcurandCreatePoissonDistribution(lambda, discrete_distribution);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t  discrete_distribution)
{
	TALLY_SPD_LOG("curandDestroyDistribution hooked");
#if defined(RUN_LOCALLY)
	return lcurandDestroyDistribution(discrete_distribution);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGeneratePoisson(curandGenerator_t  generator, unsigned int * outputPtr, size_t  n, double  lambda)
{
	TALLY_SPD_LOG("curandGeneratePoisson hooked");
#if defined(RUN_LOCALLY)
	return lcurandGeneratePoisson(generator, outputPtr, n, lambda);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGeneratePoissonMethod(curandGenerator_t  generator, unsigned int * outputPtr, size_t  n, double  lambda, curandMethod_t  method)
{
	TALLY_SPD_LOG("curandGeneratePoissonMethod hooked");
#if defined(RUN_LOCALLY)
	return lcurandGeneratePoissonMethod(generator, outputPtr, n, lambda, method);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateBinomial(curandGenerator_t  generator, unsigned int * outputPtr, size_t  num, unsigned int  n, double  p)
{
	TALLY_SPD_LOG("curandGenerateBinomial hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateBinomial(generator, outputPtr, num, n, p);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateBinomialMethod(curandGenerator_t  generator, unsigned int * outputPtr, size_t  num, unsigned int  n, double  p, curandMethod_t  method)
{
	TALLY_SPD_LOG("curandGenerateBinomialMethod hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateBinomialMethod(generator, outputPtr, num, n, p, method);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGenerateSeeds(curandGenerator_t  generator)
{
	TALLY_SPD_LOG("curandGenerateSeeds hooked");
#if defined(RUN_LOCALLY)
	return lcurandGenerateSeeds(generator);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t * vectors[], curandDirectionVectorSet_t  set)
{
	TALLY_SPD_LOG("curandGetDirectionVectors32 hooked");
#if defined(RUN_LOCALLY)
	return lcurandGetDirectionVectors32(vectors, set);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGetScrambleConstants32(unsigned int * *  constants)
{
	TALLY_SPD_LOG("curandGetScrambleConstants32 hooked");
#if defined(RUN_LOCALLY)
	return lcurandGetScrambleConstants32(constants);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t * vectors[], curandDirectionVectorSet_t  set)
{
	TALLY_SPD_LOG("curandGetDirectionVectors64 hooked");
#if defined(RUN_LOCALLY)
	return lcurandGetDirectionVectors64(vectors, set);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

curandStatus_t curandGetScrambleConstants64(unsigned long long * *  constants)
{
	TALLY_SPD_LOG("curandGetScrambleConstants64 hooked");
#if defined(RUN_LOCALLY)
	return lcurandGetScrambleConstants64(constants);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclMemAlloc(void**  ptr, size_t  size)
{
	TALLY_SPD_LOG("ncclMemAlloc hooked");
#if defined(RUN_LOCALLY)
	return lncclMemAlloc(ptr, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclMemAlloc(void**  ptr, size_t  size)
{
	TALLY_SPD_LOG("pncclMemAlloc hooked");
#if defined(RUN_LOCALLY)
	return lpncclMemAlloc(ptr, size);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclMemFree(void * ptr)
{
	TALLY_SPD_LOG("ncclMemFree hooked");
#if defined(RUN_LOCALLY)
	return lncclMemFree(ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclMemFree(void * ptr)
{
	TALLY_SPD_LOG("pncclMemFree hooked");
#if defined(RUN_LOCALLY)
	return lpncclMemFree(ptr);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclGetVersion(int * version)
{
	TALLY_SPD_LOG("ncclGetVersion hooked");
#if defined(RUN_LOCALLY)
	return lncclGetVersion(version);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclGetVersion(int * version)
{
	TALLY_SPD_LOG("pncclGetVersion hooked");
#if defined(RUN_LOCALLY)
	return lpncclGetVersion(version);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclGetUniqueId(ncclUniqueId*  uniqueId)
{
	TALLY_SPD_LOG("ncclGetUniqueId hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclGetUniqueId(uniqueId);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclGetUniqueIdArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLGETUNIQUEID;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclGetUniqueIdArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->uniqueId = uniqueId;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const ncclGetUniqueIdResponse*>(responsePayload);
			if (uniqueId) { *uniqueId = response->uniqueId; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclGetUniqueId);
	return err;
}

ncclResult_t pncclGetUniqueId(ncclUniqueId*  uniqueId)
{
	TALLY_SPD_LOG("pncclGetUniqueId hooked");
#if defined(RUN_LOCALLY)
	return lpncclGetUniqueId(uniqueId);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommInitRankConfig(ncclComm_t*  comm, int  nranks, ncclUniqueId  commId, int  rank, ncclConfig_t*  config)
{
	TALLY_SPD_LOG("pncclCommInitRankConfig hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommInitRankConfig(comm, nranks, commId, rank, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommInitRank(ncclComm_t*  comm, int  nranks, ncclUniqueId  commId, int  rank)
{
	TALLY_SPD_LOG("ncclCommInitRank hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclCommInitRank(comm, nranks, commId, rank);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclCommInitRankArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLCOMMINITRANK;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclCommInitRankArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->comm = comm;
			request->nranks = nranks;
			request->commId = commId;
			request->rank = rank;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const ncclCommInitRankResponse*>(responsePayload);
			if (comm) { *comm = response->comm; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclCommInitRank);
	return err;
}

ncclResult_t pncclCommInitRank(ncclComm_t*  comm, int  nranks, ncclUniqueId  commId, int  rank)
{
	TALLY_SPD_LOG("pncclCommInitRank hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommInitRank(comm, nranks, commId, rank);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommInitAll(ncclComm_t*  comm, int  ndev, const int*  devlist)
{
	TALLY_SPD_LOG("ncclCommInitAll hooked");
#if defined(RUN_LOCALLY)
	return lncclCommInitAll(comm, ndev, devlist);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommInitAll(ncclComm_t*  comm, int  ndev, const int*  devlist)
{
	TALLY_SPD_LOG("pncclCommInitAll hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommInitAll(comm, ndev, devlist);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommFinalize(ncclComm_t  comm)
{
	TALLY_SPD_LOG("ncclCommFinalize hooked");
#if defined(RUN_LOCALLY)
	return lncclCommFinalize(comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommFinalize(ncclComm_t  comm)
{
	TALLY_SPD_LOG("pncclCommFinalize hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommFinalize(comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommDestroy(ncclComm_t  comm)
{
	TALLY_SPD_LOG("ncclCommDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclCommDestroy(comm);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclCommDestroyArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLCOMMDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclCommDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->comm = comm;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclCommDestroy);
	return err;
}

ncclResult_t pncclCommDestroy(ncclComm_t  comm)
{
	TALLY_SPD_LOG("pncclCommDestroy hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommDestroy(comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommAbort(ncclComm_t  comm)
{
	TALLY_SPD_LOG("ncclCommAbort hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclCommAbort(comm);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclCommAbortArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLCOMMABORT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclCommAbortArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->comm = comm;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclCommAbort);
	return err;
}

ncclResult_t pncclCommAbort(ncclComm_t  comm)
{
	TALLY_SPD_LOG("pncclCommAbort hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommAbort(comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommSplit(ncclComm_t  comm, int  color, int  key, ncclComm_t * newcomm, ncclConfig_t*  config)
{
	TALLY_SPD_LOG("ncclCommSplit hooked");
#if defined(RUN_LOCALLY)
	return lncclCommSplit(comm, color, key, newcomm, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommSplit(ncclComm_t  comm, int  color, int  key, ncclComm_t * newcomm, ncclConfig_t*  config)
{
	TALLY_SPD_LOG("pncclCommSplit hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommSplit(comm, color, key, newcomm, config);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* ncclGetErrorString(ncclResult_t  result)
{
	TALLY_SPD_LOG("ncclGetErrorString hooked");
#if defined(RUN_LOCALLY)
	return lncclGetErrorString(result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* pncclGetErrorString(ncclResult_t  result)
{
	TALLY_SPD_LOG("pncclGetErrorString hooked");
#if defined(RUN_LOCALLY)
	return lpncclGetErrorString(result);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* ncclGetLastError(ncclComm_t  comm)
{
	TALLY_SPD_LOG("ncclGetLastError hooked");
#if defined(RUN_LOCALLY)
	return lncclGetLastError(comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

const char* pncclGetLastError(ncclComm_t  comm)
{
	TALLY_SPD_LOG("pncclGetLastError hooked");
#if defined(RUN_LOCALLY)
	return lpncclGetLastError(comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t  comm, ncclResult_t * asyncError)
{
	TALLY_SPD_LOG("ncclCommGetAsyncError hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclCommGetAsyncError(comm, asyncError);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclCommGetAsyncErrorArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLCOMMGETASYNCERROR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclCommGetAsyncErrorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->comm = comm;
			request->asyncError = asyncError;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const ncclCommGetAsyncErrorResponse*>(responsePayload);
			if (asyncError) { *asyncError = response->asyncError; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclCommGetAsyncError);
	return err;
}

ncclResult_t pncclCommGetAsyncError(ncclComm_t  comm, ncclResult_t * asyncError)
{
	TALLY_SPD_LOG("pncclCommGetAsyncError hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommGetAsyncError(comm, asyncError);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommCount(const ncclComm_t  comm, int*  count)
{
	TALLY_SPD_LOG("ncclCommCount hooked");
#if defined(RUN_LOCALLY)
	return lncclCommCount(comm, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommCount(const ncclComm_t  comm, int*  count)
{
	TALLY_SPD_LOG("pncclCommCount hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommCount(comm, count);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommCuDevice(const ncclComm_t  comm, int*  device)
{
	TALLY_SPD_LOG("ncclCommCuDevice hooked");
#if defined(RUN_LOCALLY)
	return lncclCommCuDevice(comm, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommCuDevice(const ncclComm_t  comm, int*  device)
{
	TALLY_SPD_LOG("pncclCommCuDevice hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommCuDevice(comm, device);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommUserRank(const ncclComm_t  comm, int*  rank)
{
	TALLY_SPD_LOG("ncclCommUserRank hooked");
#if defined(RUN_LOCALLY)
	return lncclCommUserRank(comm, rank);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommUserRank(const ncclComm_t  comm, int*  rank)
{
	TALLY_SPD_LOG("pncclCommUserRank hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommUserRank(comm, rank);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t * op, void * scalar, ncclDataType_t  datatype, ncclScalarResidence_t  residence, ncclComm_t  comm)
{
	TALLY_SPD_LOG("ncclRedOpCreatePreMulSum hooked");
#if defined(RUN_LOCALLY)
	return lncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t * op, void * scalar, ncclDataType_t  datatype, ncclScalarResidence_t  residence, ncclComm_t  comm)
{
	TALLY_SPD_LOG("pncclRedOpCreatePreMulSum hooked");
#if defined(RUN_LOCALLY)
	return lpncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t  op, ncclComm_t  comm)
{
	TALLY_SPD_LOG("ncclRedOpDestroy hooked");
#if defined(RUN_LOCALLY)
	return lncclRedOpDestroy(op, comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclRedOpDestroy(ncclRedOp_t  op, ncclComm_t  comm)
{
	TALLY_SPD_LOG("pncclRedOpDestroy hooked");
#if defined(RUN_LOCALLY)
	return lpncclRedOpDestroy(op, comm);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclReduce(const void*  sendbuff, void*  recvbuff, size_t  count, ncclDataType_t  datatype, ncclRedOp_t  op, int  root, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclReduce hooked");
#if defined(RUN_LOCALLY)
	return lncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclReduce(const void*  sendbuff, void*  recvbuff, size_t  count, ncclDataType_t  datatype, ncclRedOp_t  op, int  root, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclReduce hooked");
#if defined(RUN_LOCALLY)
	return lpncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclBcast(void*  buff, size_t  count, ncclDataType_t  datatype, int  root, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclBcast hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclBcast(buff, count, datatype, root, comm, stream);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclBcastArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLBCAST;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclBcastArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->buff = buff;
			request->count = count;
			request->datatype = datatype;
			request->root = root;
			request->comm = comm;
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclBcast);
	return err;
}

ncclResult_t pncclBcast(void*  buff, size_t  count, ncclDataType_t  datatype, int  root, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclBcast hooked");
#if defined(RUN_LOCALLY)
	return lpncclBcast(buff, count, datatype, root, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclBroadcast(const void*  sendbuff, void*  recvbuff, size_t  count, ncclDataType_t  datatype, int  root, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclBroadcast hooked");
#if defined(RUN_LOCALLY)
	return lncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclBroadcast(const void*  sendbuff, void*  recvbuff, size_t  count, ncclDataType_t  datatype, int  root, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclBroadcast hooked");
#if defined(RUN_LOCALLY)
	return lpncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclAllReduce(const void*  sendbuff, void*  recvbuff, size_t  count, ncclDataType_t  datatype, ncclRedOp_t  op, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclAllReduce hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclAllReduceArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLALLREDUCE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclAllReduceArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->sendbuff = const_cast<void *>(sendbuff);
			request->recvbuff = recvbuff;
			request->count = count;
			request->datatype = datatype;
			request->op = op;
			request->comm = comm;
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclAllReduce);
	return err;
}

ncclResult_t pncclAllReduce(const void*  sendbuff, void*  recvbuff, size_t  count, ncclDataType_t  datatype, ncclRedOp_t  op, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclAllReduce hooked");
#if defined(RUN_LOCALLY)
	return lpncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclReduceScatter(const void*  sendbuff, void*  recvbuff, size_t  recvcount, ncclDataType_t  datatype, ncclRedOp_t  op, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclReduceScatter hooked");
#if defined(RUN_LOCALLY)
	return lncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclReduceScatter(const void*  sendbuff, void*  recvbuff, size_t  recvcount, ncclDataType_t  datatype, ncclRedOp_t  op, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclReduceScatter hooked");
#if defined(RUN_LOCALLY)
	return lpncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclAllGather(const void*  sendbuff, void*  recvbuff, size_t  sendcount, ncclDataType_t  datatype, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclAllGather hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclAllGatherArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLALLGATHER;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclAllGatherArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->sendbuff = const_cast<void *>(sendbuff);
			request->recvbuff = recvbuff;
			request->sendcount = sendcount;
			request->datatype = datatype;
			request->comm = comm;
			request->stream = stream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclAllGather);
	return err;
}

ncclResult_t pncclAllGather(const void*  sendbuff, void*  recvbuff, size_t  sendcount, ncclDataType_t  datatype, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclAllGather hooked");
#if defined(RUN_LOCALLY)
	return lpncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclSend(const void*  sendbuff, size_t  count, ncclDataType_t  datatype, int  peer, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclSend hooked");
#if defined(RUN_LOCALLY)
	return lncclSend(sendbuff, count, datatype, peer, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclSend(const void*  sendbuff, size_t  count, ncclDataType_t  datatype, int  peer, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclSend hooked");
#if defined(RUN_LOCALLY)
	return lpncclSend(sendbuff, count, datatype, peer, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclRecv(void*  recvbuff, size_t  count, ncclDataType_t  datatype, int  peer, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("pncclRecv hooked");
#if defined(RUN_LOCALLY)
	return lpncclRecv(recvbuff, count, datatype, peer, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclRecv(void*  recvbuff, size_t  count, ncclDataType_t  datatype, int  peer, ncclComm_t  comm, cudaStream_t  stream)
{
	TALLY_SPD_LOG("ncclRecv hooked");
#if defined(RUN_LOCALLY)
	return lncclRecv(recvbuff, count, datatype, peer, comm, stream);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclGroupStart()
{
	TALLY_SPD_LOG("ncclGroupStart hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclGroupStart();
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclGroupStartArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLGROUPSTART;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclGroupStartArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclGroupStart);
	return err;
}

ncclResult_t pncclGroupStart()
{
	TALLY_SPD_LOG("pncclGroupStart hooked");
#if defined(RUN_LOCALLY)
	return lpncclGroupStart();
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclGroupEnd()
{
	TALLY_SPD_LOG("ncclGroupEnd hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lncclGroupEnd();
#else

    ncclResult_t err;

    IOX_CLIENT_ACQUIRE_LOCK;
    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(ncclGroupEndArg), alignof(MessageHeader_t))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::NCCLGROUPEND;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (ncclGroupEndArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            
            auto response = static_cast<const ncclResult_t*>(responsePayload);
            err = *response;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(ncclGroupEnd);
	return err;
}

ncclResult_t pncclGroupEnd()
{
	TALLY_SPD_LOG("pncclGroupEnd hooked");
#if defined(RUN_LOCALLY)
	return lpncclGroupEnd();
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommRegister(const ncclComm_t  comm, void*  buff, size_t  size, void**  handle)
{
	TALLY_SPD_LOG("ncclCommRegister hooked");
#if defined(RUN_LOCALLY)
	return lncclCommRegister(comm, buff, size, handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommRegister(const ncclComm_t  comm, void*  buff, size_t  size, void**  handle)
{
	TALLY_SPD_LOG("pncclCommRegister hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommRegister(comm, buff, size, handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t ncclCommDeregister(const ncclComm_t  comm, void*  handle)
{
	TALLY_SPD_LOG("ncclCommDeregister hooked");
#if defined(RUN_LOCALLY)
	return lncclCommDeregister(comm, handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}

ncclResult_t pncclCommDeregister(const ncclComm_t  comm, void*  handle)
{
	TALLY_SPD_LOG("pncclCommDeregister hooked");
#if defined(RUN_LOCALLY)
	return lpncclCommDeregister(comm, handle);
#else
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
#endif
}



}

