
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
	CUresult res = 		lcuGetErrorString(error, pStr);
	return res;
}

CUresult cuGetErrorName(CUresult  error, const char ** pStr)
{
	TALLY_LOG("cuGetErrorName hooked");
	CUresult res = 		lcuGetErrorName(error, pStr);
	return res;
}

CUresult cuInit(unsigned int  Flags)
{
	TALLY_LOG("cuInit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuInit(Flags);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuInitArg), alignof(cuInitArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUINIT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuInitArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_CLIENT_TRACE_API_CALL(cuInit);
	return err;
}

CUresult cuDriverGetVersion(int * driverVersion)
{
	TALLY_LOG("cuDriverGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDriverGetVersion(driverVersion);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDriverGetVersionArg), alignof(cuDriverGetVersionArg))
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
	TALLY_LOG("cuDeviceGet hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGet(device, ordinal);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetArg), alignof(cuDeviceGetArg))
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
	TALLY_LOG("cuDeviceGetCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetCount(count);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetCountArg), alignof(cuDeviceGetCountArg))
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

CUresult cuDeviceGetName(char * name, int  len, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceGetUuid(CUuuid * uuid, CUdevice  dev)
{
	TALLY_LOG("cuDeviceGetUuid hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetUuid(uuid, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetUuidArg), alignof(cuDeviceGetUuidArg))
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
	TALLY_LOG("cuDeviceGetUuid_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetUuid_v2(uuid, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetUuid_v2Arg), alignof(cuDeviceGetUuid_v2Arg))
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
	TALLY_LOG("cuDeviceGetLuid hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetLuid(luid, deviceNodeMask, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetLuidArg), alignof(cuDeviceGetLuidArg))
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
	TALLY_LOG("cuDeviceTotalMem_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceTotalMem_v2(bytes, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceTotalMem_v2Arg), alignof(cuDeviceTotalMem_v2Arg))
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
	TALLY_LOG("cuDeviceGetTexture1DLinearMaxWidth hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetTexture1DLinearMaxWidthArg), alignof(cuDeviceGetTexture1DLinearMaxWidthArg))
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
	TALLY_LOG("cuDeviceGetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetAttribute(pi, attrib, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetAttributeArg), alignof(cuDeviceGetAttributeArg))
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
	TALLY_LOG("cuDeviceGetNvSciSyncAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuDeviceSetMemPool(CUdevice  dev, CUmemoryPool  pool)
{
	TALLY_LOG("cuDeviceSetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceSetMemPool(dev, pool);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceSetMemPoolArg), alignof(cuDeviceSetMemPoolArg))
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
	TALLY_LOG("cuDeviceGetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetMemPool(pool, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetMemPoolArg), alignof(cuDeviceGetMemPoolArg))
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
	TALLY_LOG("cuDeviceGetDefaultMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetDefaultMemPool(pool_out, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetDefaultMemPoolArg), alignof(cuDeviceGetDefaultMemPoolArg))
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
	TALLY_LOG("cuDeviceGetExecAffinitySupport hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetExecAffinitySupport(pi, type, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetExecAffinitySupportArg), alignof(cuDeviceGetExecAffinitySupportArg))
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
	TALLY_LOG("cuFlushGPUDirectRDMAWrites hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuFlushGPUDirectRDMAWrites(target, scope);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuFlushGPUDirectRDMAWritesArg), alignof(cuFlushGPUDirectRDMAWritesArg))
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
	TALLY_LOG("cuDeviceGetProperties hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceGetProperties(prop, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceGetPropertiesArg), alignof(cuDeviceGetPropertiesArg))
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
	TALLY_LOG("cuDeviceComputeCapability hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDeviceComputeCapability(major, minor, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDeviceComputeCapabilityArg), alignof(cuDeviceComputeCapabilityArg))
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
	TALLY_LOG("cuDevicePrimaryCtxRetain hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxRetain(pctx, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxRetainArg), alignof(cuDevicePrimaryCtxRetainArg))
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
	TALLY_LOG("cuDevicePrimaryCtxRelease_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxRelease_v2(dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxRelease_v2Arg), alignof(cuDevicePrimaryCtxRelease_v2Arg))
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

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice  dev, unsigned int  flags)
{
	TALLY_LOG("cuDevicePrimaryCtxSetFlags_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxSetFlags_v2(dev, flags);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxSetFlags_v2Arg), alignof(cuDevicePrimaryCtxSetFlags_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXSETFLAGS_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDevicePrimaryCtxSetFlags_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dev = dev;
			request->flags = flags;

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
	TALLY_CLIENT_TRACE_API_CALL(cuDevicePrimaryCtxSetFlags_v2);
	return err;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice  dev, unsigned int * flags, int * active)
{
	TALLY_LOG("cuDevicePrimaryCtxGetState hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxGetState(dev, flags, active);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxGetStateArg), alignof(cuDevicePrimaryCtxGetStateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDEVICEPRIMARYCTXGETSTATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuDevicePrimaryCtxGetStateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dev = dev;
			request->flags = flags;
			request->active = active;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuDevicePrimaryCtxGetStateResponse*>(responsePayload);
			if (flags) { *flags = response->flags; }
			if (active) { *active = response->active; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuDevicePrimaryCtxGetState);
	return err;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice  dev)
{
	TALLY_LOG("cuDevicePrimaryCtxReset_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDevicePrimaryCtxReset_v2(dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDevicePrimaryCtxReset_v2Arg), alignof(cuDevicePrimaryCtxReset_v2Arg))
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

CUresult cuCtxCreate_v2(CUcontext * pctx, unsigned int  flags, CUdevice  dev)
{
	TALLY_LOG("cuCtxCreate_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxCreate_v2(pctx, flags, dev);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxCreate_v2Arg), alignof(cuCtxCreate_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXCREATE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxCreate_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pctx = pctx;
			request->flags = flags;
			request->dev = dev;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxCreate_v2Response*>(responsePayload);
			if (pctx) { *pctx = response->pctx; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxCreate_v2);
	return err;
}

CUresult cuCtxCreate_v3(CUcontext * pctx, CUexecAffinityParam * paramsArray, int  numParams, unsigned int  flags, CUdevice  dev)
{
	TALLY_LOG("cuCtxCreate_v3 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCtxDestroy_v2(CUcontext  ctx)
{
	TALLY_LOG("cuCtxDestroy_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxDestroy_v2(ctx);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxDestroy_v2Arg), alignof(cuCtxDestroy_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXDESTROY_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxDestroy_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_CLIENT_TRACE_API_CALL(cuCtxDestroy_v2);
	return err;
}

CUresult cuCtxPushCurrent_v2(CUcontext  ctx)
{
	TALLY_LOG("cuCtxPushCurrent_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxPushCurrent_v2(ctx);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxPushCurrent_v2Arg), alignof(cuCtxPushCurrent_v2Arg))
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
	TALLY_LOG("cuCtxPopCurrent_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxPopCurrent_v2(pctx);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxPopCurrent_v2Arg), alignof(cuCtxPopCurrent_v2Arg))
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
	TALLY_LOG("cuCtxSetCurrent hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetCurrent(ctx);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetCurrentArg), alignof(cuCtxSetCurrentArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXSETCURRENT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxSetCurrentArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_CLIENT_TRACE_API_CALL(cuCtxSetCurrent);
	return err;
}

CUresult cuCtxGetCurrent(CUcontext * pctx)
{
	TALLY_LOG("cuCtxGetCurrent hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetCurrent(pctx);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetCurrentArg), alignof(cuCtxGetCurrentArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETCURRENT;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetCurrentArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pctx = pctx;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetCurrentResponse*>(responsePayload);
			if (pctx) { *pctx = response->pctx; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetCurrent);
	return err;
}

CUresult cuCtxGetDevice(CUdevice * device)
{
	TALLY_LOG("cuCtxGetDevice hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetDevice(device);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetDeviceArg), alignof(cuCtxGetDeviceArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETDEVICE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetDeviceArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetDeviceResponse*>(responsePayload);
			if (device) { *device = response->device; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetDevice);
	return err;
}

CUresult cuCtxGetFlags(unsigned int * flags)
{
	TALLY_LOG("cuCtxGetFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetFlags(flags);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetFlagsArg), alignof(cuCtxGetFlagsArg))
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
	TALLY_LOG("cuCtxSetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCtxGetId(CUcontext  ctx, unsigned long long * ctxId)
{
	TALLY_LOG("cuCtxGetId hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCtxSynchronize()
{
	TALLY_LOG("cuCtxSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSynchronize();
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSynchronizeArg), alignof(cuCtxSynchronizeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXSYNCHRONIZE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxSynchronizeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

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
	TALLY_CLIENT_TRACE_API_CALL(cuCtxSynchronize);
	return err;
}

CUresult cuCtxSetLimit(CUlimit  limit, size_t  value)
{
	TALLY_LOG("cuCtxSetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetLimit(limit, value);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetLimitArg), alignof(cuCtxSetLimitArg))
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
	TALLY_LOG("cuCtxGetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetLimit(pvalue, limit);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetLimitArg), alignof(cuCtxGetLimitArg))
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
	TALLY_LOG("cuCtxGetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetCacheConfig(pconfig);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetCacheConfigArg), alignof(cuCtxGetCacheConfigArg))
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
	TALLY_LOG("cuCtxSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetCacheConfig(config);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetCacheConfigArg), alignof(cuCtxSetCacheConfigArg))
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
	TALLY_LOG("cuCtxGetSharedMemConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetSharedMemConfig(pConfig);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetSharedMemConfigArg), alignof(cuCtxGetSharedMemConfigArg))
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
	TALLY_LOG("cuCtxSetSharedMemConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxSetSharedMemConfig(config);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxSetSharedMemConfigArg), alignof(cuCtxSetSharedMemConfigArg))
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

CUresult cuCtxGetApiVersion(CUcontext  ctx, unsigned int * version)
{
	TALLY_LOG("cuCtxGetApiVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetApiVersion(ctx, version);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetApiVersionArg), alignof(cuCtxGetApiVersionArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUCTXGETAPIVERSION;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuCtxGetApiVersionArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->ctx = ctx;
			request->version = version;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cuCtxGetApiVersionResponse*>(responsePayload);
			if (version) { *version = response->version; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cuCtxGetApiVersion);
	return err;
}

CUresult cuCtxGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	TALLY_LOG("cuCtxGetStreamPriorityRange hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetStreamPriorityRange(leastPriority, greatestPriority);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetStreamPriorityRangeArg), alignof(cuCtxGetStreamPriorityRangeArg))
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
	TALLY_LOG("cuCtxResetPersistingL2Cache hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxResetPersistingL2Cache();
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxResetPersistingL2CacheArg), alignof(cuCtxResetPersistingL2CacheArg))
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
	TALLY_LOG("cuCtxGetExecAffinity hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxGetExecAffinity(pExecAffinity, type);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxGetExecAffinityArg), alignof(cuCtxGetExecAffinityArg))
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
	TALLY_LOG("cuCtxAttach hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxAttach(pctx, flags);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxAttachArg), alignof(cuCtxAttachArg))
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
	TALLY_LOG("cuCtxDetach hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuCtxDetach(ctx);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuCtxDetachArg), alignof(cuCtxDetachArg))
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
	TALLY_LOG("cuModuleLoad hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuModuleUnload(hmod);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuModuleUnloadArg), alignof(cuModuleUnloadArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMODULEUNLOAD;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuModuleUnloadArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->hmod = hmod;

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
	TALLY_CLIENT_TRACE_API_CALL(cuModuleUnload);
	return err;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode * mode)
{
	TALLY_LOG("cuModuleGetLoadingMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule  hmod, const char * name)
{
	TALLY_LOG("cuModuleGetGlobal_v2 hooked");
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

CUresult cuLibraryLoadData(CUlibrary * library, const void * code, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void**  libraryOptionValues, unsigned int  numLibraryOptions)
{
	TALLY_LOG("cuLibraryLoadData hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryLoadFromFile(CUlibrary * library, const char * fileName, CUjit_option * jitOptions, void ** jitOptionsValues, unsigned int  numJitOptions, CUlibraryOption * libraryOptions, void ** libraryOptionValues, unsigned int  numLibraryOptions)
{
	TALLY_LOG("cuLibraryLoadFromFile hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryUnload(CUlibrary  library)
{
	TALLY_LOG("cuLibraryUnload hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryGetKernel(CUkernel * pKernel, CUlibrary  library, const char * name)
{
	TALLY_LOG("cuLibraryGetKernel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryGetModule(CUmodule * pMod, CUlibrary  library)
{
	TALLY_LOG("cuLibraryGetModule hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuKernelGetFunction(CUfunction * pFunc, CUkernel  kernel)
{
	TALLY_LOG("cuKernelGetFunction hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryGetGlobal(CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name)
{
	TALLY_LOG("cuLibraryGetGlobal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryGetManaged(CUdeviceptr * dptr, size_t * bytes, CUlibrary  library, const char * name)
{
	TALLY_LOG("cuLibraryGetManaged hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuLibraryGetUnifiedFunction(void ** fptr, CUlibrary  library, const char * symbol)
{
	TALLY_LOG("cuLibraryGetUnifiedFunction hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuKernelGetAttribute(int * pi, CUfunction_attribute  attrib, CUkernel  kernel, CUdevice  dev)
{
	TALLY_LOG("cuKernelGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuKernelSetAttribute(CUfunction_attribute  attrib, int  val, CUkernel  kernel, CUdevice  dev)
{
	TALLY_LOG("cuKernelSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuKernelSetCacheConfig(CUkernel  kernel, CUfunc_cache  config, CUdevice  dev)
{
	TALLY_LOG("cuKernelSetCacheConfig hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemFree_v2(dptr);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemFree_v2Arg), alignof(cuMemFree_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMFREE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemFree_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dptr = dptr;

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
	TALLY_CLIENT_TRACE_API_CALL(cuMemFree_v2);
	return err;
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemcpyAsync(dst, src, ByteCount, hStream);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemcpyAsyncArg), alignof(cuMemcpyAsyncArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUMEMCPYASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cuMemcpyAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->dst = dst;
			request->src = src;
			request->ByteCount = ByteCount;
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
	TALLY_CLIENT_TRACE_API_CALL(cuMemcpyAsync);
	return err;
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

CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUarray  array, CUdevice  device)
{
	TALLY_LOG("cuArrayGetMemoryRequirements hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS * memoryRequirements, CUmipmappedArray  mipmap, CUdevice  device)
{
	TALLY_LOG("cuMipmappedArrayGetMemoryRequirements hooked");
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

CUresult cuMemGetHandleForAddressRange(void * handle, CUdeviceptr  dptr, size_t  size, CUmemRangeHandleType  handleType, unsigned long long  flags)
{
	TALLY_LOG("cuMemGetHandleForAddressRange hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuMemAllocFromPoolAsyncArg), alignof(cuMemAllocFromPoolAsyncArg))
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

CUresult cuMulticastCreate(CUmemGenericAllocationHandle * mcHandle, const CUmulticastObjectProp * prop)
{
	TALLY_LOG("cuMulticastCreate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle  mcHandle, CUdevice  dev)
{
	TALLY_LOG("cuMulticastAddDevice hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMulticastBindMem(CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUmemGenericAllocationHandle  memHandle, size_t  memOffset, size_t  size, unsigned long long  flags)
{
	TALLY_LOG("cuMulticastBindMem hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle  mcHandle, size_t  mcOffset, CUdeviceptr  memptr, size_t  size, unsigned long long  flags)
{
	TALLY_LOG("cuMulticastBindAddr hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMulticastUnbind(CUmemGenericAllocationHandle  mcHandle, CUdevice  dev, size_t  mcOffset, size_t  size)
{
	TALLY_LOG("cuMulticastUnbind hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMulticastGetGranularity(size_t * granularity, const CUmulticastObjectProp * prop, CUmulticastGranularity_flags  option)
{
	TALLY_LOG("cuMulticastGetGranularity hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPrefetchAsync(CUdeviceptr  devPtr, size_t  count, CUdevice  dstDevice, CUstream  hStream)
{
	TALLY_LOG("cuMemPrefetchAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemPrefetchAsync_v2(CUdeviceptr  devPtr, size_t  count, CUmemLocation  location, unsigned int  flags, CUstream  hStream)
{
	TALLY_LOG("cuMemPrefetchAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAdvise(CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUdevice  device)
{
	TALLY_LOG("cuMemAdvise hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuMemAdvise_v2(CUdeviceptr  devPtr, size_t  count, CUmem_advise  advice, CUmemLocation  location)
{
	TALLY_LOG("cuMemAdvise_v2 hooked");
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

CUresult cuStreamGetId(CUstream  hStream, unsigned long long * streamId)
{
	TALLY_LOG("cuStreamGetId hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcuDestroyExternalMemory(extMem);
#else

    CUresult err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cuDestroyExternalMemoryArg), alignof(cuDestroyExternalMemoryArg))
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

CUresult cuStreamWaitValue32_v2(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWaitValue32_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWaitValue64_v2(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWaitValue64_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWriteValue32_v2(CUstream  stream, CUdeviceptr  addr, cuuint32_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWriteValue32_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamWriteValue64_v2(CUstream  stream, CUdeviceptr  addr, cuuint64_t  value, unsigned int  flags)
{
	TALLY_LOG("cuStreamWriteValue64_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuStreamBatchMemOp_v2(CUstream  stream, unsigned int  count, CUstreamBatchMemOpParams * paramArray, unsigned int  flags)
{
	TALLY_LOG("cuStreamBatchMemOp_v2 hooked");
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

CUresult cuLaunchKernelEx(const CUlaunchConfig * config, CUfunction  f, void ** kernelParams, void ** extra)
{
	TALLY_LOG("cuLaunchKernelEx hooked");
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

CUresult cuGraphAddKernelNode_v2(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddKernelNode_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode  hNode, CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphKernelNodeGetParams_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphKernelNodeSetParams_v2 hooked");
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

CUresult cuGraphAddBatchMemOpNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphAddBatchMemOpNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode  hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams_out)
{
	TALLY_LOG("cuGraphBatchMemOpNodeGetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphBatchMemOpNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExecBatchMemOpNodeSetParams hooked");
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

CUresult cuGraphInstantiateWithFlags(CUgraphExec * phGraphExec, CUgraph  hGraph, unsigned long long  flags)
{
	TALLY_LOG("cuGraphInstantiateWithFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphInstantiateWithParams(CUgraphExec * phGraphExec, CUgraph  hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS * instantiateParams)
{
	TALLY_LOG("cuGraphInstantiateWithParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecGetFlags(CUgraphExec  hGraphExec, cuuint64_t * flags)
{
	TALLY_LOG("cuGraphExecGetFlags hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec  hGraphExec, CUgraphNode  hNode, const CUDA_KERNEL_NODE_PARAMS * nodeParams)
{
	TALLY_LOG("cuGraphExecKernelNodeSetParams_v2 hooked");
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

CUresult cuGraphNodeSetEnabled(CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int  isEnabled)
{
	TALLY_LOG("cuGraphNodeSetEnabled hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphNodeGetEnabled(CUgraphExec  hGraphExec, CUgraphNode  hNode, unsigned int * isEnabled)
{
	TALLY_LOG("cuGraphNodeGetEnabled hooked");
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

CUresult cuGraphExecUpdate_v2(CUgraphExec  hGraphExec, CUgraph  hGraph, CUgraphExecUpdateResultInfo * resultInfo)
{
	TALLY_LOG("cuGraphExecUpdate_v2 hooked");
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

CUresult cuGraphAddNode(CUgraphNode * phGraphNode, CUgraph  hGraph, const CUgraphNode * dependencies, size_t  numDependencies, CUgraphNodeParams * nodeParams)
{
	TALLY_LOG("cuGraphAddNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphNodeSetParams(CUgraphNode  hNode, CUgraphNodeParams * nodeParams)
{
	TALLY_LOG("cuGraphNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGraphExecNodeSetParams(CUgraphExec  hGraphExec, CUgraphNode  hNode, CUgraphNodeParams * nodeParams)
{
	TALLY_LOG("cuGraphExecNodeSetParams hooked");
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

CUresult cuOccupancyMaxPotentialClusterSize(int * clusterSize, CUfunction  func, const CUlaunchConfig * config)
{
	TALLY_LOG("cuOccupancyMaxPotentialClusterSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuOccupancyMaxActiveClusters(int * numClusters, CUfunction  func, const CUlaunchConfig * config)
{
	TALLY_LOG("cuOccupancyMaxActiveClusters hooked");
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

CUresult cuTensorMapEncodeTiled(CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const cuuint32_t * boxDim, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill)
{
	TALLY_LOG("cuTensorMapEncodeTiled hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTensorMapEncodeIm2col(CUtensorMap * tensorMap, CUtensorMapDataType  tensorDataType, cuuint32_t  tensorRank, void * globalAddress, const cuuint64_t * globalDim, const cuuint64_t * globalStrides, const int * pixelBoxLowerCorner, const int * pixelBoxUpperCorner, cuuint32_t  channelsPerPixel, cuuint32_t  pixelsPerColumn, const cuuint32_t * elementStrides, CUtensorMapInterleave  interleave, CUtensorMapSwizzle  swizzle, CUtensorMapL2promotion  l2Promotion, CUtensorMapFloatOOBfill  oobFill)
{
	TALLY_LOG("cuTensorMapEncodeIm2col hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuTensorMapReplaceAddress(CUtensorMap * tensorMap, void * globalAddress)
{
	TALLY_LOG("cuTensorMapReplaceAddress hooked");
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

CUresult cuGetProcAddress_v2(const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags, CUdriverProcAddressQueryResult * symbolStatus)
{
	TALLY_LOG("cuGetProcAddress_v2 hooked");
	CUresult res = 		lcuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
	return res;
}

CUresult cuCoredumpGetAttribute(CUcoredumpSettings  attrib, void*  value, size_t * size)
{
	TALLY_LOG("cuCoredumpGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings  attrib, void * value, size_t * size)
{
	TALLY_LOG("cuCoredumpGetAttributeGlobal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCoredumpSetAttribute(CUcoredumpSettings  attrib, void*  value, size_t * size)
{
	TALLY_LOG("cuCoredumpSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings  attrib, void * value, size_t * size)
{
	TALLY_LOG("cuCoredumpSetAttributeGlobal hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

CUresult cuGetExportTable(const void ** ppExportTable, const CUuuid * pExportTableId)
{
	TALLY_LOG("cuGetExportTable hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceReset()
{
	TALLY_LOG("cudaDeviceReset hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceReset();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceResetArg), alignof(cudaDeviceResetArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceReset);
	return err;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit  limit, size_t  value)
{
	TALLY_LOG("cudaDeviceSetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetLimit(limit, value);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetLimitArg), alignof(cudaDeviceSetLimitArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetLimit);
	return err;
}

cudaError_t cudaDeviceGetLimit(size_t * pValue, enum cudaLimit  limit)
{
	TALLY_LOG("cudaDeviceGetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetLimit(pValue, limit);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetLimitArg), alignof(cudaDeviceGetLimitArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetLimit);
	return err;
}

cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const struct cudaChannelFormatDesc * fmtDesc, int  device)
{
	TALLY_LOG("cudaDeviceGetTexture1DLinearMaxWidth hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
	TALLY_LOG("cudaDeviceGetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetCacheConfig(pCacheConfig);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetCacheConfigArg), alignof(cudaDeviceGetCacheConfigArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetCacheConfig);
	return err;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority)
{
	TALLY_LOG("cudaDeviceGetStreamPriorityRange hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetStreamPriorityRangeArg), alignof(cudaDeviceGetStreamPriorityRangeArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetStreamPriorityRange);
	return err;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache  cacheConfig)
{
	TALLY_LOG("cudaDeviceSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetCacheConfig(cacheConfig);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetCacheConfigArg), alignof(cudaDeviceSetCacheConfigArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetCacheConfig);
	return err;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig)
{
	TALLY_LOG("cudaDeviceGetSharedMemConfig hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig  config)
{
	TALLY_LOG("cudaDeviceSetSharedMemConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetSharedMemConfig(config);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetSharedMemConfigArg), alignof(cudaDeviceSetSharedMemConfigArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetSharedMemConfig);
	return err;
}

cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId)
{
	TALLY_LOG("cudaDeviceGetByPCIBusId hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int  len, int  device)
{
	TALLY_LOG("cudaDeviceGetPCIBusId hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetPCIBusId(pciBusId, len, device);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetPCIBusIdArg), alignof(cudaDeviceGetPCIBusIdArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetPCIBusId);
	return err;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t  event)
{
	TALLY_LOG("cudaIpcGetEventHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcGetEventHandle(handle, event);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcGetEventHandleArg), alignof(cudaIpcGetEventHandleArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcGetEventHandle);
	return err;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t  handle)
{
	TALLY_LOG("cudaIpcOpenEventHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcOpenEventHandle(event, handle);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcOpenEventHandleArg), alignof(cudaIpcOpenEventHandleArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcOpenEventHandle);
	return err;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr)
{
	TALLY_LOG("cudaIpcGetMemHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcGetMemHandle(handle, devPtr);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcGetMemHandleArg), alignof(cudaIpcGetMemHandleArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcGetMemHandle);
	return err;
}

cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t  handle, unsigned int  flags)
{
	TALLY_LOG("cudaIpcOpenMemHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcOpenMemHandle(devPtr, handle, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcOpenMemHandleArg), alignof(cudaIpcOpenMemHandleArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcOpenMemHandle);
	return err;
}

cudaError_t cudaIpcCloseMemHandle(void * devPtr)
{
	TALLY_LOG("cudaIpcCloseMemHandle hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaIpcCloseMemHandle(devPtr);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaIpcCloseMemHandleArg), alignof(cudaIpcCloseMemHandleArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaIpcCloseMemHandle);
	return err;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget  target, enum cudaFlushGPUDirectRDMAWritesScope  scope)
{
	TALLY_LOG("cudaDeviceFlushGPUDirectRDMAWrites hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceFlushGPUDirectRDMAWrites(target, scope);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceFlushGPUDirectRDMAWritesArg), alignof(cudaDeviceFlushGPUDirectRDMAWritesArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceFlushGPUDirectRDMAWrites);
	return err;
}

cudaError_t cudaThreadExit()
{
	TALLY_LOG("cudaThreadExit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadExit();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadExitArg), alignof(cudaThreadExitArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadExit);
	return err;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit  limit, size_t  value)
{
	TALLY_LOG("cudaThreadSetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadSetLimit(limit, value);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadSetLimitArg), alignof(cudaThreadSetLimitArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadSetLimit);
	return err;
}

cudaError_t cudaThreadGetLimit(size_t * pValue, enum cudaLimit  limit)
{
	TALLY_LOG("cudaThreadGetLimit hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadGetLimit(pValue, limit);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadGetLimitArg), alignof(cudaThreadGetLimitArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadGetLimit);
	return err;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * pCacheConfig)
{
	TALLY_LOG("cudaThreadGetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadGetCacheConfig(pCacheConfig);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadGetCacheConfigArg), alignof(cudaThreadGetCacheConfigArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadGetCacheConfig);
	return err;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache  cacheConfig)
{
	TALLY_LOG("cudaThreadSetCacheConfig hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaThreadSetCacheConfig(cacheConfig);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaThreadSetCacheConfigArg), alignof(cudaThreadSetCacheConfigArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaThreadSetCacheConfig);
	return err;
}

cudaError_t cudaGetLastError()
{
	TALLY_LOG("cudaGetLastError hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGetLastError();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGetLastErrorArg), alignof(cudaGetLastErrorArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAGETLASTERROR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaGetLastErrorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGetLastError);
	return err;
}

cudaError_t cudaPeekAtLastError()
{
	TALLY_LOG("cudaPeekAtLastError hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaPeekAtLastError();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaPeekAtLastErrorArg), alignof(cudaPeekAtLastErrorArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAPEEKATLASTERROR;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaPeekAtLastErrorArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));

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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaPeekAtLastError);
	return err;
}

const char* cudaGetErrorName(cudaError_t  error)
{
	TALLY_LOG("cudaGetErrorName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

const char* cudaGetErrorString(cudaError_t  error)
{
	TALLY_LOG("cudaGetErrorString hooked");
	const char* res = 		lcudaGetErrorString(error);
	return res;
}

cudaError_t cudaGetDeviceCount(int * count)
{
	TALLY_LOG("cudaGetDeviceCount hooked");
	cudaError_t res = 		lcudaGetDeviceCount(count);
	return res;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp * prop, int  device)
{
	TALLY_LOG("cudaGetDeviceProperties_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGetDeviceProperties_v2(prop, device);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGetDeviceProperties_v2Arg), alignof(cudaGetDeviceProperties_v2Arg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGetDeviceProperties_v2);
	return err;
}

cudaError_t cudaDeviceGetAttribute(int * value, enum cudaDeviceAttr  attr, int  device)
{
	TALLY_LOG("cudaDeviceGetAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetAttribute(value, attr, device);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetAttributeArg), alignof(cudaDeviceGetAttributeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDADEVICEGETATTRIBUTE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaDeviceGetAttributeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->value = value;
			request->attr = attr;
			request->device = device;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaDeviceGetAttributeResponse*>(responsePayload);
			if (value) { *value = response->value; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetAttribute);
	return err;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int  device)
{
	TALLY_LOG("cudaDeviceGetDefaultMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetDefaultMemPool(memPool, device);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetDefaultMemPoolArg), alignof(cudaDeviceGetDefaultMemPoolArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetDefaultMemPool);
	return err;
}

cudaError_t cudaDeviceSetMemPool(int  device, cudaMemPool_t  memPool)
{
	TALLY_LOG("cudaDeviceSetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceSetMemPool(device, memPool);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceSetMemPoolArg), alignof(cudaDeviceSetMemPoolArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceSetMemPool);
	return err;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t * memPool, int  device)
{
	TALLY_LOG("cudaDeviceGetMemPool hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetMemPool(memPool, device);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetMemPoolArg), alignof(cudaDeviceGetMemPoolArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetMemPool);
	return err;
}

cudaError_t cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int  device, int  flags)
{
	TALLY_LOG("cudaDeviceGetNvSciSyncAttributes hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaDeviceGetP2PAttribute(int * value, enum cudaDeviceP2PAttr  attr, int  srcDevice, int  dstDevice)
{
	TALLY_LOG("cudaDeviceGetP2PAttribute hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaDeviceGetP2PAttributeArg), alignof(cudaDeviceGetP2PAttributeArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaDeviceGetP2PAttribute);
	return err;
}

cudaError_t cudaInitDevice(int  device, unsigned int  deviceFlags, unsigned int  flags)
{
	TALLY_LOG("cudaInitDevice hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetDevice(int * device)
{
	TALLY_LOG("cudaGetDevice hooked");
	cudaError_t res = 		lcudaGetDevice(device);
	return res;
}

cudaError_t cudaSetValidDevices(int * device_arr, int  len)
{
	TALLY_LOG("cudaSetValidDevices hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags)
{
	TALLY_LOG("cudaSetDeviceFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaSetDeviceFlags(flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaSetDeviceFlagsArg), alignof(cudaSetDeviceFlagsArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaSetDeviceFlags);
	return err;
}

cudaError_t cudaGetDeviceFlags(unsigned int * flags)
{
	TALLY_LOG("cudaGetDeviceFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGetDeviceFlags(flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGetDeviceFlagsArg), alignof(cudaGetDeviceFlagsArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGetDeviceFlags);
	return err;
}

cudaError_t cudaStreamCreate(cudaStream_t * pStream)
{
	TALLY_LOG("cudaStreamCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCreate(pStream);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCreateArg), alignof(cudaStreamCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pStream = pStream;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamCreateResponse*>(responsePayload);
			if (pStream) { *pStream = response->pStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCreate);
	return err;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int  flags)
{
	TALLY_LOG("cudaStreamCreateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCreateWithFlags(pStream, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCreateWithFlagsArg), alignof(cudaStreamCreateWithFlagsArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCREATEWITHFLAGS;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCreateWithFlagsArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pStream = pStream;
			request->flags = flags;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamCreateWithFlagsResponse*>(responsePayload);
			if (pStream) { *pStream = response->pStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCreateWithFlags);
	return err;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned int  flags, int  priority)
{
	TALLY_LOG("cudaStreamCreateWithPriority hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCreateWithPriority(pStream, flags, priority);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCreateWithPriorityArg), alignof(cudaStreamCreateWithPriorityArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMCREATEWITHPRIORITY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamCreateWithPriorityArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pStream = pStream;
			request->flags = flags;
			request->priority = priority;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cudaStreamCreateWithPriorityResponse*>(responsePayload);
			if (pStream) { *pStream = response->pStream; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCreateWithPriority);
	return err;
}

cudaError_t cudaStreamGetPriority(cudaStream_t  hStream, int * priority)
{
	TALLY_LOG("cudaStreamGetPriority hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamGetPriority(hStream, priority);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamGetPriorityArg), alignof(cudaStreamGetPriorityArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamGetPriority);
	return err;
}

cudaError_t cudaStreamGetFlags(cudaStream_t  hStream, unsigned int * flags)
{
	TALLY_LOG("cudaStreamGetFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamGetFlags(hStream, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamGetFlagsArg), alignof(cudaStreamGetFlagsArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamGetFlags);
	return err;
}

cudaError_t cudaStreamGetId(cudaStream_t  hStream, unsigned long long * streamId)
{
	TALLY_LOG("cudaStreamGetId hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaCtxResetPersistingL2Cache()
{
	TALLY_LOG("cudaCtxResetPersistingL2Cache hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaCtxResetPersistingL2Cache();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaCtxResetPersistingL2CacheArg), alignof(cudaCtxResetPersistingL2CacheArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaCtxResetPersistingL2Cache);
	return err;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t  dst, cudaStream_t  src)
{
	TALLY_LOG("cudaStreamCopyAttributes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamCopyAttributes(dst, src);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamCopyAttributesArg), alignof(cudaStreamCopyAttributesArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamCopyAttributes);
	return err;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t  hStream, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out)
{
	TALLY_LOG("cudaStreamGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamSetAttribute(cudaStream_t  hStream, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value)
{
	TALLY_LOG("cudaStreamSetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamDestroy(cudaStream_t  stream)
{
	TALLY_LOG("cudaStreamDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamDestroy(stream);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamDestroyArg), alignof(cudaStreamDestroyArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamDestroy);
	return err;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t  stream, cudaEvent_t  event, unsigned int  flags)
{
	TALLY_LOG("cudaStreamWaitEvent hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamWaitEvent(stream, event, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamWaitEventArg), alignof(cudaStreamWaitEventArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamWaitEvent);
	return err;
}

cudaError_t cudaStreamAddCallback(cudaStream_t  stream, cudaStreamCallback_t  callback, void * userData, unsigned int  flags)
{
	TALLY_LOG("cudaStreamAddCallback hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamQuery(cudaStream_t  stream)
{
	TALLY_LOG("cudaStreamQuery hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamQuery(stream);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamQueryArg), alignof(cudaStreamQueryArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamQuery);
	return err;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t  stream, void * devPtr, size_t  length, unsigned int  flags)
{
	TALLY_LOG("cudaStreamAttachMemAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamBeginCapture(cudaStream_t  stream, enum cudaStreamCaptureMode  mode)
{
	TALLY_LOG("cudaStreamBeginCapture hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamBeginCapture(stream, mode);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamBeginCaptureArg), alignof(cudaStreamBeginCaptureArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMBEGINCAPTURE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamBeginCaptureArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->mode = mode;

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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamBeginCapture);
	return err;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode * mode)
{
	TALLY_LOG("cudaThreadExchangeStreamCaptureMode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaStreamEndCapture(cudaStream_t  stream, cudaGraph_t * pGraph)
{
	TALLY_LOG("cudaStreamEndCapture hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamEndCapture(stream, pGraph);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamEndCaptureArg), alignof(cudaStreamEndCaptureArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDASTREAMENDCAPTURE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaStreamEndCaptureArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->stream = stream;
			request->pGraph = pGraph;

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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamEndCapture);
	return err;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t  stream, enum cudaStreamCaptureStatus * pCaptureStatus)
{
	TALLY_LOG("cudaStreamIsCapturing hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaStreamIsCapturing(stream, pCaptureStatus);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaStreamIsCapturingArg), alignof(cudaStreamIsCapturingArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaStreamIsCapturing);
	return err;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t  stream, cudaGraphNode_t * dependencies, size_t  numDependencies, unsigned int  flags)
{
	TALLY_LOG("cudaStreamUpdateCaptureDependencies hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaEventCreate(cudaEvent_t * event)
{
	TALLY_LOG("cudaEventCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventCreate(event);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventCreateArg), alignof(cudaEventCreateArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventCreate);
	return err;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int  flags)
{
	TALLY_LOG("cudaEventCreateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventCreateWithFlags(event, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventCreateWithFlagsArg), alignof(cudaEventCreateWithFlagsArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventCreateWithFlags);
	return err;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t  event, cudaStream_t  stream, unsigned int  flags)
{
	TALLY_LOG("cudaEventRecordWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventRecordWithFlags(event, stream, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventRecordWithFlagsArg), alignof(cudaEventRecordWithFlagsArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventRecordWithFlags);
	return err;
}

cudaError_t cudaEventQuery(cudaEvent_t  event)
{
	TALLY_LOG("cudaEventQuery hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventQuery(event);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventQueryArg), alignof(cudaEventQueryArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventQuery);
	return err;
}

cudaError_t cudaEventSynchronize(cudaEvent_t  event)
{
	TALLY_LOG("cudaEventSynchronize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventSynchronize(event);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventSynchronizeArg), alignof(cudaEventSynchronizeArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventSynchronize);
	return err;
}

cudaError_t cudaEventDestroy(cudaEvent_t  event)
{
	TALLY_LOG("cudaEventDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventDestroy(event);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventDestroyArg), alignof(cudaEventDestroyArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventDestroy);
	return err;
}

cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t  start, cudaEvent_t  end)
{
	TALLY_LOG("cudaEventElapsedTime hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaEventElapsedTime(ms, start, end);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaEventElapsedTimeArg), alignof(cudaEventElapsedTimeArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaEventElapsedTime);
	return err;
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

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args)
{
	TALLY_LOG("cudaLaunchKernelExC hooked");
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

cudaError_t cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig)
{
	TALLY_LOG("cudaOccupancyMaxPotentialClusterSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig)
{
	TALLY_LOG("cudaOccupancyMaxActiveClusters hooked");
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
	cudaError_t res = 		lcudaMallocHost(ptr, size);
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
	cudaError_t res = 		lcudaFreeHost(ptr);
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaMemGetInfo(free, total);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaMemGetInfoArg), alignof(cudaMemGetInfoArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaMemGetInfo);
	return err;
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

cudaError_t cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t  array, int  device)
{
	TALLY_LOG("cudaArrayGetMemoryRequirements hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t  mipmap, int  device)
{
	TALLY_LOG("cudaMipmappedArrayGetMemoryRequirements hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaMemset(devPtr, value, count);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaMemsetArg), alignof(cudaMemsetArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMSET;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemsetArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->devPtr = devPtr;
			request->value = value;
			request->count = count;

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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaMemset);
	return err;
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaMemsetAsync(devPtr, value, count, stream);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaMemsetAsyncArg), alignof(cudaMemsetAsyncArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUDAMEMSETASYNC;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cudaMemsetAsyncArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->devPtr = devPtr;
			request->value = value;
			request->count = count;
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaMemsetAsync);
	return err;
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

cudaError_t cudaMemPrefetchAsync_v2(const void * devPtr, size_t  count, struct cudaMemLocation  location, unsigned int  flags, cudaStream_t  stream)
{
	TALLY_LOG("cudaMemPrefetchAsync_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemAdvise(const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, int  device)
{
	TALLY_LOG("cudaMemAdvise hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaMemAdvise_v2(const void * devPtr, size_t  count, enum cudaMemoryAdvise  advice, struct cudaMemLocation  location)
{
	TALLY_LOG("cudaMemAdvise_v2 hooked");
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
	cudaError_t res = 		lcudaDriverGetVersion(driverVersion);
	return res;
}

cudaError_t cudaRuntimeGetVersion(int * runtimeVersion)
{
	TALLY_LOG("cudaRuntimeGetVersion hooked");
	cudaError_t res = 		lcudaRuntimeGetVersion(runtimeVersion);
	return res;
}

cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned int  flags)
{
	TALLY_LOG("cudaGraphCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphCreate(pGraph, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphCreateArg), alignof(cudaGraphCreateArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphCreate);
	return err;
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

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, cudaLaunchAttributeValue * value_out)
{
	TALLY_LOG("cudaGraphKernelNodeGetAttribute hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t  hNode, cudaLaunchAttributeID  attr, const cudaLaunchAttributeValue * value)
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

cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)
{
	TALLY_LOG("cudaGraphInstantiate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, unsigned long long  flags)
{
	TALLY_LOG("cudaGraphInstantiateWithFlags hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphInstantiateWithFlags(pGraphExec, graph, flags);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphInstantiateWithFlagsArg), alignof(cudaGraphInstantiateWithFlagsArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphInstantiateWithFlags);
	return err;
}

cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t * pGraphExec, cudaGraph_t  graph, cudaGraphInstantiateParams * instantiateParams)
{
	TALLY_LOG("cudaGraphInstantiateWithParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t  graphExec, unsigned long long * flags)
{
	TALLY_LOG("cudaGraphExecGetFlags hooked");
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

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int  isEnabled)
{
	TALLY_LOG("cudaGraphNodeSetEnabled hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t  hGraphExec, cudaGraphNode_t  hNode, unsigned int * isEnabled)
{
	TALLY_LOG("cudaGraphNodeGetEnabled hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t  hGraphExec, cudaGraph_t  hGraph, cudaGraphExecUpdateResultInfo * resultInfo)
{
	TALLY_LOG("cudaGraphExecUpdate hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphUpload(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	TALLY_LOG("cudaGraphUpload hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphUpload(graphExec, stream);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphUploadArg), alignof(cudaGraphUploadArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphUpload);
	return err;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t  graphExec, cudaStream_t  stream)
{
	TALLY_LOG("cudaGraphLaunch hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphLaunch(graphExec, stream);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphLaunchArg), alignof(cudaGraphLaunchArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphLaunch);
	return err;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t  graphExec)
{
	TALLY_LOG("cudaGraphExecDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphExecDestroy(graphExec);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphExecDestroyArg), alignof(cudaGraphExecDestroyArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphExecDestroy);
	return err;
}

cudaError_t cudaGraphDestroy(cudaGraph_t  graph)
{
	TALLY_LOG("cudaGraphDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaGraphDestroy(graph);
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaGraphDestroyArg), alignof(cudaGraphDestroyArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaGraphDestroy);
	return err;
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

cudaError_t cudaGraphAddNode(cudaGraphNode_t * pGraphNode, cudaGraph_t  graph, const cudaGraphNode_t * pDependencies, size_t  numDependencies, struct cudaGraphNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphAddNode hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t  graphExec, cudaGraphNode_t  node, struct cudaGraphNodeParams * nodeParams)
{
	TALLY_LOG("cudaGraphExecNodeSetParams hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned long long  flags, enum cudaDriverEntryPointQueryResult * driverStatus)
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

cudaError_t cudaGetKernel(cudaKernel_t * kernelPtr, const void * entryFuncAddr)
{
	TALLY_LOG("cudaGetKernel hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

size_t cudnnGetVersion()
{
	TALLY_LOG("cudnnGetVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetVersion();
#else

    size_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetVersionArg), alignof(cudnnGetVersionArg))
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
	TALLY_LOG("cudnnGetMaxDeviceVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetMaxDeviceVersion();
#else

    size_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetMaxDeviceVersionArg), alignof(cudnnGetMaxDeviceVersionArg))
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
	TALLY_LOG("cudnnGetCudartVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetCudartVersion();
#else

    size_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetCudartVersionArg), alignof(cudnnGetCudartVersionArg))
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

const char * cudnnGetErrorString(cudnnStatus_t  status)
{
	TALLY_LOG("cudnnGetErrorString hooked");
	const char * res = 		lcudnnGetErrorString(status);
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetProperty(type, value);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetPropertyArg), alignof(cudnnGetPropertyArg))
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
	TALLY_LOG("cudnnDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroy(handle);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyArg), alignof(cudnnDestroyArg))
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
	TALLY_LOG("cudnnSetStream hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetStream(handle, streamId);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetStreamArg), alignof(cudnnSetStreamArg))
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
	TALLY_LOG("cudnnGetStream hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetStream(handle, streamId);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetStreamArg), alignof(cudnnGetStreamArg))
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
	TALLY_LOG("cudnnCreateTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateTensorDescriptor(tensorDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateTensorDescriptorArg), alignof(cudnnCreateTensorDescriptorArg))
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
	TALLY_LOG("cudnnSetTensor4dDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetTensor4dDescriptorArg), alignof(cudnnSetTensor4dDescriptorArg))
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
	TALLY_LOG("cudnnSetTensor4dDescriptorEx hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetTensor4dDescriptorExArg), alignof(cudnnSetTensor4dDescriptorExArg))
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
	TALLY_LOG("cudnnGetTensor4dDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetTensor4dDescriptor(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetTensor4dDescriptorArg), alignof(cudnnGetTensor4dDescriptorArg))
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
	TALLY_LOG("cudnnSetTensorNdDescriptorEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t  tensorDesc, size_t * size)
{
	TALLY_LOG("cudnnGetTensorSizeInBytes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetTensorSizeInBytes(tensorDesc, size);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetTensorSizeInBytesArg), alignof(cudnnGetTensorSizeInBytesArg))
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
	TALLY_LOG("cudnnDestroyTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyTensorDescriptor(tensorDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyTensorDescriptorArg), alignof(cudnnDestroyTensorDescriptorArg))
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
	TALLY_LOG("cudnnInitTransformDest hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnInitTransformDest(transformDesc, srcDesc, destDesc, destSizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnInitTransformDestArg), alignof(cudnnInitTransformDestArg))
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
	TALLY_LOG("cudnnCreateTensorTransformDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateTensorTransformDescriptor(transformDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateTensorTransformDescriptorArg), alignof(cudnnCreateTensorTransformDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyTensorTransformDescriptor(transformDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyTensorTransformDescriptorArg), alignof(cudnnDestroyTensorTransformDescriptorArg))
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
	TALLY_LOG("cudnnTransformTensorEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t * opTensorDesc)
{
	TALLY_LOG("cudnnCreateOpTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateOpTensorDescriptor(opTensorDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateOpTensorDescriptorArg), alignof(cudnnCreateOpTensorDescriptorArg))
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
	TALLY_LOG("cudnnSetOpTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetOpTensorDescriptorArg), alignof(cudnnSetOpTensorDescriptorArg))
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
	TALLY_LOG("cudnnGetOpTensorDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetOpTensorDescriptorArg), alignof(cudnnGetOpTensorDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateFilterDescriptor(filterDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateFilterDescriptorArg), alignof(cudnnCreateFilterDescriptorArg))
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
	TALLY_LOG("cudnnSetFilter4dDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetFilter4dDescriptorArg), alignof(cudnnSetFilter4dDescriptorArg))
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
	TALLY_LOG("cudnnGetFilter4dDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t  filterDesc, size_t * size)
{
	TALLY_LOG("cudnnGetFilterSizeInBytes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetFilterSizeInBytes(filterDesc, size);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetFilterSizeInBytesArg), alignof(cudnnGetFilterSizeInBytesArg))
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
	TALLY_LOG("cudnnTransformFilter hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t  filterDesc)
{
	TALLY_LOG("cudnnDestroyFilterDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyFilterDescriptor(filterDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyFilterDescriptorArg), alignof(cudnnDestroyFilterDescriptorArg))
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
	TALLY_LOG("cudnnCreatePoolingDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreatePoolingDescriptor(poolingDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreatePoolingDescriptorArg), alignof(cudnnCreatePoolingDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyPoolingDescriptor(poolingDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyPoolingDescriptorArg), alignof(cudnnDestroyPoolingDescriptorArg))
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
	TALLY_LOG("cudnnCreateActivationDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateActivationDescriptor(activationDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateActivationDescriptorArg), alignof(cudnnCreateActivationDescriptorArg))
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
	TALLY_LOG("cudnnSetActivationDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetActivationDescriptorArg), alignof(cudnnSetActivationDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyActivationDescriptor(activationDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyActivationDescriptorArg), alignof(cudnnDestroyActivationDescriptorArg))
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
	TALLY_LOG("cudnnCreateLRNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateLRNDescriptor(normDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateLRNDescriptorArg), alignof(cudnnCreateLRNDescriptorArg))
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
	TALLY_LOG("cudnnSetLRNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetLRNDescriptorArg), alignof(cudnnSetLRNDescriptorArg))
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
	TALLY_LOG("cudnnGetLRNDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t  lrnDesc)
{
	TALLY_LOG("cudnnDestroyLRNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyLRNDescriptor(lrnDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyLRNDescriptorArg), alignof(cudnnDestroyLRNDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateDropoutDescriptor(dropoutDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateDropoutDescriptorArg), alignof(cudnnCreateDropoutDescriptorArg))
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
	TALLY_LOG("cudnnDestroyDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyDropoutDescriptor(dropoutDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyDropoutDescriptorArg), alignof(cudnnDestroyDropoutDescriptorArg))
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
	TALLY_LOG("cudnnDropoutGetStatesSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDropoutGetStatesSize(handle, sizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDropoutGetStatesSizeArg), alignof(cudnnDropoutGetStatesSizeArg))
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
	TALLY_LOG("cudnnDropoutGetReserveSpaceSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t  dropoutDesc, cudnnHandle_t  handle, float  dropout, void * states, size_t  stateSizeInBytes, unsigned long long  seed)
{
	TALLY_LOG("cudnnSetDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetDropoutDescriptorArg), alignof(cudnnSetDropoutDescriptorArg))
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
	TALLY_LOG("cudnnRestoreDropoutDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRestoreDropoutDescriptorArg), alignof(cudnnRestoreDropoutDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnOpsInferVersionCheck();
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnOpsInferVersionCheckArg), alignof(cudnnOpsInferVersionCheckArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeArg), alignof(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeArg))
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
	TALLY_LOG("cudnnGetBatchNormalizationBackwardExWorkspaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, sizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetBatchNormalizationBackwardExWorkspaceSizeArg), alignof(cudnnGetBatchNormalizationBackwardExWorkspaceSizeArg))
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
	TALLY_LOG("cudnnGetBatchNormalizationTrainingExReserveSpaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetBatchNormalizationTrainingExReserveSpaceSizeArg), alignof(cudnnGetBatchNormalizationTrainingExReserveSpaceSizeArg))
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
	TALLY_LOG("cudnnBatchNormalizationForwardTraining hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t  handle, cudnnBatchNormMode_t  mode, const void * alphaDataDiff, const void * betaDataDiff, const void * alphaParamDiff, const void * betaParamDiff, const cudnnTensorDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  dyDesc, const void * dy, const cudnnTensorDescriptor_t  dxDesc, void * dx, const cudnnTensorDescriptor_t  dBnScaleBiasDesc, const void * bnScale, void * dBnScaleResult, void * dBnBiasResult, double  epsilon, const void * savedMean, const void * savedInvVariance)
{
	TALLY_LOG("cudnnBatchNormalizationBackward hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnOpsTrainVersionCheck();
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnOpsTrainVersionCheckArg), alignof(cudnnOpsTrainVersionCheckArg))
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
	TALLY_LOG("cudnnCreateRNNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateRNNDescriptor(rnnDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateRNNDescriptorArg), alignof(cudnnCreateRNNDescriptorArg))
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
	TALLY_LOG("cudnnDestroyRNNDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyRNNDescriptor(rnnDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyRNNDescriptorArg), alignof(cudnnDestroyRNNDescriptorArg))
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
	TALLY_LOG("cudnnSetRNNDescriptor_v8 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNDescriptor_v8(rnnDesc, algo, cellMode, biasMode, dirMode, inputMode, dataType, mathPrec, mathType, inputSize, hiddenSize, projSize, numLayers, dropoutDesc, auxFlags);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNDescriptor_v8Arg), alignof(cudnnSetRNNDescriptor_v8Arg))
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
	TALLY_LOG("cudnnGetRNNDescriptor_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, const int  hiddenSize, const int  numLayers, cudnnDropoutDescriptor_t  dropoutDesc, cudnnRNNInputMode_t  inputMode, cudnnDirectionMode_t  direction, cudnnRNNMode_t  cellMode, cudnnRNNAlgo_t  algo, cudnnDataType_t  mathPrec)
{
	TALLY_LOG("cudnnSetRNNDescriptor_v6 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, cellMode, algo, mathPrec);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNDescriptor_v6Arg), alignof(cudnnSetRNNDescriptor_v6Arg))
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
	TALLY_LOG("cudnnGetRNNDescriptor_v6 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t  rnnDesc, cudnnMathType_t  mType)
{
	TALLY_LOG("cudnnSetRNNMatrixMathType hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNMatrixMathType(rnnDesc, mType);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNMatrixMathTypeArg), alignof(cudnnSetRNNMatrixMathTypeArg))
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
	TALLY_LOG("cudnnGetRNNMatrixMathType hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNMatrixMathType(rnnDesc, mType);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNMatrixMathTypeArg), alignof(cudnnGetRNNMatrixMathTypeArg))
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
	TALLY_LOG("cudnnSetRNNBiasMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNBiasMode(rnnDesc, biasMode);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNBiasModeArg), alignof(cudnnSetRNNBiasModeArg))
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
	TALLY_LOG("cudnnGetRNNBiasMode hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNBiasMode(rnnDesc, biasMode);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNBiasModeArg), alignof(cudnnGetRNNBiasModeArg))
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
	TALLY_LOG("cudnnRNNSetClip_v8 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNSetClip_v8(rnnDesc, clipMode, clipNanOpt, lclip, rclip);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNSetClip_v8Arg), alignof(cudnnRNNSetClip_v8Arg))
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
	TALLY_LOG("cudnnRNNGetClip_v8 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnRNNClipMode_t  clipMode, cudnnNanPropagation_t  clipNanOpt, double  lclip, double  rclip)
{
	TALLY_LOG("cudnnRNNSetClip hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnRNNSetClipArg), alignof(cudnnRNNSetClipArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBuildRNNDynamic(handle, rnnDesc, miniBatch);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBuildRNNDynamicArg), alignof(cudnnBuildRNNDynamicArg))
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
	TALLY_LOG("cudnnGetRNNTempSpaceSizes hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNTempSpaceSizes(handle, rnnDesc, fwdMode, xDesc, workSpaceSize, reserveSpaceSize);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNTempSpaceSizesArg), alignof(cudnnGetRNNTempSpaceSizesArg))
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
	TALLY_LOG("cudnnGetRNNParamsSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNParamsSize(handle, rnnDesc, xDesc, sizeInBytes, dataType);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNParamsSizeArg), alignof(cudnnGetRNNParamsSizeArg))
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
	TALLY_LOG("cudnnGetRNNWeightSpaceSize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNWeightSpaceSize(handle, rnnDesc, weightSpaceSize);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNWeightSpaceSizeArg), alignof(cudnnGetRNNWeightSpaceSizeArg))
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
	TALLY_LOG("cudnnGetRNNLinLayerMatrixParams hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNLinLayerMatrixParamsArg), alignof(cudnnGetRNNLinLayerMatrixParamsArg))
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
	TALLY_LOG("cudnnGetRNNLinLayerBiasParams hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNLinLayerBiasParamsArg), alignof(cudnnGetRNNLinLayerBiasParamsArg))
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
	TALLY_LOG("cudnnGetRNNWeightParams hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNWeightParams(handle, rnnDesc, pseudoLayer, weightSpaceSize, weightSpace, linLayerID, mDesc, mAddr, bDesc, bAddr);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNWeightParamsArg), alignof(cudnnGetRNNWeightParamsArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateRNNDataDescriptor(rnnDataDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateRNNDataDescriptorArg), alignof(cudnnCreateRNNDataDescriptorArg))
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
	TALLY_LOG("cudnnDestroyRNNDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyRNNDataDescriptor(rnnDataDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyRNNDataDescriptorArg), alignof(cudnnDestroyRNNDataDescriptorArg))
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
	TALLY_LOG("cudnnGetRNNDataDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnHandle_t  handle, const cudnnRNNDescriptor_t  rnnDesc, const cudnnRNNDataDescriptor_t  xDesc, const void * x, const cudnnTensorDescriptor_t  hxDesc, const void * hx, const cudnnTensorDescriptor_t  cxDesc, const void * cx, const cudnnFilterDescriptor_t  wDesc, const void * w, const cudnnRNNDataDescriptor_t  yDesc, void * y, const cudnnTensorDescriptor_t  hyDesc, void * hy, const cudnnTensorDescriptor_t  cyDesc, void * cy, const cudnnRNNDataDescriptor_t  kDesc, const void * keys, const cudnnRNNDataDescriptor_t  cDesc, void * cAttn, const cudnnRNNDataDescriptor_t  iDesc, void * iAttn, const cudnnRNNDataDescriptor_t  qDesc, void * queries, void * workSpace, size_t  workSpaceSizeInBytes)
{
	TALLY_LOG("cudnnRNNForwardInferenceEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t  handle, cudnnRNNDescriptor_t  rnnDesc, cudnnAlgorithmDescriptor_t  algoDesc)
{
	TALLY_LOG("cudnnSetRNNAlgorithmDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetRNNAlgorithmDescriptorArg), alignof(cudnnSetRNNAlgorithmDescriptorArg))
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
	TALLY_LOG("cudnnGetRNNForwardInferenceAlgorithmMaxCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, count);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetRNNForwardInferenceAlgorithmMaxCountArg), alignof(cudnnGetRNNForwardInferenceAlgorithmMaxCountArg))
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
	TALLY_LOG("cudnnFindRNNForwardInferenceAlgorithmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t * seqDataDesc)
{
	TALLY_LOG("cudnnCreateSeqDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateSeqDataDescriptor(seqDataDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateSeqDataDescriptorArg), alignof(cudnnCreateSeqDataDescriptorArg))
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
	TALLY_LOG("cudnnDestroySeqDataDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroySeqDataDescriptor(seqDataDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroySeqDataDescriptorArg), alignof(cudnnDestroySeqDataDescriptorArg))
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
	TALLY_LOG("cudnnCreateAttnDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateAttnDescriptor(attnDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateAttnDescriptorArg), alignof(cudnnCreateAttnDescriptorArg))
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
	TALLY_LOG("cudnnDestroyAttnDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyAttnDescriptor(attnDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyAttnDescriptorArg), alignof(cudnnDestroyAttnDescriptorArg))
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
	TALLY_LOG("cudnnSetAttnDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnSetAttnDescriptorArg), alignof(cudnnSetAttnDescriptorArg))
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
	TALLY_LOG("cudnnGetAttnDescriptor hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t  handle, const cudnnAttnDescriptor_t  attnDesc, size_t * weightSizeInBytes, size_t * workSpaceSizeInBytes, size_t * reserveSpaceSizeInBytes)
{
	TALLY_LOG("cudnnGetMultiHeadAttnBuffers hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetMultiHeadAttnBuffers(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetMultiHeadAttnBuffersArg), alignof(cudnnGetMultiHeadAttnBuffersArg))
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
	TALLY_LOG("cudnnGetMultiHeadAttnWeights hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnAdvInferVersionCheck()
{
	TALLY_LOG("cudnnAdvInferVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnAdvInferVersionCheck();
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnAdvInferVersionCheckArg), alignof(cudnnAdvInferVersionCheckArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnAdvTrainVersionCheck();
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnAdvTrainVersionCheckArg), alignof(cudnnAdvTrainVersionCheckArg))
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
	TALLY_LOG("cudnnCreateConvolutionDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCreateConvolutionDescriptor(convDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCreateConvolutionDescriptorArg), alignof(cudnnCreateConvolutionDescriptorArg))
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
	TALLY_LOG("cudnnDestroyConvolutionDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnDestroyConvolutionDescriptor(convDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnDestroyConvolutionDescriptorArg), alignof(cudnnDestroyConvolutionDescriptorArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionForwardWorkspaceSizeArg), alignof(cudnnGetConvolutionForwardWorkspaceSizeArg))
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
	TALLY_LOG("cudnnConvolutionBiasActivationForward hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t  handle, int * count)
{
	TALLY_LOG("cudnnGetConvolutionBackwardDataAlgorithmMaxCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, count);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionBackwardDataAlgorithmMaxCountArg), alignof(cudnnGetConvolutionBackwardDataAlgorithmMaxCountArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionBackwardDataWorkspaceSizeArg), alignof(cudnnGetConvolutionBackwardDataWorkspaceSizeArg))
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
	TALLY_LOG("cudnnConvolutionBackwardData hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t  handle, const cudnnFilterDescriptor_t  filterDesc, const cudnnTensorDescriptor_t  diffDesc, const cudnnConvolutionDescriptor_t  convDesc, const cudnnTensorDescriptor_t  gradDesc, const cudnnTensorFormat_t  transformFormat, cudnnFilterDescriptor_t  foldedFilterDesc, cudnnTensorDescriptor_t  paddedDiffDesc, cudnnConvolutionDescriptor_t  foldedConvDesc, cudnnTensorDescriptor_t  foldedGradDesc, cudnnTensorTransformDescriptor_t  filterFoldTransDesc, cudnnTensorTransformDescriptor_t  diffPadTransDesc, cudnnTensorTransformDescriptor_t  gradFoldTransDesc, cudnnTensorTransformDescriptor_t  gradUnfoldTransDesc)
{
	TALLY_LOG("cudnnGetFoldedConvBackwardDataDescriptors hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetFoldedConvBackwardDataDescriptorsArg), alignof(cudnnGetFoldedConvBackwardDataDescriptorsArg))
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
	TALLY_LOG("cudnnCnnInferVersionCheck hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCnnInferVersionCheck();
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCnnInferVersionCheckArg), alignof(cudnnCnnInferVersionCheckArg))
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
	TALLY_LOG("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, count);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnGetConvolutionBackwardFilterAlgorithmMaxCountArg), alignof(cudnnGetConvolutionBackwardFilterAlgorithmMaxCountArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnCnnTrainVersionCheck();
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnCnnTrainVersionCheckArg), alignof(cudnnCnnTrainVersionCheckArg))
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
	TALLY_LOG("cudnnBackendCreateDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendCreateDescriptor(descriptorType, descriptor);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendCreateDescriptorArg), alignof(cudnnBackendCreateDescriptorArg))
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
	TALLY_LOG("cudnnBackendDestroyDescriptor hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendDestroyDescriptor(descriptor);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendDestroyDescriptorArg), alignof(cudnnBackendDestroyDescriptorArg))
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
	TALLY_LOG("cudnnBackendInitialize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendInitialize(descriptor);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendInitializeArg), alignof(cudnnBackendInitializeArg))
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
	TALLY_LOG("cudnnBackendFinalize hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudnnBackendFinalize(descriptor);
#else

    cudnnStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudnnBackendFinalizeArg), alignof(cudnnBackendFinalizeArg))
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

cublasStatus_t cublasDestroy_v2(cublasHandle_t  handle)
{
	TALLY_LOG("cublasDestroy_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasDestroy_v2(handle);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasDestroy_v2Arg), alignof(cublasDestroy_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASDESTROY_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasDestroy_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasDestroy_v2);
	return err;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t  handle, int*  version)
{
	TALLY_LOG("cublasGetVersion_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetVersion_v2(handle, version);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetVersion_v2Arg), alignof(cublasGetVersion_v2Arg))
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
	TALLY_LOG("cublasGetProperty hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetProperty(type, value);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetPropertyArg), alignof(cublasGetPropertyArg))
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
	TALLY_LOG("cublasGetCudartVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetCudartVersion();
#else

    size_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetCudartVersionArg), alignof(cublasGetCudartVersionArg))
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

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t  handle, void*  workspace, size_t  workspaceSizeInBytes)
{
	TALLY_LOG("cublasSetWorkspace_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetWorkspace_v2(handle, workspace, workspaceSizeInBytes);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetWorkspace_v2Arg), alignof(cublasSetWorkspace_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETWORKSPACE_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetWorkspace_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->workspace = workspace;
			request->workspaceSizeInBytes = workspaceSizeInBytes;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasSetWorkspace_v2);
	return err;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t  handle, cudaStream_t  streamId)
{
	TALLY_LOG("cublasSetStream_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetStream_v2(handle, streamId);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetStream_v2Arg), alignof(cublasSetStream_v2Arg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETSTREAM_V2;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetStream_v2Arg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->handle = handle;
			request->streamId = streamId;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasSetStream_v2);
	return err;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t  handle, cudaStream_t*  streamId)
{
	TALLY_LOG("cublasGetStream_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetStream_v2(handle, streamId);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetStream_v2Arg), alignof(cublasGetStream_v2Arg))
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
	TALLY_LOG("cublasGetPointerMode_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetPointerMode_v2(handle, mode);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetPointerMode_v2Arg), alignof(cublasGetPointerMode_v2Arg))
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
	TALLY_LOG("cublasSetPointerMode_v2 hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetPointerMode_v2(handle, mode);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetPointerMode_v2Arg), alignof(cublasSetPointerMode_v2Arg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetMathMode(handle, mode);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetMathModeArg), alignof(cublasSetMathModeArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASSETMATHMODE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasSetMathModeArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
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
	TALLY_CLIENT_TRACE_API_CALL(cublasSetMathMode);
	return err;
}

cublasStatus_t cublasGetSmCountTarget(cublasHandle_t  handle, int*  smCountTarget)
{
	TALLY_LOG("cublasGetSmCountTarget hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetSmCountTarget(handle, smCountTarget);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetSmCountTargetArg), alignof(cublasGetSmCountTargetArg))
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
	TALLY_LOG("cublasSetSmCountTarget hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetSmCountTarget(handle, smCountTarget);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetSmCountTargetArg), alignof(cublasSetSmCountTargetArg))
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetLoggerCallback(userCallback);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetLoggerCallbackArg), alignof(cublasSetLoggerCallbackArg))
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
	TALLY_LOG("cublasGetLoggerCallback hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasGetLoggerCallback(userCallback);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasGetLoggerCallbackArg), alignof(cublasGetLoggerCallbackArg))
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
	TALLY_LOG("cublasSetVector hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasSetVector(n, elemSize, x, incx, devicePtr, incy);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasSetVectorArg), alignof(cublasSetVectorArg))
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
	TALLY_LOG("cublasSetVector_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetVector(int  n, int  elemSize, const void*  x, int  incx, void*  y, int  incy)
{
	TALLY_LOG("cublasGetVector hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetVector_64(int64_t  n, int64_t  elemSize, const void*  x, int64_t  incx, void*  y, int64_t  incy)
{
	TALLY_LOG("cublasGetVector_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	TALLY_LOG("cublasSetMatrix hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMatrix_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb)
{
	TALLY_LOG("cublasSetMatrix_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMatrix(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb)
{
	TALLY_LOG("cublasGetMatrix hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMatrix_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb)
{
	TALLY_LOG("cublasGetMatrix_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetVectorAsync(int  n, int  elemSize, const void*  hostPtr, int  incx, void*  devicePtr, int  incy, cudaStream_t  stream)
{
	TALLY_LOG("cublasSetVectorAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetVectorAsync_64(int64_t  n, int64_t  elemSize, const void*  hostPtr, int64_t  incx, void*  devicePtr, int64_t  incy, cudaStream_t  stream)
{
	TALLY_LOG("cublasSetVectorAsync_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetVectorAsync(int  n, int  elemSize, const void*  devicePtr, int  incx, void*  hostPtr, int  incy, cudaStream_t  stream)
{
	TALLY_LOG("cublasGetVectorAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetVectorAsync_64(int64_t  n, int64_t  elemSize, const void*  devicePtr, int64_t  incx, void*  hostPtr, int64_t  incy, cudaStream_t  stream)
{
	TALLY_LOG("cublasGetVectorAsync_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	TALLY_LOG("cublasSetMatrixAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSetMatrixAsync_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream)
{
	TALLY_LOG("cublasSetMatrixAsync_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMatrixAsync(int  rows, int  cols, int  elemSize, const void*  A, int  lda, void*  B, int  ldb, cudaStream_t  stream)
{
	TALLY_LOG("cublasGetMatrixAsync hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGetMatrixAsync_64(int64_t  rows, int64_t  cols, int64_t  elemSize, const void*  A, int64_t  lda, void*  B, int64_t  ldb, cudaStream_t  stream)
{
	TALLY_LOG("cublasGetMatrixAsync_64 hooked");
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

cublasStatus_t cublasNrm2Ex_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasNrm2Ex_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasSnrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result)
{
	TALLY_LOG("cublasSnrm2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDnrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result)
{
	TALLY_LOG("cublasDnrm2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScnrm2_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasScnrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScnrm2_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result)
{
	TALLY_LOG("cublasScnrm2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDznrm2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDznrm2_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result)
{
	TALLY_LOG("cublasDznrm2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDotEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasDotEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDotEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasDotEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDotcEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, const void*  y, cudaDataType  yType, int  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasDotcEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDotcEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, const void*  y, cudaDataType  yType, int64_t  incy, void*  result, cudaDataType  resultType, cudaDataType  executionType)
{
	TALLY_LOG("cublasDotcEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSdot_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, const float*  y, int  incy, float*  result)
{
	TALLY_LOG("cublasSdot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSdot_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  result)
{
	TALLY_LOG("cublasSdot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDdot_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, const double*  y, int  incy, double*  result)
{
	TALLY_LOG("cublasDdot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDdot_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  result)
{
	TALLY_LOG("cublasDdot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	TALLY_LOG("cublasCdotu_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdotu_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result)
{
	TALLY_LOG("cublasCdotu_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  result)
{
	TALLY_LOG("cublasCdotc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdotc_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  result)
{
	TALLY_LOG("cublasCdotc_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	TALLY_LOG("cublasZdotu_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdotu_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result)
{
	TALLY_LOG("cublasZdotu_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  result)
{
	TALLY_LOG("cublasZdotc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdotc_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  result)
{
	TALLY_LOG("cublasZdotc_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScalEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int  incx, cudaDataType  executionType)
{
	TALLY_LOG("cublasScalEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScalEx_64(cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, void*  x, cudaDataType  xType, int64_t  incx, cudaDataType  executionType)
{
	TALLY_LOG("cublasScalEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, float*  x, int  incx)
{
	TALLY_LOG("cublasSscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSscal_v2_64(cublasHandle_t  handle, int64_t  n, const float*  alpha, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasSscal_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, double*  x, int  incx)
{
	TALLY_LOG("cublasDscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDscal_v2_64(cublasHandle_t  handle, int64_t  n, const double*  alpha, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDscal_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCscal_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCscal_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCscal_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsscal_v2(cublasHandle_t  handle, int  n, const float*  alpha, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCsscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsscal_v2_64(cublasHandle_t  handle, int64_t  n, const float*  alpha, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCsscal_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZscal_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZscal_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZscal_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdscal_v2(cublasHandle_t  handle, int  n, const double*  alpha, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZdscal_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdscal_v2_64(cublasHandle_t  handle, int64_t  n, const double*  alpha, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZdscal_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasAxpyEx(cublasHandle_t  handle, int  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, cudaDataType  executiontype)
{
	TALLY_LOG("cublasAxpyEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasAxpyEx_64(cublasHandle_t  handle, int64_t  n, const void*  alpha, cudaDataType  alphaType, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, cudaDataType  executiontype)
{
	TALLY_LOG("cublasAxpyEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t  handle, int  n, const float*  alpha, const float*  x, int  incx, float*  y, int  incy)
{
	TALLY_LOG("cublasSaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSaxpy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t  handle, int  n, const double*  alpha, const double*  x, int  incx, double*  y, int  incy)
{
	TALLY_LOG("cublasDaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDaxpy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCaxpy_v2(cublasHandle_t  handle, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasCaxpy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZaxpy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZaxpy_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZaxpy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCopyEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	TALLY_LOG("cublasCopyEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCopyEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy)
{
	TALLY_LOG("cublasCopyEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScopy_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  y, int  incy)
{
	TALLY_LOG("cublasScopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScopy_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasScopy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  y, int  incy)
{
	TALLY_LOG("cublasDcopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDcopy_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDcopy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCcopy_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCcopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCcopy_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasCcopy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZcopy_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZcopy_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZcopy_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSswap_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy)
{
	TALLY_LOG("cublasSswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSswap_v2_64(cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSswap_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDswap_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy)
{
	TALLY_LOG("cublasDswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDswap_v2_64(cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDswap_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCswap_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCswap_v2_64(cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasCswap_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZswap_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZswap_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZswap_v2_64(cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZswap_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSwapEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy)
{
	TALLY_LOG("cublasSwapEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSwapEx_64(cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy)
{
	TALLY_LOG("cublasSwapEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIsamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIsamax_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIsamax_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIdamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIdamax_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIdamax_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIcamax_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIcamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIcamax_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIcamax_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIzamax_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIzamax_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIzamax_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIzamax_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIamaxEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	TALLY_LOG("cublasIamaxEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIamaxEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIamaxEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIsamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIsamin_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIsamin_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIdamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIdamin_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIdamin_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIcamin_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIcamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIcamin_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIcamin_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIzamin_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, int*  result)
{
	TALLY_LOG("cublasIzamin_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIzamin_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIzamin_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIaminEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, int*  result)
{
	TALLY_LOG("cublasIaminEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasIaminEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, int64_t*  result)
{
	TALLY_LOG("cublasIaminEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasAsumEx(cublasHandle_t  handle, int  n, const void*  x, cudaDataType  xType, int  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasAsumEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasAsumEx_64(cublasHandle_t  handle, int64_t  n, const void*  x, cudaDataType  xType, int64_t  incx, void*  result, cudaDataType  resultType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasAsumEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSasum_v2(cublasHandle_t  handle, int  n, const float*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasSasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSasum_v2_64(cublasHandle_t  handle, int64_t  n, const float*  x, int64_t  incx, float*  result)
{
	TALLY_LOG("cublasSasum_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDasum_v2(cublasHandle_t  handle, int  n, const double*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDasum_v2_64(cublasHandle_t  handle, int64_t  n, const double*  x, int64_t  incx, double*  result)
{
	TALLY_LOG("cublasDasum_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScasum_v2(cublasHandle_t  handle, int  n, const cuComplex*  x, int  incx, float*  result)
{
	TALLY_LOG("cublasScasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasScasum_v2_64(cublasHandle_t  handle, int64_t  n, const cuComplex*  x, int64_t  incx, float*  result)
{
	TALLY_LOG("cublasScasum_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDzasum_v2(cublasHandle_t  handle, int  n, const cuDoubleComplex*  x, int  incx, double*  result)
{
	TALLY_LOG("cublasDzasum_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDzasum_v2_64(cublasHandle_t  handle, int64_t  n, const cuDoubleComplex*  x, int64_t  incx, double*  result)
{
	TALLY_LOG("cublasDzasum_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSrot_v2(cublasHandle_t  handle, int  n, float*  x, int  incx, float*  y, int  incy, const float*  c, const float*  s)
{
	TALLY_LOG("cublasSrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSrot_v2_64(cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  c, const float*  s)
{
	TALLY_LOG("cublasSrot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrot_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  c, const double*  s)
{
	TALLY_LOG("cublasDrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrot_v2_64(cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  c, const double*  s)
{
	TALLY_LOG("cublasDrot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const cuComplex*  s)
{
	TALLY_LOG("cublasCrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCrot_v2_64(cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const cuComplex*  s)
{
	TALLY_LOG("cublasCrot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsrot_v2(cublasHandle_t  handle, int  n, cuComplex*  x, int  incx, cuComplex*  y, int  incy, const float*  c, const float*  s)
{
	TALLY_LOG("cublasCsrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsrot_v2_64(cublasHandle_t  handle, int64_t  n, cuComplex*  x, int64_t  incx, cuComplex*  y, int64_t  incy, const float*  c, const float*  s)
{
	TALLY_LOG("cublasCsrot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const cuDoubleComplex*  s)
{
	TALLY_LOG("cublasZrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZrot_v2_64(cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const cuDoubleComplex*  s)
{
	TALLY_LOG("cublasZrot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdrot_v2(cublasHandle_t  handle, int  n, cuDoubleComplex*  x, int  incx, cuDoubleComplex*  y, int  incy, const double*  c, const double*  s)
{
	TALLY_LOG("cublasZdrot_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdrot_v2_64(cublasHandle_t  handle, int64_t  n, cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  y, int64_t  incy, const double*  c, const double*  s)
{
	TALLY_LOG("cublasZdrot_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotEx_64(cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  c, const void*  s, cudaDataType  csType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotEx_64 hooked");
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

cublasStatus_t cublasSrotm_v2_64(cublasHandle_t  handle, int64_t  n, float*  x, int64_t  incx, float*  y, int64_t  incy, const float*  param)
{
	TALLY_LOG("cublasSrotm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrotm_v2(cublasHandle_t  handle, int  n, double*  x, int  incx, double*  y, int  incy, const double*  param)
{
	TALLY_LOG("cublasDrotm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDrotm_v2_64(cublasHandle_t  handle, int64_t  n, double*  x, int64_t  incx, double*  y, int64_t  incy, const double*  param)
{
	TALLY_LOG("cublasDrotm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotmEx(cublasHandle_t  handle, int  n, void*  x, cudaDataType  xType, int  incx, void*  y, cudaDataType  yType, int  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasRotmEx_64(cublasHandle_t  handle, int64_t  n, void*  x, cudaDataType  xType, int64_t  incx, void*  y, cudaDataType  yType, int64_t  incy, const void*  param, cudaDataType  paramType, cudaDataType  executiontype)
{
	TALLY_LOG("cublasRotmEx_64 hooked");
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

cublasStatus_t cublasSgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSgemv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDgemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDgemv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCgemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasCgemv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZgemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZgemv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSgbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDgbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasCgbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgbmv_v2(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, int  kl, int  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZgbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgbmv_v2_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, int64_t  kl, int64_t  ku, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZgbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasStrmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDtrmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCtrmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtrmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZtrmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasStbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDtbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCtbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZtbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	TALLY_LOG("cublasStpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasStpmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	TALLY_LOG("cublasDtpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDtpmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCtpmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZtpmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasStrsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDtrsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCtrsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtrsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZtrsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const float*  AP, float*  x, int  incx)
{
	TALLY_LOG("cublasStpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const float*  AP, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasStpsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const double*  AP, double*  x, int  incx)
{
	TALLY_LOG("cublasDtpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const double*  AP, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDtpsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuComplex*  AP, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuComplex*  AP, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCtpsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtpsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtpsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, const cuDoubleComplex*  AP, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZtpsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const float*  A, int  lda, float*  x, int  incx)
{
	TALLY_LOG("cublasStbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const float*  A, int64_t  lda, float*  x, int64_t  incx)
{
	TALLY_LOG("cublasStbsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const double*  A, int  lda, double*  x, int  incx)
{
	TALLY_LOG("cublasDtbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const double*  A, int64_t  lda, double*  x, int64_t  incx)
{
	TALLY_LOG("cublasDtbsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuComplex*  A, int  lda, cuComplex*  x, int  incx)
{
	TALLY_LOG("cublasCtbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuComplex*  A, int64_t  lda, cuComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasCtbsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtbsv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  n, int  k, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  x, int  incx)
{
	TALLY_LOG("cublasZtbsv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtbsv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  n, int64_t  k, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  x, int64_t  incx)
{
	TALLY_LOG("cublasZtbsv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSsymv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDsymv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasCsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasCsymv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsymv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZsymv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsymv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZsymv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasChemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChemv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasChemv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhemv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZhemv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhemv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZhemv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSsbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSsbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDsbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDsbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasChbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasChbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhbmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZhbmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhbmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZhbmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  AP, const float*  x, int  incx, const float*  beta, float*  y, int  incy)
{
	TALLY_LOG("cublasSspmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  AP, const float*  x, int64_t  incx, const float*  beta, float*  y, int64_t  incy)
{
	TALLY_LOG("cublasSspmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  AP, const double*  x, int  incx, const double*  beta, double*  y, int  incy)
{
	TALLY_LOG("cublasDspmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  AP, const double*  x, int64_t  incx, const double*  beta, double*  y, int64_t  incy)
{
	TALLY_LOG("cublasDspmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int  incx, const cuComplex*  beta, cuComplex*  y, int  incy)
{
	TALLY_LOG("cublasChpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  AP, const cuComplex*  x, int64_t  incx, const cuComplex*  beta, cuComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasChpmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpmv_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy)
{
	TALLY_LOG("cublasZhpmv_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpmv_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  AP, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy)
{
	TALLY_LOG("cublasZhpmv_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSger_v2(cublasHandle_t  handle, int  m, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	TALLY_LOG("cublasSger_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSger_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda)
{
	TALLY_LOG("cublasSger_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDger_v2(cublasHandle_t  handle, int  m, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	TALLY_LOG("cublasDger_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDger_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda)
{
	TALLY_LOG("cublasDger_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCgeru_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeru_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasCgeru_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCgerc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgerc_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasCgerc_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeru_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZgeru_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeru_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasZgeru_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgerc_v2(cublasHandle_t  handle, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZgerc_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgerc_v2_64(cublasHandle_t  handle, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasZgerc_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  A, int  lda)
{
	TALLY_LOG("cublasSsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  A, int64_t  lda)
{
	TALLY_LOG("cublasSsyr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  A, int  lda)
{
	TALLY_LOG("cublasDsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  A, int64_t  lda)
{
	TALLY_LOG("cublasDsyr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasCsyr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZsyr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasZsyr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCher_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasCher_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZher_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasZher_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, float*  AP)
{
	TALLY_LOG("cublasSspr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, float*  AP)
{
	TALLY_LOG("cublasSspr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, double*  AP)
{
	TALLY_LOG("cublasDspr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, double*  AP)
{
	TALLY_LOG("cublasDspr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const cuComplex*  x, int  incx, cuComplex*  AP)
{
	TALLY_LOG("cublasChpr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const cuComplex*  x, int64_t  incx, cuComplex*  AP)
{
	TALLY_LOG("cublasChpr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpr_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZhpr_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpr_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZhpr_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  A, int  lda)
{
	TALLY_LOG("cublasSsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  A, int64_t  lda)
{
	TALLY_LOG("cublasSsyr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  A, int  lda)
{
	TALLY_LOG("cublasDsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  A, int64_t  lda)
{
	TALLY_LOG("cublasDsyr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasCsyr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZsyr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasZsyr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  A, int  lda)
{
	TALLY_LOG("cublasCher2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasCher2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  A, int  lda)
{
	TALLY_LOG("cublasZher2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  A, int64_t  lda)
{
	TALLY_LOG("cublasZher2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const float*  alpha, const float*  x, int  incx, const float*  y, int  incy, float*  AP)
{
	TALLY_LOG("cublasSspr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSspr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const float*  alpha, const float*  x, int64_t  incx, const float*  y, int64_t  incy, float*  AP)
{
	TALLY_LOG("cublasSspr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const double*  alpha, const double*  x, int  incx, const double*  y, int  incy, double*  AP)
{
	TALLY_LOG("cublasDspr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDspr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const double*  alpha, const double*  x, int64_t  incx, const double*  y, int64_t  incy, double*  AP)
{
	TALLY_LOG("cublasDspr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuComplex*  alpha, const cuComplex*  x, int  incx, const cuComplex*  y, int  incy, cuComplex*  AP)
{
	TALLY_LOG("cublasChpr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChpr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuComplex*  alpha, const cuComplex*  x, int64_t  incx, const cuComplex*  y, int64_t  incy, cuComplex*  AP)
{
	TALLY_LOG("cublasChpr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpr2_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int  incx, const cuDoubleComplex*  y, int  incy, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZhpr2_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhpr2_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  x, int64_t  incx, const cuDoubleComplex*  y, int64_t  incy, cuDoubleComplex*  AP)
{
	TALLY_LOG("cublasZhpr2_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasSgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasSgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  xarray[], int  incx, const double*  beta, double* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasDgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  xarray[], int64_t  incx, const double*  beta, double* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasDgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  xarray[], int  incx, const cuComplex*  beta, cuComplex* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasCgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  xarray[], int64_t  incx, const cuComplex*  beta, cuComplex* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasCgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  xarray[], int  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasZgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  xarray[], int64_t  incx, const cuDoubleComplex*  beta, cuDoubleComplex* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasZgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSHgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, __half* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasHSHgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSHgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, __half* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasHSHgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSSgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half* const  Aarray[], int  lda, const __half* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasHSSgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSSgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasHSSgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSTgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, __nv_bfloat16* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasTSTgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSTgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, __nv_bfloat16* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasTSTgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSSgemvBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int  lda, const __nv_bfloat16* const  xarray[], int  incx, const float*  beta, float* const  yarray[], int  incy, int  batchCount)
{
	TALLY_LOG("cublasTSSgemvBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSSgemvBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16* const  Aarray[], int64_t  lda, const __nv_bfloat16* const  xarray[], int64_t  incx, const float*  beta, float* const  yarray[], int64_t  incy, int64_t  batchCount)
{
	TALLY_LOG("cublasTSSgemvBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const float*  A, int  lda, long long int  strideA, const float*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasSgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasSgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  x, int  incx, long long int  stridex, const double*  beta, double*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasDgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  x, int64_t  incx, long long int  stridex, const double*  beta, double*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasDgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  x, int  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasCgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  x, int64_t  incx, long long int  stridex, const cuComplex*  beta, cuComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasCgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  x, int  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasZgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  x, int64_t  incx, long long int  stridex, const cuDoubleComplex*  beta, cuDoubleComplex*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasZgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSHgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, __half*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasHSHgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSHgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, __half*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasHSHgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSSgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasHSSgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHSSgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasHSSgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSTgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasTSTgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSTgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, __nv_bfloat16*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasTSTgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSSgemvStridedBatched(cublasHandle_t  handle, cublasOperation_t  trans, int  m, int  n, const float*  alpha, const __nv_bfloat16*  A, int  lda, long long int  strideA, const __nv_bfloat16*  x, int  incx, long long int  stridex, const float*  beta, float*  y, int  incy, long long int  stridey, int  batchCount)
{
	TALLY_LOG("cublasTSSgemvStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasTSSgemvStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  trans, int64_t  m, int64_t  n, const float*  alpha, const __nv_bfloat16*  A, int64_t  lda, long long int  strideA, const __nv_bfloat16*  x, int64_t  incx, long long int  stridex, const float*  beta, float*  y, int64_t  incy, long long int  stridey, int64_t  batchCount)
{
	TALLY_LOG("cublasTSSgemvStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSgemm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDgemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDgemm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCgemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCgemm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCgemm3m hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3m_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCgemm3m_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCgemm3mEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasCgemm3mEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZgemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemm_v2_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZgemm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemm3m(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZgemm3m hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemm3m_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZgemm3m_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemm(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, const __half*  B, int  ldb, const __half*  beta, __half*  C, int  ldc)
{
	TALLY_LOG("cublasHgemm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemm_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, const __half*  B, int64_t  ldb, const __half*  beta, __half*  C, int64_t  ldc)
{
	TALLY_LOG("cublasHgemm_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemmEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasSgemmEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const void*  B, cudaDataType  Btype, int  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCgemmEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const void*  B, cudaDataType  Btype, int64_t  ldb, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasCgemmEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSsyrk_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDsyrk_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCsyrk_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyrk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsyrk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyrk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZsyrk_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCsyrkEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrkEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasCsyrkEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCsyrk3mEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrk3mEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const cuComplex*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasCsyrk3mEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const cuComplex*  A, int  lda, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCherk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const cuComplex*  A, int64_t  lda, const float*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCherk_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZherk_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const cuDoubleComplex*  A, int  lda, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZherk_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZherk_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const cuDoubleComplex*  A, int64_t  lda, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZherk_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherkEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCherkEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherkEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasCherkEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherk3mEx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const void*  A, cudaDataType  Atype, int  lda, const float*  beta, void*  C, cudaDataType  Ctype, int  ldc)
{
	TALLY_LOG("cublasCherk3mEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherk3mEx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, const float*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc)
{
	TALLY_LOG("cublasCherk3mEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSsyr2k_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDsyr2k_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCsyr2k_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsyr2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyr2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZsyr2k_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCher2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCher2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCher2k_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher2k_v2(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZher2k_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZher2k_v2_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZher2k_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSsyrkx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDsyrkx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCsyrkx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyrkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsyrkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsyrkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZsyrkx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const float*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCherkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCherkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const float*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCherkx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZherkx(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const double*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZherkx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZherkx_64(cublasHandle_t  handle, cublasFillMode_t  uplo, cublasOperation_t  trans, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const double*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZherkx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, const float*  beta, float*  C, int  ldc)
{
	TALLY_LOG("cublasSsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, const float*  beta, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSsymm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, const double*  beta, double*  C, int  ldc)
{
	TALLY_LOG("cublasDsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, const double*  beta, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDsymm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCsymm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsymm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZsymm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZsymm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZsymm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, const cuComplex*  beta, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasChemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasChemm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, const cuComplex*  beta, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasChemm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhemm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZhemm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZhemm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZhemm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, float*  B, int  ldb)
{
	TALLY_LOG("cublasStrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, float*  B, int64_t  ldb)
{
	TALLY_LOG("cublasStrsm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, double*  B, int  ldb)
{
	TALLY_LOG("cublasDtrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, double*  B, int64_t  ldb)
{
	TALLY_LOG("cublasDtrsm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, cuComplex*  B, int  ldb)
{
	TALLY_LOG("cublasCtrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, cuComplex*  B, int64_t  ldb)
{
	TALLY_LOG("cublasCtrsm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, cuDoubleComplex*  B, int  ldb)
{
	TALLY_LOG("cublasZtrsm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, cuDoubleComplex*  B, int64_t  ldb)
{
	TALLY_LOG("cublasZtrsm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  B, int  ldb, float*  C, int  ldc)
{
	TALLY_LOG("cublasStrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  B, int64_t  ldb, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasStrmm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  B, int  ldb, double*  C, int  ldc)
{
	TALLY_LOG("cublasDtrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  B, int64_t  ldb, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDtrmm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCtrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCtrmm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrmm_v2(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZtrmm_v2 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrmm_v2_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZtrmm_v2_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half* const  Aarray[], int  lda, const __half* const  Barray[], int  ldb, const __half*  beta, __half* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasHgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half* const  Aarray[], int64_t  lda, const __half* const  Barray[], int64_t  ldb, const __half*  beta, __half* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_LOG("cublasHgemmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const float*  alpha, const float* const  Aarray[], int  lda, const float* const  Barray[], int  ldb, const float*  beta, float* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasSgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float* const  Aarray[], int64_t  lda, const float* const  Barray[], int64_t  ldb, const float*  beta, float* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_LOG("cublasSgemmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double* const  Aarray[], int  lda, const double* const  Barray[], int  ldb, const double*  beta, double* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasDgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double* const  Aarray[], int64_t  lda, const double* const  Barray[], int64_t  ldb, const double*  beta, double* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_LOG("cublasDgemmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasCgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_LOG("cublasCgemmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int  lda, const cuComplex* const  Barray[], int  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasCgemm3mBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex* const  Aarray[], int64_t  lda, const cuComplex* const  Barray[], int64_t  ldb, const cuComplex*  beta, cuComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_LOG("cublasCgemm3mBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemmBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int  lda, const cuDoubleComplex* const  Barray[], int  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int  ldc, int  batchCount)
{
	TALLY_LOG("cublasZgemmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemmBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  Aarray[], int64_t  lda, const cuDoubleComplex* const  Barray[], int64_t  ldb, const cuDoubleComplex*  beta, cuDoubleComplex* const  Carray[], int64_t  ldc, int64_t  batchCount)
{
	TALLY_LOG("cublasZgemmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const __half*  alpha, const __half*  A, int  lda, long long int  strideA, const __half*  B, int  ldb, long long int  strideB, const __half*  beta, __half*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasHgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasHgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const __half*  alpha, const __half*  A, int64_t  lda, long long int  strideA, const __half*  B, int64_t  ldb, long long int  strideB, const __half*  beta, __half*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_LOG("cublasHgemmStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const float*  alpha, const float*  A, int64_t  lda, long long int  strideA, const float*  B, int64_t  ldb, long long int  strideB, const float*  beta, float*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_LOG("cublasSgemmStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const double*  alpha, const double*  A, int  lda, long long int  strideA, const double*  B, int  ldb, long long int  strideB, const double*  beta, double*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasDgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const double*  alpha, const double*  A, int64_t  lda, long long int  strideA, const double*  B, int64_t  ldb, long long int  strideB, const double*  beta, double*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_LOG("cublasDgemmStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasCgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_LOG("cublasCgemmStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuComplex*  alpha, const cuComplex*  A, int  lda, long long int  strideA, const cuComplex*  B, int  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasCgemm3mStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgemm3mStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, long long int  strideA, const cuComplex*  B, int64_t  ldb, long long int  strideB, const cuComplex*  beta, cuComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_LOG("cublasCgemm3mStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, long long int  strideA, const cuDoubleComplex*  B, int  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int  ldc, long long int  strideC, int  batchCount)
{
	TALLY_LOG("cublasZgemmStridedBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgemmStridedBatched_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, long long int  strideA, const cuDoubleComplex*  B, int64_t  ldb, long long int  strideB, const cuDoubleComplex*  beta, cuDoubleComplex*  C, int64_t  ldc, long long int  strideC, int64_t  batchCount)
{
	TALLY_LOG("cublasZgemmStridedBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int  lda, const void* const  Barray[], cudaDataType  Btype, int  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int  ldc, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmBatchedEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void* const  Aarray[], cudaDataType  Atype, int64_t  lda, const void* const  Barray[], cudaDataType  Btype, int64_t  ldb, const void*  beta, void* const  Carray[], cudaDataType  Ctype, int64_t  ldc, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmBatchedEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, int  k, const void*  alpha, const void*  A, cudaDataType  Atype, int  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int  ldc, long long int  strideC, int  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmStridedBatchedEx hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasGemmStridedBatchedEx_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, int64_t  k, const void*  alpha, const void*  A, cudaDataType  Atype, int64_t  lda, long long int  strideA, const void*  B, cudaDataType  Btype, int64_t  ldb, long long int  strideB, const void*  beta, void*  C, cudaDataType  Ctype, int64_t  ldc, long long int  strideC, int64_t  batchCount, cublasComputeType_t  computeType, cublasGemmAlgo_t  algo)
{
	TALLY_LOG("cublasGemmStridedBatchedEx_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const float*  alpha, const float*  A, int  lda, const float*  beta, const float*  B, int  ldb, float*  C, int  ldc)
{
	TALLY_LOG("cublasSgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const float*  alpha, const float*  A, int64_t  lda, const float*  beta, const float*  B, int64_t  ldb, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSgeam_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const double*  alpha, const double*  A, int  lda, const double*  beta, const double*  B, int  ldb, double*  C, int  ldc)
{
	TALLY_LOG("cublasDgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const double*  alpha, const double*  A, int64_t  lda, const double*  beta, const double*  B, int64_t  ldb, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDgeam_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuComplex*  alpha, const cuComplex*  A, int  lda, const cuComplex*  beta, const cuComplex*  B, int  ldb, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex*  A, int64_t  lda, const cuComplex*  beta, const cuComplex*  B, int64_t  ldb, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCgeam_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeam(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int  ldb, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZgeam hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZgeam_64(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  beta, const cuDoubleComplex*  B, int64_t  ldb, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZgeam_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const float*  alpha, const float* const  A[], int  lda, float* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasStrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasStrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const float*  alpha, const float* const  A[], int64_t  lda, float* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_LOG("cublasStrsmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const double*  alpha, const double* const  A[], int  lda, double* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasDtrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDtrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const double*  alpha, const double* const  A[], int64_t  lda, double* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_LOG("cublasDtrsmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuComplex*  alpha, const cuComplex* const  A[], int  lda, cuComplex* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasCtrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCtrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuComplex*  alpha, const cuComplex* const  A[], int64_t  lda, cuComplex* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_LOG("cublasCtrsmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int  m, int  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int  lda, cuDoubleComplex* const  B[], int  ldb, int  batchCount)
{
	TALLY_LOG("cublasZtrsmBatched hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZtrsmBatched_64(cublasHandle_t  handle, cublasSideMode_t  side, cublasFillMode_t  uplo, cublasOperation_t  trans, cublasDiagType_t  diag, int64_t  m, int64_t  n, const cuDoubleComplex*  alpha, const cuDoubleComplex* const  A[], int64_t  lda, cuDoubleComplex* const  B[], int64_t  ldb, int64_t  batchCount)
{
	TALLY_LOG("cublasZtrsmBatched_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const float*  A, int  lda, const float*  x, int  incx, float*  C, int  ldc)
{
	TALLY_LOG("cublasSdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasSdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const float*  A, int64_t  lda, const float*  x, int64_t  incx, float*  C, int64_t  ldc)
{
	TALLY_LOG("cublasSdgmm_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const double*  A, int  lda, const double*  x, int  incx, double*  C, int  ldc)
{
	TALLY_LOG("cublasDdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasDdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const double*  A, int64_t  lda, const double*  x, int64_t  incx, double*  C, int64_t  ldc)
{
	TALLY_LOG("cublasDdgmm_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuComplex*  A, int  lda, const cuComplex*  x, int  incx, cuComplex*  C, int  ldc)
{
	TALLY_LOG("cublasCdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasCdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuComplex*  A, int64_t  lda, const cuComplex*  x, int64_t  incx, cuComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasCdgmm_64 hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdgmm(cublasHandle_t  handle, cublasSideMode_t  mode, int  m, int  n, const cuDoubleComplex*  A, int  lda, const cuDoubleComplex*  x, int  incx, cuDoubleComplex*  C, int  ldc)
{
	TALLY_LOG("cublasZdgmm hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasZdgmm_64(cublasHandle_t  handle, cublasSideMode_t  mode, int64_t  m, int64_t  n, const cuDoubleComplex*  A, int64_t  lda, const cuDoubleComplex*  x, int64_t  incx, cuDoubleComplex*  C, int64_t  ldc)
{
	TALLY_LOG("cublasZdgmm_64 hooked");
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

cublasStatus_t cublasUint8gemmBias(cublasHandle_t  handle, cublasOperation_t  transa, cublasOperation_t  transb, cublasOperation_t  transc, int  m, int  n, int  k, const unsigned char*  A, int  A_bias, int  lda, const unsigned char*  B, int  B_bias, int  ldb, unsigned char*  C, int  C_bias, int  ldc, int  C_mult, int  C_shift)
{
	TALLY_LOG("cublasUint8gemmBias hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cudaError_t cudaProfilerStart()
{
	TALLY_LOG("cudaProfilerStart hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaProfilerStart();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaProfilerStartArg), alignof(cudaProfilerStartArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaProfilerStart);
	return err;
}

cudaError_t cudaProfilerStop()
{
	TALLY_LOG("cudaProfilerStop hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcudaProfilerStop();
#else

    cudaError_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cudaProfilerStopArg), alignof(cudaProfilerStopArg))
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
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cudaProfilerStop);
	return err;
}

const char * nvrtcGetErrorString(nvrtcResult  result)
{
	TALLY_LOG("nvrtcGetErrorString hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcVersion(int * major, int * minor)
{
	TALLY_LOG("nvrtcVersion hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetNumSupportedArchs(int*  numArchs)
{
	TALLY_LOG("nvrtcGetNumSupportedArchs hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetSupportedArchs(int*  supportedArchs)
{
	TALLY_LOG("nvrtcGetSupportedArchs hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram * prog, const char * src, const char * name, int  numHeaders, const char * const * headers, const char * const * includeNames)
{
	TALLY_LOG("nvrtcCreateProgram hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram * prog)
{
	TALLY_LOG("nvrtcDestroyProgram hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram  prog, int  numOptions, const char * const * options)
{
	TALLY_LOG("nvrtcCompileProgram hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram  prog, size_t * ptxSizeRet)
{
	TALLY_LOG("nvrtcGetPTXSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetPTX(nvrtcProgram  prog, char * ptx)
{
	TALLY_LOG("nvrtcGetPTX hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
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

nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram  prog, size_t * LTOIRSizeRet)
{
	TALLY_LOG("nvrtcGetLTOIRSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetLTOIR(nvrtcProgram  prog, char * LTOIR)
{
	TALLY_LOG("nvrtcGetLTOIR hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram  prog, size_t * optixirSizeRet)
{
	TALLY_LOG("nvrtcGetOptiXIRSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetOptiXIR(nvrtcProgram  prog, char * optixir)
{
	TALLY_LOG("nvrtcGetOptiXIR hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram  prog, size_t * logSizeRet)
{
	TALLY_LOG("nvrtcGetProgramLogSize hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram  prog, char * log)
{
	TALLY_LOG("nvrtcGetProgramLog hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcAddNameExpression(nvrtcProgram  prog, const char * const  name_expression)
{
	TALLY_LOG("nvrtcAddNameExpression hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

nvrtcResult nvrtcGetLoweredName(nvrtcProgram  prog, const char *const  name_expression, const char**  lowered_name)
{
	TALLY_LOG("nvrtcGetLoweredName hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t*  lightHandle)
{
	TALLY_LOG("cublasLtCreate hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtCreate(lightHandle);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtCreateArg), alignof(cublasLtCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->lightHandle = lightHandle;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtCreateResponse*>(responsePayload);
			if (lightHandle) { *lightHandle = response->lightHandle; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtCreate);
	return err;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t  lightHandle)
{
	TALLY_LOG("cublasLtDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtDestroy(lightHandle);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtDestroyArg), alignof(cublasLtDestroyArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->lightHandle = lightHandle;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasLtDestroy);
	return err;
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtGetVersion();
#else

    size_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtGetVersionArg), alignof(cublasLtGetVersionArg))
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
	TALLY_LOG("cublasLtGetCudartVersion hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtGetCudartVersion();
#else

    size_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtGetCudartVersionArg), alignof(cublasLtGetCudartVersionArg))
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
	TALLY_LOG("cublasLtGetProperty hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtHeuristicsCacheGetCapacity(size_t*  capacity)
{
	TALLY_LOG("cublasLtHeuristicsCacheGetCapacity hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

cublasStatus_t cublasLtHeuristicsCacheSetCapacity(size_t  capacity)
{
	TALLY_LOG("cublasLtHeuristicsCacheSetCapacity hooked");
	throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented.");
}

unsigned cublasLtDisableCpuInstructionsSetMask(unsigned  mask)
{
	TALLY_LOG("cublasLtDisableCpuInstructionsSetMask hooked");
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatrixLayoutCreate(matLayout, type, rows, cols, ld);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatrixLayoutCreateArg), alignof(cublasLtMatrixLayoutCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatrixLayoutCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->matLayout = matLayout;
			request->type = type;
			request->rows = rows;
			request->cols = cols;
			request->ld = ld;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatrixLayoutCreateResponse*>(responsePayload);
			if (matLayout) { *matLayout = response->matLayout; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixLayoutCreate);
	return err;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t  matLayout)
{
	TALLY_LOG("cublasLtMatrixLayoutDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatrixLayoutDestroy(matLayout);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatrixLayoutDestroyArg), alignof(cublasLtMatrixLayoutDestroyArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATRIXLAYOUTDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatrixLayoutDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->matLayout = matLayout;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatrixLayoutDestroy);
	return err;
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatmulDescCreate(matmulDesc, computeType, scaleType);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatmulDescCreateArg), alignof(cublasLtMatmulDescCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCCREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulDescCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->matmulDesc = matmulDesc;
			request->computeType = computeType;
			request->scaleType = scaleType;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatmulDescCreateResponse*>(responsePayload);
			if (matmulDesc) { *matmulDesc = response->matmulDesc; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulDescCreate);
	return err;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t  matmulDesc)
{
	TALLY_LOG("cublasLtMatmulDescDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatmulDescDestroy(matmulDesc);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatmulDescDestroyArg), alignof(cublasLtMatmulDescDestroyArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMULDESCDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulDescDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->matmulDesc = matmulDesc;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulDescDestroy);
	return err;
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatmulPreferenceCreate(pref);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatmulPreferenceCreateArg), alignof(cublasLtMatmulPreferenceCreateArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCECREATE;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulPreferenceCreateArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pref = pref;

            TallyClient::client->iox_client->send(header).or_else(
                [&](auto& error) { LOG_ERR_AND_EXIT("Could not send Request: ", error); });
        })
        .or_else([](auto& error) { LOG_ERR_AND_EXIT("Could not allocate Request: ", error); });

    while(!TallyClient::client->iox_client->take()
        .and_then([&](const auto& responsePayload) {
            auto response = static_cast<const cublasLtMatmulPreferenceCreateResponse*>(responsePayload);
			if (pref) { *pref = response->pref; }

            err = response->err;
            TallyClient::client->iox_client->releaseResponse(responsePayload);
        }))
    {};
#endif
	TALLY_CLIENT_PROFILE_END;
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulPreferenceCreate);
	return err;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t  pref)
{
	TALLY_LOG("cublasLtMatmulPreferenceDestroy hooked");
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtMatmulPreferenceDestroy(pref);
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtMatmulPreferenceDestroyArg), alignof(cublasLtMatmulPreferenceDestroyArg))
        .and_then([&](auto& requestPayload) {

            auto header = static_cast<MessageHeader_t*>(requestPayload);
            header->api_id = CUDA_API_ENUM::CUBLASLTMATMULPREFERENCEDESTROY;
            header->client_id = TallyClient::client->client_id;
            
            auto request = (cublasLtMatmulPreferenceDestroyArg*) (static_cast<uint8_t*>(requestPayload) + sizeof(MessageHeader_t));
			request->pref = pref;

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
	TALLY_CLIENT_TRACE_API_CALL(cublasLtMatmulPreferenceDestroy);
	return err;
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
	TALLY_CLIENT_PROFILE_START;
#if defined(RUN_LOCALLY)
	auto err = lcublasLtLoggerForceDisable();
#else

    cublasStatus_t err;

    TallyClient::client->iox_client->loan(sizeof(MessageHeader_t) + sizeof(cublasLtLoggerForceDisableArg), alignof(cublasLtLoggerForceDisableArg))
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



}

