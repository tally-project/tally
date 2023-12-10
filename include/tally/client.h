#ifndef TALLY_CLIENT_H
#define TALLY_CLIENT_H

#include <signal.h>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <sstream>
#include <unistd.h>

#include "iceoryx_dust/posix_wrapper/signal_watcher.hpp"
#include "iceoryx_posh/popo/untyped_client.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iox/detail/unique_id.hpp"

#include "tally/msg_struct.h"

#ifdef ENABLE_PROFILING
    #define TALLY_CLIENT_PROFILE_START \
        auto __tally_call_start = std::chrono::high_resolution_clock::now();

    #define TALLY_CLIENT_PROFILE_END \
        auto __tally_call_end = std::chrono::high_resolution_clock::now(); \
        TallyClient::client->_profile_cpu_timestamps.push_back({ __tally_call_start, __tally_call_end });   \
        auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(__tally_call_start.time_since_epoch()).count();    \
        auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(__tally_call_end.time_since_epoch()).count();    \
        std::cout << "Duration: " << end_ns - start_ns << "ns" << std::endl;

    #define TALLY_CLIENT_TRACE_API_CALL(CLIENT_API_CALL) \
        TallyClient::client->_profile_kernel_seq.push_back((void *) l##CLIENT_API_CALL);    \
        std::cout << #CLIENT_API_CALL << std::endl; \

    #define TALLY_CLIENT_TRACE_KERNEL_CALL(FUNC) \
        TallyClient::client->_profile_kernel_seq.push_back((void *) FUNC);
#else
    #define TALLY_CLIENT_PROFILE_START
    #define TALLY_CLIENT_PROFILE_END
    #define TALLY_CLIENT_TRACE_API_CALL(CLIENT_API_CALL)
    #define TALLY_CLIENT_TRACE_KERNEL_CALL(FUNC)
#endif

class TallyClient {

typedef std::chrono::time_point<std::chrono::system_clock> time_point_t;

public:

    static TallyClient *client;

    int32_t client_id;

    std::mutex iox_mtx;

    // For performance measurements
    std::vector<const void *> _profile_kernel_seq;
    std::vector<std::pair<time_point_t, time_point_t>> _profile_cpu_timestamps;
    std::map<const void *, std::string> _profile_kernel_map;

    std::map<const void *, std::string> host_func_to_demangled_kernel_name_map;
    std::map<std::string, std::vector<uint32_t>> _kernel_name_to_args;

    std::unordered_map<const void *, std::vector<uint32_t>> _kernel_addr_to_args;
    std::unordered_map<CUfunction, std::vector<uint32_t>> _jit_kernel_addr_to_args;

#ifndef RUN_LOCALLY
    iox::popo::UntypedClient *iox_client;
#endif

    void register_profile_kernel_map();
    void print_profile_trace()
    {
        assert(_profile_cpu_timestamps.size() == _profile_kernel_seq.size());
        for (size_t i = 0; i < _profile_kernel_seq.size(); i++) {
            auto _trace_addr = _profile_kernel_seq[i];
            std::string _trace_name;
            if (_profile_kernel_map.find(_trace_addr) != _profile_kernel_map.end()) {
                _trace_name = _profile_kernel_map[_trace_addr];
            } else if (host_func_to_demangled_kernel_name_map.find(_trace_addr) != host_func_to_demangled_kernel_name_map.end()) {
                _trace_name = host_func_to_demangled_kernel_name_map[_trace_addr];
            } else {
                std::cerr << "Cannot find _trace_addr in _profile_kernel_map" << std::endl;
                continue;
            }

            std::ostringstream stream;

            stream << _trace_name;
            auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    _profile_cpu_timestamps[i].first.time_since_epoch()).count();
            auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    _profile_cpu_timestamps[i].second.time_since_epoch()).count();

            stream << " Duration: " << end_ns - start_ns << "ns";

            std::cout << stream.str() << std::endl;
        }
    }

    TallyClient()
    {
        client_id = getpid();
        register_profile_kernel_map();

#ifndef RUN_LOCALLY

        int32_t priority = std::getenv("PRIORITY") ? std::stoi(std::getenv("PRIORITY")) : 1;

        auto app_name_str_base = std::string("tally-client-app");
        auto app_name_str = app_name_str_base + std::to_string(client_id);

        char APP_NAME[100];
        strcpy(APP_NAME, app_name_str.c_str()); 

        iox::runtime::PoshRuntime::initRuntime(APP_NAME);

        iox::popo::UntypedClient client_handshake({"Tally", "handshake", "event"});

        // Send handshake to server
        client_handshake.loan(sizeof(HandshakeMessgae), alignof(HandshakeMessgae))
            .and_then([&](auto& requestPayload) {

                auto request = static_cast<HandshakeMessgae*>(requestPayload);
                request->client_id = client_id;
                request->priority = priority;

                client_handshake.send(request).or_else(
                    [&](auto& error) { std::cout << "Could not send Request! Error: " << error << std::endl; });
            })
            .or_else([](auto& error) { std::cout << "Could not allocate Request! Error: " << error << std::endl; });

        while (!client_handshake.take().and_then([&](const auto& responsePayload) {

            auto response = static_cast<const HandshakeResponse*>(responsePayload);
            
            bool success = response->success;
            if (!success) {
                std::cout << "Handshake with tally server failed. Exiting ..." << std::endl;
                exit(1);
            }

            client_handshake.releaseResponse(responsePayload);

        })) {};

        auto channel_desc_str = std::string("Tally-Communication") + std::to_string(client_id);
        char channel_desc[100];
        strcpy(channel_desc, channel_desc_str.c_str()); 
        iox_client = new iox::popo::UntypedClient({channel_desc, "tally", "tally"});
#endif
    }

    ~TallyClient(){
        print_profile_trace();
    }
};

const std::vector<std::string> preload_libs {
    "libcuda.so.1",
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcufft.so.11",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libcudnn.so.8"
};

const std::vector<std::string> cuGetProcAddress_funcs = {
    "cuInit",
    "cuMemcpy",
    "cuMemcpyPeer",
    "cuMemcpyHtoD",
    "cuMemcpyDtoH",
    "cuMemcpyDtoD",
    "cuMemcpyDtoA",
    "cuMemcpyAtoD",
    "cuMemcpyHtoA",
    "cuMemcpyAtoH",
    "cuMemcpyAtoA",
    "cuMemcpy3DPeer",
    "cuMemcpyAsync",
    "cuMemcpy3D",
    "cuMemcpy2DUnaligned",
    "cuMemcpy2D",
    "cuMemcpyPeerAsync",
    "cuMemcpyHtoDAsync",
    "cuMemcpyDtoHAsync",
    "cuMemcpyDtoDAsync",
    "cuDeviceGet",
    "cuDeviceGetName",
    "cuDeviceGetCount",
    "cuStreamBeginCapture_v2",
    "cuGetErrorString",
    "cuGetErrorName",
    "cuDriverGetVersion",
    "cuDeviceGetUuid",
    "cuDeviceGetUuid_v2",
    "cuDeviceTotalMem_v2",
    "cuDeviceGetTexture1DLinearMaxWidth",
    "cuDeviceGetAttribute",
    "cuDeviceGetNvSciSyncAttributes",
    "cuDeviceSetMemPool",
    "cuDeviceGetMemPool",
    "cuDeviceGetDefaultMemPool",
    "cuDeviceGetExecAffinitySupport",
    "cuDeviceGetP2PAttribute",
    "cuFlushGPUDirectRDMAWrites",
    "cuDeviceGetProperties",
    "cuDeviceComputeCapability",
    "cuDevicePrimaryCtxRetain",
    "cuDevicePrimaryCtxRelease_v2",
    "cuDevicePrimaryCtxSetFlags_v2",
    "cuDevicePrimaryCtxGetState",
    "cuDevicePrimaryCtxReset_v2",
    "cuCtxCreate_v2",
    "cuCtxCreate_v3",
    "cuCtxDestroy_v2",
    "cuCtxPushCurrent_v2",
    "cuCtxPopCurrent_v2",
    "cuCtxSetCurrent",
    "cuCtxGetCurrent",
    "cuCtxGetDevice",
    "cuCtxGetFlags",
    "cuCtxSetFlags",
    "cuCtxGetId",
    "cuCtxSynchronize",
    "cuCtxSetLimit",
    "cuCtxGetLimit",
    "cuCtxGetCacheConfig",
    "cuCtxSetCacheConfig",
    "cuCtxGetSharedMemConfig",
    "cuCtxSetSharedMemConfig",
    "cuCtxGetApiVersion",
    "cuCtxGetStreamPriorityRange",
    "cuCtxResetPersistingL2Cache",
    "cuCtxGetExecAffinity",
    "cuCtxAttach",
    "cuCtxDetach",
    "cuModuleLoad",
    "cuModuleLoadData",
    "cuModuleLoadDataEx",
    "cuModuleLoadFatBinary",
    "cuModuleUnload",
    "cuModuleGetLoadingMode",
    "cuModuleGetFunction",
    "cuModuleGetGlobal_v2",
    "cuLinkCreate_v2",
    "cuLinkAddData_v2",
    "cuLinkAddFile_v2",
    "cuLinkComplete",
    "cuLinkDestroy",
    "cuModuleGetTexRef",
    "cuModuleGetSurfRef",
    "cuLibraryLoadData",
    "cuLibraryLoadFromFile",
    "cuLibraryUnload",
    "cuLibraryGetKernel",
    "cuLibraryGetModule",
    "cuKernelGetFunction",
    "cuLibraryGetGlobal",
    "cuLibraryGetManaged",
    "cuLibraryGetUnifiedFunction",
    "cuKernelGetAttribute",
    "cuKernelSetAttribute",
    "cuKernelSetCacheConfig",
    "cuMemGetInfo_v2",
    "cuMemAlloc_v2",
    "cuMemAllocPitch_v2",
    "cuMemFree_v2",
    "cuMemGetAddressRange_v2",
    "cuMemAllocHost_v2",
    "cuMemFreeHost",
    "cuMemHostAlloc",
    "cuMemHostGetDevicePointer_v2",
    "cuMemHostGetFlags",
    "cuMemAllocManaged",
    "cuDeviceGetByPCIBusId",
    "cuDeviceGetPCIBusId",
    "cuIpcGetEventHandle",
    "cuIpcOpenEventHandle",
    "cuIpcGetMemHandle",
    "cuIpcOpenMemHandle_v2",
    "cuIpcCloseMemHandle",
    "cuMemHostRegister_v2",
    "cuMemHostUnregister",
    "cuMemcpy",
    "cuMemcpyPeer",
    "cuMemcpyHtoD_v2",
    "cuMemcpyDtoH_v2",
    "cuMemcpyDtoD_v2",
    "cuMemcpyDtoA_v2",
    "cuMemcpyAtoD_v2",
    "cuMemcpyHtoA_v2",
    "cuMemcpyAtoH_v2",
    "cuMemcpyAtoA_v2",
    "cuMemcpy2D_v2",
    "cuMemcpy2DUnaligned_v2",
    "cuMemcpy3D_v2",
    "cuMemcpy3DPeer",
    "cuMemcpyAsync",
    "cuMemcpyPeerAsync",
    "cuMemcpyHtoDAsync_v2",
    "cuMemcpyDtoHAsync_v2",
    "cuMemcpyDtoDAsync_v2",
    "cuMemcpyHtoAAsync_v2",
    "cuMemcpyAtoHAsync_v2",
    "cuMemcpy2DAsync_v2",
    "cuMemcpy3DAsync_v2",
    "cuMemcpy3DPeerAsync",
    "cuMemsetD8_v2",
    "cuMemsetD16_v2",
    "cuMemsetD32_v2",
    "cuMemsetD2D8_v2",
    "cuMemsetD2D16_v2",
    "cuMemsetD2D32_v2",
    "cuMemsetD8Async",
    "cuMemsetD16Async",
    "cuMemsetD32Async",
    "cuMemsetD2D8Async",
    "cuMemsetD2D16Async",
    "cuMemsetD2D32Async",
    "cuArrayCreate_v2",
    "cuArrayGetDescriptor_v2",
    "cuArrayGetSparseProperties",
    "cuMipmappedArrayGetSparseProperties",
    "cuArrayGetMemoryRequirements",
    "cuMipmappedArrayGetMemoryRequirements",
    "cuArrayGetPlane",
    "cuArrayDestroy",
    "cuArray3DCreate_v2",
    "cuArray3DGetDescriptor_v2",
    "cuMipmappedArrayCreate",
    "cuMipmappedArrayGetLevel",
    "cuMipmappedArrayDestroy",
    "cuMemGetHandleForAddressRange",
    "cuMemAddressReserve",
    "cuMemAddressFree",
    "cuMemCreate",
    "cuMemRelease",
    "cuMemMap",
    "cuMemMapArrayAsync",
    "cuMemUnmap",
    "cuMemSetAccess",
    "cuMemGetAccess",
    "cuMemExportToShareableHandle",
    "cuMemImportFromShareableHandle",
    "cuMemGetAllocationGranularity",
    "cuMemGetAllocationPropertiesFromHandle",
    "cuMemRetainAllocationHandle",
    "cuMemFreeAsync",
    "cuMemAllocAsync",
    "cuMemPoolTrimTo",
    "cuMemPoolSetAttribute",
    "cuMemPoolGetAttribute",
    "cuMemPoolSetAccess",
    "cuMemPoolGetAccess",
    "cuMemPoolCreate",
    "cuMemPoolDestroy",
    "cuMemAllocFromPoolAsync",
    "cuMemPoolExportToShareableHandle",
    "cuMemPoolImportFromShareableHandle",
    "cuMemPoolExportPointer",
    "cuMemPoolImportPointer",
    "cuMulticastCreate",
    "cuMulticastAddDevice",
    "cuMulticastBindMem",
    "cuMulticastBindAddr",
    "cuMulticastUnbind",
    "cuMulticastGetGranularity",
    "cuPointerGetAttribute",
    "cuMemPrefetchAsync",
    "cuMemPrefetchAsync_v2",
    "cuMemAdvise",
    "cuMemAdvise_v2",
    "cuMemRangeGetAttribute",
    "cuMemRangeGetAttributes",
    "cuPointerSetAttribute",
    "cuPointerGetAttributes",
    "cuStreamCreate",
    "cuStreamCreateWithPriority",
    "cuStreamGetPriority",
    "cuStreamGetFlags",
    "cuStreamGetId",
    "cuStreamGetCtx",
    "cuStreamWaitEvent",
    "cuStreamAddCallback",
    "cuStreamBeginCapture_v2",
    "cuThreadExchangeStreamCaptureMode",
    "cuStreamEndCapture",
    "cuStreamIsCapturing",
    "cuStreamGetCaptureInfo_v2",
    "cuStreamUpdateCaptureDependencies",
    "cuStreamAttachMemAsync",
    "cuStreamQuery",
    "cuStreamSynchronize",
    "cuStreamDestroy_v2",
    "cuStreamCopyAttributes",
    "cuStreamGetAttribute",
    "cuStreamSetAttribute",
    "cuEventCreate",
    "cuEventRecord",
    "cuEventRecordWithFlags",
    "cuEventQuery",
    "cuEventSynchronize",
    "cuEventDestroy_v2",
    "cuEventElapsedTime",
    "cuImportExternalMemory",
    "cuExternalMemoryGetMappedBuffer",
    "cuExternalMemoryGetMappedMipmappedArray",
    "cuDestroyExternalMemory",
    "cuImportExternalSemaphore",
    "cuSignalExternalSemaphoresAsync",
    "cuWaitExternalSemaphoresAsync",
    "cuDestroyExternalSemaphore",
    "cuStreamWaitValue32_v2",
    "cuStreamWaitValue64_v2",
    "cuStreamWriteValue32_v2",
    "cuStreamWriteValue64_v2",
    "cuStreamBatchMemOp_v2",
    "cuFuncGetAttribute",
    "cuFuncSetAttribute",
    "cuFuncSetCacheConfig",
    "cuFuncSetSharedMemConfig",
    "cuFuncGetModule",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cuLaunchCooperativeKernel",
    "cuLaunchCooperativeKernelMultiDevice",
    "cuLaunchHostFunc",
    "cuFuncSetBlockShape",
    "cuFuncSetSharedSize",
    "cuParamSetSize",
    "cuParamSeti",
    "cuParamSetf",
    "cuParamSetv",
    "cuLaunch",
    "cuLaunchGrid",
    "cuLaunchGridAsync",
    "cuParamSetTexRef",
    "cuGraphCreate",
    "cuGraphAddKernelNode_v2",
    "cuGraphKernelNodeGetParams_v2",
    "cuGraphKernelNodeSetParams_v2",
    "cuGraphAddMemcpyNode",
    "cuGraphMemcpyNodeGetParams",
    "cuGraphMemcpyNodeSetParams",
    "cuGraphAddMemsetNode",
    "cuGraphMemsetNodeGetParams",
    "cuGraphMemsetNodeSetParams",
    "cuGraphAddHostNode",
    "cuGraphHostNodeGetParams",
    "cuGraphHostNodeSetParams",
    "cuGraphAddChildGraphNode",
    "cuGraphChildGraphNodeGetGraph",
    "cuGraphAddEmptyNode",
    "cuGraphAddEventRecordNode",
    "cuGraphEventRecordNodeGetEvent",
    "cuGraphEventRecordNodeSetEvent",
    "cuGraphAddEventWaitNode",
    "cuGraphEventWaitNodeGetEvent",
    "cuGraphEventWaitNodeSetEvent",
    "cuGraphAddExternalSemaphoresSignalNode",
    "cuGraphExternalSemaphoresSignalNodeGetParams",
    "cuGraphExternalSemaphoresSignalNodeSetParams",
    "cuGraphAddExternalSemaphoresWaitNode",
    "cuGraphExternalSemaphoresWaitNodeGetParams",
    "cuGraphExternalSemaphoresWaitNodeSetParams",
    "cuGraphAddBatchMemOpNode",
    "cuGraphBatchMemOpNodeGetParams",
    "cuGraphBatchMemOpNodeSetParams",
    "cuGraphExecBatchMemOpNodeSetParams",
    "cuGraphAddMemAllocNode",
    "cuGraphMemAllocNodeGetParams",
    "cuGraphAddMemFreeNode",
    "cuGraphMemFreeNodeGetParams",
    "cuDeviceGraphMemTrim",
    "cuDeviceGetGraphMemAttribute",
    "cuDeviceSetGraphMemAttribute",
    "cuGraphClone",
    "cuGraphNodeFindInClone",
    "cuGraphNodeGetType",
    "cuGraphGetNodes",
    "cuGraphGetRootNodes",
    "cuGraphGetEdges",
    "cuGraphNodeGetDependencies",
    "cuGraphNodeGetDependentNodes",
    "cuGraphAddDependencies",
    "cuGraphRemoveDependencies",
    "cuGraphDestroyNode",
    "cuGraphInstantiate",
    "cuGraphInstantiateWithFlags",
    "cuGraphInstantiateWithParams",
    "cuGraphInstantiateWithParams_ptsz",
    "cuGraphExecGetFlags",
    "cuGraphExecKernelNodeSetParams_v2",
    "cuGraphExecMemcpyNodeSetParams",
    "cuGraphExecMemsetNodeSetParams",
    "cuGraphExecHostNodeSetParams",
    "cuGraphExecChildGraphNodeSetParams",
    "cuGraphExecEventRecordNodeSetEvent",
    "cuGraphExecEventWaitNodeSetEvent",
    "cuGraphExecExternalSemaphoresSignalNodeSetParams",
    "cuGraphExecExternalSemaphoresWaitNodeSetParams",
    "cuGraphNodeSetEnabled",
    "cuGraphNodeGetEnabled",
    "cuGraphUpload",
    "cuGraphLaunch",
    "cuGraphExecDestroy",
    "cuGraphDestroy",
    "cuGraphExecUpdate_v2",
    "cuGraphKernelNodeCopyAttributes",
    "cuGraphKernelNodeGetAttribute",
    "cuGraphKernelNodeSetAttribute",
    "cuGraphDebugDotPrint",
    "cuUserObjectCreate",
    "cuUserObjectRetain",
    "cuUserObjectRelease",
    "cuGraphRetainUserObject",
    "cuGraphReleaseUserObject",
    "cuGraphAddNode",
    "cuGraphNodeSetParams",
    "cuGraphExecNodeSetParams",
    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cuOccupancyMaxPotentialBlockSize",
    "cuOccupancyMaxPotentialBlockSizeWithFlags",
    "cuOccupancyAvailableDynamicSMemPerBlock",
    "cuOccupancyMaxPotentialClusterSize",
    "cuOccupancyMaxActiveClusters",
    "cuTexRefSetArray",
    "cuTexRefSetMipmappedArray",
    "cuTexRefSetAddress_v2",
    "cuTexRefSetAddress2D_v3",
    "cuTexRefSetFormat",
    "cuTexRefSetAddressMode",
    "cuTexRefSetFilterMode",
    "cuTexRefSetMipmapFilterMode",
    "cuTexRefSetMipmapLevelBias",
    "cuTexRefSetMipmapLevelClamp",
    "cuTexRefSetMaxAnisotropy",
    "cuTexRefSetBorderColor",
    "cuTexRefSetFlags",
    "cuTexRefGetAddress_v2",
    "cuTexRefGetArray",
    "cuTexRefGetMipmappedArray",
    "cuTexRefGetAddressMode",
    "cuTexRefGetFilterMode",
    "cuTexRefGetFormat",
    "cuTexRefGetMipmapFilterMode",
    "cuTexRefGetMipmapLevelBias",
    "cuTexRefGetMipmapLevelClamp",
    "cuTexRefGetMaxAnisotropy",
    "cuTexRefGetBorderColor",
    "cuTexRefGetFlags",
    "cuTexRefCreate",
    "cuTexRefDestroy",
    "cuSurfRefSetArray",
    "cuSurfRefGetArray",
    "cuTexObjectCreate",
    "cuTexObjectDestroy",
    "cuTexObjectGetResourceDesc",
    "cuTexObjectGetTextureDesc",
    "cuTexObjectGetResourceViewDesc",
    "cuSurfObjectCreate",
    "cuSurfObjectDestroy",
    "cuSurfObjectGetResourceDesc",
    "cuTensorMapEncodeTiled",
    "cuTensorMapEncodeIm2col",
    "cuTensorMapReplaceAddress",
    "cuDeviceCanAccessPeer",
    "cuCtxEnablePeerAccess",
    "cuCtxDisablePeerAccess",
    "cuDeviceGetP2PAttribute",
    "cuGraphicsUnregisterResource",
    "cuGraphicsSubResourceGetMappedArray",
    "cuGraphicsResourceGetMappedMipmappedArray",
    "cuGraphicsResourceGetMappedPointer_v2",
    "cuGraphicsResourceSetMapFlags_v2",
    "cuGraphicsMapResources",
    "cuGraphicsUnmapResources",
    "cuCoredumpGetAttribute",
    "cuCoredumpGetAttributeGlobal",
    "cuCoredumpSetAttribute",
    "cuCoredumpSetAttributeGlobal",
    "cuGetExportTable",
    "cuProfilerStart",
    "cuProfilerStop",
    "cudaGetTextureReference",
    "cudaGetSurfaceReference",
    "cudaBindTexture",
    "cudaBindTexture2D",
    "cudaBindTextureToArray",
    "cudaBindTextureToMipmappedArray",
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
    "cudaMemcpyToSymbolAsync",
    "cudaMemcpyFromSymbolAsync",
    "cudaGetSymbolAddress",
    "cudaGetSymbolSize",
    "cudaUnbindTexture",
    "cudaGetTextureAlignmentOffset",
    "cudaBindSurfaceToArray",
    "cudaGetFuncBySymbol",
    "cudaSetValidDevices",
    "cudaGraphExecMemcpyNodeSetParamsFromSymbol",
    "cudaGraphExecMemcpyNodeSetParamsToSymbol",
    "cudaGraphAddMemcpyNodeToSymbol",
    "cudaGraphAddMemcpyNodeFromSymbol",
    "cudaGraphMemcpyNodeSetParamsToSymbol",
    "cudaGraphMemcpyNodeSetParamsFromSymbol",
    "cudaProfilerInitialize",
    "cudaProfilerStart",
    "cudaProfilerStop",
    "cuProfilerInitialize",
    "cuProfilerStart",
    "cuProfilerStop",
    "cuDeviceGetLuid"
};

const std::vector<std::string> cuGetProcAddress_v2funcs = {
    "cuDeviceTotalMem",
    "cuStreamBeginCapture",
    "cuDevicePrimaryCtxRelease",
    "cuDevicePrimaryCtxSetFlags",
    "cuDevicePrimaryCtxReset",
    "cuCtxCreate",
    "cuCtxDestroy",
    "cuCtxPushCurrent",
    "cuCtxPopCurrent",
    "cuModuleGetGlobal",
    "cuLinkCreate",
    "cuLinkAddData",
    "cuLinkAddFile",
    "cuMemGetInfo",
    "cuMemAlloc",
    "cuMemAllocPitch",
    "cuMemFree",
    "cuMemGetAddressRange",
    "cuMemAllocHost",
    "cuMemHostGetDevicePointer",
    "cuIpcOpenMemHandle",
    "cuMemHostRegister",
    "cuMemcpyHtoD",
    "cuMemcpyDtoH",
    "cuMemcpyDtoD",
    "cuMemcpyDtoA",
    "cuMemcpyAtoD",
    "cuMemcpyHtoA",
    "cuMemcpyAtoH",
    "cuMemcpyAtoA",
    "cuMemcpy2D",
    "cuMemcpy2DUnaligned",
    "cuMemcpy3D",
    "cuMemcpyHtoDAsync",
    "cuMemcpyDtoHAsync",
    "cuMemcpyDtoDAsync",
    "cuMemcpyHtoAAsync",
    "cuMemcpyAtoHAsync",
    "cuMemcpy2DAsync",
    "cuMemcpy3DAsync",
    "cuMemsetD8",
    "cuMemsetD16",
    "cuMemsetD32",
    "cuMemsetD2D8",
    "cuMemsetD2D16",
    "cuMemsetD2D32",
    "cuArrayCreate",
    "cuArrayGetDescriptor",
    "cuArray3DCreate",
    "cuArray3DGetDescriptor",
    "cuMemPrefetchAsync",
    "cuMemAdvise",
    "cuStreamBeginCapture",
    "cuStreamGetCaptureInfo",
    "cuStreamDestroy",
    "cuEventDestroy",
    "cuStreamWaitValue32",
    "cuStreamWaitValue64",
    "cuStreamWriteValue32",
    "cuStreamWriteValue64",
    "cuStreamBatchMemOp",
    "cuGraphAddKernelNode",
    "cuGraphKernelNodeGetParams",
    "cuGraphKernelNodeSetParams",
    "cuGraphExecKernelNodeSetParams",
    "cuGraphExecUpdate",
    "cuTexRefSetAddress",
    "cuTexRefGetAddress",
    "cuGraphicsResourceGetMappedPointer",
    "cuGraphicsResourceSetMapFlags"
};

const std::vector<std::string> cuGetProcAddress_v3funcs = {
    "cuTexRefSetAddress2D"
};


const std::vector<std::string> cuGetProcAddress_direct_call_funcs = {
    "cuGraphicsGLRegisterBuffer",
    "cuGraphicsGLRegisterImage",
    "cuWGLGetDevice",
    "cuGLGetDevices",
    "cuGLCtxCreate",
    "cuGLInit",
    "cuGLRegisterBufferObject",
    "cuGLMapBufferObject",
    "cuGLUnmapBufferObject",
    "cuGLUnregisterBufferObject",
    "cuGLSetBufferObjectMapFlags",
    "cuGLMapBufferObjectAsync",
    "cuGLUnmapBufferObjectAsync",
    "cudaGLGetDevices",
    "cudaGraphicsGLRegisterImage",
    "cudaGraphicsGLRegisterBuffer",
    "cudaWGLGetDevice",
    "cudaGLSetGLDevice",
    "cudaGLRegisterBufferObject",
    "cudaGLMapBufferObject",
    "cudaGLUnmapBufferObject",
    "cudaGLUnregisterBufferObject",
    "cudaGLSetBufferObjectMapFlags",
    "cudaGLMapBufferObjectAsync",
    "cudaGLUnmapBufferObjectAsync",
    "cuGraphicsEGLRegisterImage",
    "cuEGLStreamConsumerConnect",
    "cuEGLStreamConsumerConnectWithFlags",
    "cuEGLStreamConsumerDisconnect",
    "cuEGLStreamConsumerAcquireFrame",
    "cuEGLStreamConsumerReleaseFrame",
    "cuEGLStreamProducerConnect",
    "cuEGLStreamProducerDisconnect",
    "cuEGLStreamProducerPresentFrame",
    "cuEGLStreamProducerReturnFrame",
    "cuGraphicsResourceGetMappedEglFrame",
    "cuEventCreateFromEGLSync",
    "cudaGraphicsEGLRegisterImage",
    "cudaEGLStreamConsumerConnect",
    "cudaEGLStreamConsumerConnectWithFlags",
    "cudaEGLStreamConsumerDisconnect",
    "cudaEGLStreamConsumerAcquireFrame",
    "cudaEGLStreamConsumerReleaseFrame",
    "cudaEGLStreamProducerConnect",
    "cudaEGLStreamProducerDisconnect",
    "cudaEGLStreamProducerPresentFrame",
    "cudaEGLStreamProducerReturnFrame",
    "cudaGraphicsResourceGetMappedEglFrame",
    "cudaEventCreateFromEGLSync",
    "cuVDPAUGetDevice",
    "cuVDPAUCtxCreate",
    "cuGraphicsVDPAURegisterVideoSurface",
    "cuGraphicsVDPAURegisterOutputSurface",
    "cudaVDPAUGetDevice",
    "cudaVDPAUSetVDPAUDevice",
    "cudaGraphicsVDPAURegisterVideoSurface",
    "cudaGraphicsVDPAURegisterOutputSurface"
};

#endif // TALLY_CLIENT_H