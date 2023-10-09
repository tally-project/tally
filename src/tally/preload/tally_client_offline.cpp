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
#include <string>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include "tally/log.h"
#include <tally/cache_util.h>
#include <tally/util.h>
#include <tally/msg_struct.h>
#include <tally/env.h>
#include <tally/transform.h>
#include <tally/generated/cuda_api.h>
#include <tally/client_offline.h>

const char *cubin_data;
size_t cubin_size;

// Store cubin uid
std::vector<uint32_t> cubin_register_queue;

void register_kernels()
{
    if (cubin_register_queue.size() > 0) {

        for (auto cubin_uid : cubin_register_queue) {
            auto cubin_data = TallyCache::cache->cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
            auto cubin_size = TallyCache::cache->cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
            TallyClientOffline::client_offline->register_ptx_transform(cubin_data, cubin_size);
        }

        cubin_register_queue.clear();

        TallyClientOffline::client_offline->register_measurements();
    }
}

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    register_kernels();

    CudaLaunchCall launch_call(func, gridDim, blockDim);

    // Look up cache for best-performance config
    bool found_in_cache;
    auto res = TallyClientOffline::client_offline->get_single_kernel_best_config(launch_call, &found_in_cache);

    if (!found_in_cache) {

        auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
        auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;
        auto configs = CudaLaunchConfig::get_workload_agnostic_sharing_configs(threads_per_block, num_blocks);

        TallyClientOffline::client_offline->tune_kernel_launch(configs, func, gridDim, blockDim, args, sharedMem, stream);
        res = TallyClientOffline::client_offline->get_single_kernel_best_config(launch_call, &found_in_cache);
        assert(found_in_cache);
    }

    auto config = res.config;

    config = CudaLaunchConfig::default_config;

    auto err = TallyClientOffline::client_offline->launch_kernel(config, func, gridDim, blockDim, args, sharedMem, stream);
    if (!err) {
        return cudaSuccess;
    } else {
        return cudaErrorInvalidValue;
    }
}

CUresult cuLaunchKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra)
{
    auto func = TallyClientOffline::client_offline->cu_func_addr_mapping[f];

    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    dim3 blockDim(blockDimX, blockDimY, blockDimZ);

    CudaLaunchCall launch_call(func, gridDim, blockDim);

    // Look up cache for best-performance config
    bool found_in_cache;
    auto res = TallyClientOffline::client_offline->get_single_kernel_best_config(launch_call, &found_in_cache);

    if (!found_in_cache) {

        auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
        auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;
        auto configs = CudaLaunchConfig::get_workload_agnostic_sharing_configs(threads_per_block, num_blocks);

        TallyClientOffline::client_offline->tune_kernel_launch(configs, func, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);
        res = TallyClientOffline::client_offline->get_single_kernel_best_config(launch_call, &found_in_cache);
        assert(found_in_cache);
    }

    auto config = res.config;

    auto err = TallyClientOffline::client_offline->launch_kernel(config, func, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);

    return err;
}

void** __cudaRegisterFatBinary( void *fatCubin ) {
    // std::cout << "__cudaRegisterFatBinary" << std::endl;

    auto wp = (__fatBinC_Wrapper_t *) fatCubin;
    auto fbh = (struct fatBinaryHeader *) wp->data;
    cubin_data = (const char *) wp->data;
    cubin_size = fbh->headerSize + fbh->fatSize;

    // cache fatbin
    cache_cubin_data(cubin_data, cubin_size);

    uint32_t cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
    cubin_register_queue.push_back(cubin_uid);

    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    // std::cout << "__cudaRegisterFunction" << std::endl;

    auto kernel_name = std::string(deviceName);
    auto demangled_kernel_name = demangleFunc(kernel_name);

    uint32_t cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);

    TallyClientOffline::client_offline->host_func_to_demangled_kernel_name_map[hostFun] = demangled_kernel_name;
    TallyClientOffline::client_offline->host_func_to_cubin_uid_map[hostFun] = cubin_uid;

    TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name] = hostFun;
    TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid][demangled_kernel_name] = hostFun;

    TallyClientOffline::client_offline->demangled_kernel_name_and_cubin_uid_to_host_func_map[
        std::make_pair(demangled_kernel_name, cubin_uid)
    ] = hostFun;

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
    l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

CUresult cuModuleLoadData(CUmodule * module, const void * image)
{
    // std::cout << "cuModuleLoadData" << std::endl;
    auto fbh = (struct fatBinaryHeader *) image;
    const char *cubin_data = (const char *) image;
    size_t cubin_size = fbh->headerSize + fbh->fatSize;

    // cache fatbin
    cache_cubin_data(cubin_data, cubin_size);

    uint32_t cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
    auto err = lcuModuleLoadData(module, image);

    auto cached_cubin_data = TallyCache::cache->cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
    auto cached_cubin_size = TallyCache::cache->cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);

    TallyClientOffline::client_offline->jit_module_to_cubin_map[*module] = std::make_pair(cached_cubin_data, cached_cubin_size);

    return err;
}

CUresult cuModuleGetFunction(CUfunction * hfunc, CUmodule  hmod, const char * name)
{
    // std::cout << "cuModuleGetFunction" << std::endl;

    auto err = lcuModuleGetFunction(hfunc, hmod, name);

    auto kernel_name = std::string(name);

    auto cubin_data_size = TallyClientOffline::client_offline->jit_module_to_cubin_map[hmod];
    auto cubin_data = cubin_data_size.first;
    auto cubin_size = cubin_data_size.second;

    uint32_t cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);

    if (TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid].find(kernel_name) == TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid].end()) {
        void *hostFun = malloc(8);

        TallyClientOffline::client_offline->host_func_to_demangled_kernel_name_map[hostFun] = kernel_name;
        TallyClientOffline::client_offline->host_func_to_cubin_uid_map[hostFun] = cubin_uid;

        TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name] = hostFun;
    
        TallyClientOffline::client_offline->cu_func_addr_mapping[*hfunc] = hostFun;

        TallyClientOffline::client_offline->demangled_kernel_name_and_cubin_uid_to_host_func_map[
            std::make_pair(kernel_name, cubin_uid)
        ] = hostFun;

        TallyClientOffline::client_offline->register_measurements();
    }

    TallyClientOffline::client_offline->register_ptx_transform(cubin_data, cubin_size);

    return err;
}

cudaError_t cudaFuncSetAttribute(const void * func, enum cudaFuncAttribute  attr, int  value)
{
    register_kernels();

    auto cu_func = TallyClientOffline::client_offline->original_kernel_map[func].func;
    auto cu_func_ptb = TallyClientOffline::client_offline->ptb_kernel_map[func].func;
    auto cu_func_dynamic_ptb = TallyClientOffline::client_offline->dynamic_ptb_kernel_map[func].func;
    auto cu_func_preemptive_ptb = TallyClientOffline::client_offline->preemptive_ptb_kernel_map[func].func;

    auto cu_attr = convert_func_attribute(attr);

    lcuFuncSetAttribute(cu_func, cu_attr, value);
    lcuFuncSetAttribute(cu_func_ptb, cu_attr, value);
    lcuFuncSetAttribute(cu_func_dynamic_ptb, cu_attr, value);
    lcuFuncSetAttribute(cu_func_preemptive_ptb, cu_attr, value);

    return lcudaFuncSetAttribute(func, attr, value);
}

CUresult cuFuncSetAttribute(CUfunction  hfunc, CUfunction_attribute  attrib, int  value)
{
    auto func = TallyClientOffline::client_offline->cu_func_addr_mapping[hfunc];

    auto cu_func = TallyClientOffline::client_offline->original_kernel_map[func].func;
    auto cu_func_ptb = TallyClientOffline::client_offline->ptb_kernel_map[func].func;
    auto cu_func_dynamic_ptb = TallyClientOffline::client_offline->dynamic_ptb_kernel_map[func].func;
    auto cu_func_preemptive_ptb = TallyClientOffline::client_offline->preemptive_ptb_kernel_map[func].func;

    auto err = lcuFuncSetAttribute(cu_func, attrib, value);

    lcuFuncSetAttribute(cu_func_ptb, attrib, value);
    lcuFuncSetAttribute(cu_func_dynamic_ptb, attrib, value);
    lcuFuncSetAttribute(cu_func_preemptive_ptb, attrib, value);

    return err;
}

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
    "cuGetProcAddress_v2",
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
    "cuGetProcAddress_v2",
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
    "cuGetProcAddress",
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
    "cuGraphicsResourceSetMapFlags",
    "cuGetProcAddress"
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

CUresult cuGetProcAddress_v2(const char * symbol, void ** pfn, int  cudaVersion, cuuint64_t  flags, CUdriverProcAddressQueryResult * symbolStatus)
{
	TALLY_LOG("cuGetProcAddress_v2 hooked");

    std::string symbol_str(symbol);
    TALLY_LOG("cuGetProcAddress symbol: " + symbol_str);

    CUresult res = lcuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);

    if (res) {
        TALLY_LOG("cuGetProcAddress failed");
    }

    if (symbol_str == "") {
        // do nothing
        return res;
    }
    else if (symbol_str == "cuGetProcAddress") {
        return res;
    }
    else if (std::find(cuGetProcAddress_funcs.begin(), cuGetProcAddress_funcs.end(), symbol_str) != cuGetProcAddress_funcs.end()) {
        *pfn = dlsym(RTLD_DEFAULT, symbol_str.c_str());
        assert(pfn);
    }
    else if (std::find(cuGetProcAddress_v2funcs.begin(), cuGetProcAddress_v2funcs.end(), symbol_str) != cuGetProcAddress_v2funcs.end()) {
        auto symbol_v2_str = symbol_str + "_v2";
        *pfn = dlsym(RTLD_DEFAULT, symbol_v2_str.c_str());
        assert(pfn);
    }
    else if (std::find(cuGetProcAddress_v3funcs.begin(), cuGetProcAddress_v3funcs.end(), symbol_str) != cuGetProcAddress_v3funcs.end()) {
        auto symbol_v3_str = symbol_str + "_v3";
        *pfn = dlsym(RTLD_DEFAULT, symbol_v3_str.c_str());
        assert(pfn);
    }
    else if (std::find(cuGetProcAddress_direct_call_funcs.begin(), cuGetProcAddress_direct_call_funcs.end(), symbol_str) != cuGetProcAddress_direct_call_funcs.end()) {
        return lcuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    }
    else {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unimplemented cuGetProcAddress_v2 lookup.");
    }

	return res;
}

}

