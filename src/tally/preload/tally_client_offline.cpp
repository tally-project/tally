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

extern "C" { 

const char *cubin_data;
size_t cubin_size;

// Store cubin uid
std::vector<uint32_t> cubin_register_queue;

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    if (cubin_register_queue.size() > 0) {

        for (auto cubin_uid : cubin_register_queue) {
            auto cubin_data = TallyCache::cache->cubin_cache.get_cubin_data_str_ptr_from_cubin_uid(cubin_uid);
            auto cubin_size = TallyCache::cache->cubin_cache.get_cubin_size_from_cubin_uid(cubin_uid);
            TallyClientOffline::client_offline->register_ptx_transform(cubin_data, cubin_size);
        }

        cubin_register_queue.clear();
    }

    auto err = TallyClientOffline::client_offline->launch_kernel(CudaLaunchConfig::default_config, func, gridDim, blockDim, args, sharedMem, stream);
    if (!err) {
        return cudaSuccess;
    } else {
        return cudaErrorInvalidValue;
    }
}

CUresult cuLaunchKernel(CUfunction  f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream  hStream, void ** kernelParams, void ** extra)
{
    // std::cout << "cuLaunchKernel" << std::endl;

    auto func = TallyClientOffline::client_offline->cu_func_addr_mapping[f];

    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    dim3 blockDim(blockDimX, blockDimY, blockDimZ);

    auto err = TallyClientOffline::client_offline->launch_kernel(CudaLaunchConfig::default_config, func, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);

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

    TallyClientOffline::client_offline->host_func_to_demangled_kernel_name_map[hostFun] = kernel_name;
    TallyClientOffline::client_offline->host_func_to_cubin_uid_map[hostFun] = cubin_uid;

    TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name] = hostFun;
    TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid][demangled_kernel_name] = hostFun;

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

    TallyClientOffline::client_offline->jit_module_to_cubin_map[*module] = std::make_pair<const char *, size_t>(
        std::move(cached_cubin_data),
        std::move(cached_cubin_size)
    );

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
    }

    TallyClientOffline::client_offline->register_ptx_transform(cubin_data, cubin_size);

    return err;
}

}
