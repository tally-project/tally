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

const char *__cubin_data;
size_t __cubin_size;
uint32_t __cubin_uid;

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

    if (!TallyClientOffline::client_offline->curr_idx_arr) {
        cudaMalloc((void **)&(TallyClientOffline::client_offline->global_idx), sizeof(uint32_t));
        cudaMalloc((void **)&(TallyClientOffline::client_offline->curr_idx_arr), sizeof(uint32_t) * CUDA_NUM_SM * 20);
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

    if (config.use_dynamic_ptb) {
        cudaStreamSynchronize(stream);
        cudaMemsetAsync(TallyClientOffline::client_offline->global_idx, 0, sizeof(uint32_t), stream);
    }

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

    auto wp = (__fatBinC_Wrapper_t *) fatCubin;
    auto fbh = (struct fatBinaryHeader *) wp->data;
    __cubin_data = (const char *) wp->data;
    __cubin_size = fbh->headerSize + fbh->fatSize;

    // cache fatbin is not exists
    if (!TallyCache::cache->cubin_cache.contains(__cubin_data, __cubin_size)) {
        cache_cubin_data(__cubin_data, __cubin_size);
    }

    __cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(__cubin_data, __cubin_size);
    cubin_register_queue.push_back(__cubin_uid);

    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    auto kernel_name = std::string(deviceName);
    auto demangled_kernel_name = demangleFunc(kernel_name);

    TallyClientOffline::client_offline->host_func_to_demangled_kernel_name_map[hostFun] = demangled_kernel_name;
    TallyClientOffline::client_offline->host_func_to_cubin_uid_map[hostFun] = __cubin_uid;

    TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[__cubin_uid][kernel_name] = hostFun;
    TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[__cubin_uid][demangled_kernel_name] = hostFun;

    TallyClientOffline::client_offline->demangled_kernel_name_and_cubin_uid_to_host_func_map[
        std::make_pair(demangled_kernel_name, __cubin_uid)
    ] = hostFun;

    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
    l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

CUresult cuModuleLoadData(CUmodule * module, const void * image)
{
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
    
        TallyClientOffline::client_offline->demangled_kernel_name_and_cubin_uid_to_host_func_map[
            std::make_pair(kernel_name, cubin_uid)
        ] = hostFun;

        TallyClientOffline::client_offline->register_measurements();

        TallyClientOffline::client_offline->register_ptx_transform(cubin_data, cubin_size);
    }

    TallyClientOffline::client_offline->cu_func_addr_mapping[*hfunc] = TallyClientOffline::client_offline->cubin_to_kernel_name_to_host_func_map[cubin_uid][kernel_name];

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

}

