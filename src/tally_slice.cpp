
#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
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
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>
#include <thread>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <regex>

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include <tally/util.h>
#include <tally/def.h>
#include <tally/const.h>
#include <tally/kernel_slice.h>
#include <tally/cuda_api.h>

class CudaGraphCall {

public:
    const void *_host_func;
    std::vector<void *> _args;
    dim3 _gridDim;
    dim3 _blockDim;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool instantiated = false;

    CudaGraphCall(const void * host_func, void **args, size_t nargs, dim3 gridDim, dim3 blockDim)
    {
        _host_func = host_func;
        for (size_t i = 0; i < nargs; i++) {
            _args.push_back(args[i]);
        }
        _gridDim = gridDim;
        _blockDim = blockDim;
    }

    bool equals(const void * host_func, void **args, size_t nargs, dim3 gridDim, dim3 blockDim) {
        if (host_func != _host_func || _args.size() != nargs ||
            gridDim.x != _gridDim.x || gridDim.y != _gridDim.y ||  gridDim.z != _gridDim.z || 
            blockDim.x != _blockDim.x || blockDim.y != _blockDim.y || blockDim.z != _blockDim.z)
        {
            return false;
        }

        for (size_t i = 0; i < nargs; i++) {
            if (args[i] != _args[i]) {
                return false;
            }
        }

        return true;
    }
};


class Preload {

public:

    std::map<std::string, const void *> kernel_name_to_host_func_map;
    std::map<const void *, std::string> host_func_to_kernel_name_map;
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> kernel_map;
    std::unordered_map<const void *, bool> kernel_slice_map;
    std::vector<CudaGraphCall*> cuda_graph_vec;
    cudaStream_t stream;

    std::vector<std::pair<std::string, std::string>> sliced_ptx_fatbin_strs;
    bool kernels_registered = false;

    void register_kernels()
    {
        lcudaStreamCreate(&stream);

        kernel_map = register_kernels_from_ptx_fatbin(sliced_ptx_fatbin_strs, kernel_name_to_host_func_map);
        kernels_registered = true;
    }

    Preload(){}
    ~Preload(){}
};

Preload tracer;

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    assert(lcudaLaunchKernel);
    assert(lcuLaunchKernel);

    cudaError_t err;

    if (!tracer.kernels_registered) {
        tracer.register_kernels();
    }

    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t num_threads = gridDim.x * gridDim.y * gridDim.z * threads_per_block;

    bool run_profile = false;
    float prev_time_ms = 0.0f;
    float sliced_time_ms = 0.0f;
    cudaEvent_t _start, _stop;

    if (tracer.kernel_slice_map.find(func) == tracer.kernel_slice_map.end()) {
        run_profile = true;
    }

    // Only slice if num_threads > THREADS_PER_SLICE
    // And either profiling or have decided to use sliced version
    if (num_threads > THREADS_PER_SLICE && (run_profile || tracer.kernel_slice_map[func])) {

        auto cu_func = tracer.kernel_map[func].first;
        auto num_args = tracer.kernel_map[func].second;

        assert(cu_func);

        CudaGraphCall *cuda_graph_call = nullptr;

        if (USE_CUDA_GRAPH) {

            // Try to use cuda graph first 
            for (auto *call : tracer.cuda_graph_vec) {
                if (call->equals(func, args, num_args - 1, gridDim, blockDim)) {
                    cuda_graph_call = call;
                    break;
                }
            }

            if (!cuda_graph_call) {
                // Otherwise, construct one
                cuda_graph_call = new CudaGraphCall(func, args, num_args - 1, gridDim, blockDim);
                tracer.cuda_graph_vec.push_back(cuda_graph_call);
            }
        }

        dim3 new_grid_dim;
        dim3 blockOffset(0, 0, 0);

        uint32_t num_blocks = (THREADS_PER_SLICE + threads_per_block - 1) / threads_per_block;
        if (num_blocks <= gridDim.x) {
            new_grid_dim = dim3(num_blocks, 1, 1);
        } else {
            uint32_t num_blocks_y = (num_blocks + gridDim.x - 1) / gridDim.x;
            if (num_blocks_y <= gridDim.y) {
                new_grid_dim = dim3(gridDim.x, num_blocks_y, 1);
            } else {
                uint32_t num_blocks_z = (num_blocks_y + gridDim.y - 1) / gridDim.y;
                new_grid_dim = dim3(gridDim.x, gridDim.y, std::min(num_blocks_z, gridDim.z));
            }
        }

        if (USE_CUDA_GRAPH) {
            cudaStreamBeginCapture(tracer.stream, cudaStreamCaptureModeGlobal);
        } else {
            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaDeviceSynchronize();

                cudaEventRecord(_start);
            }
        }

        CUresult res;
        while (blockOffset.x < gridDim.x && blockOffset.y < gridDim.y && blockOffset.z < gridDim.z) {

            void *KernelParams[num_args];
            for (size_t i = 0; i < num_args - 1; i++) {
                KernelParams[i] = args[i];
            }
            KernelParams[num_args - 1] = &blockOffset;

            // This ensure that you won't go over the original grid size
            dim3 curr_grid_dim (
                std::min(gridDim.x - blockOffset.x, new_grid_dim.x),
                std::min(gridDim.y - blockOffset.y, new_grid_dim.y),
                std::min(gridDim.z - blockOffset.z, new_grid_dim.z)
            );

            res = lcuLaunchKernel(cu_func, curr_grid_dim.x, curr_grid_dim.y, curr_grid_dim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, tracer.stream, KernelParams, NULL);

            if (res != CUDA_SUCCESS) {
                std::cerr << "Encountering res != CUDA_SUCCESS" << std::endl;
                return cudaErrorInvalidValue;
            }

            blockOffset.x += new_grid_dim.x;

            if (blockOffset.x >= gridDim.x) {
                blockOffset.x = 0;
                blockOffset.y += new_grid_dim.y;

                if (blockOffset.y >= gridDim.y) {
                    blockOffset.y = 0;
                    blockOffset.z += new_grid_dim.z;
                }
            }
        }

        if (USE_CUDA_GRAPH) {
            cudaStreamEndCapture(tracer.stream, &(cuda_graph_call->graph));

            if (!cuda_graph_call->instantiated) {
                cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                cuda_graph_call->instantiated = true;
            } else {
                // graph already exists; try to apply changes

                cudaGraphExecUpdateResult update;

                if (cudaGraphExecUpdate(cuda_graph_call->instance, cuda_graph_call->graph, NULL, &update) != cudaSuccess) 
                {
                    cudaGraphExecDestroy(cuda_graph_call->instance);
                    cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                    std::cout << "update fail" << std::endl;
                }
            }

            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaDeviceSynchronize();

                cudaEventRecord(_start);
            }

            err = cudaGraphLaunch(cuda_graph_call->instance, stream);
        } else {
            err = cudaSuccess;
        }

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&sliced_time_ms, _start, _stop);
        }

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaDeviceSynchronize();

            cudaEventRecord(_start);
        
            err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&prev_time_ms, _start, _stop);

            // Does not invoke the sliced version if overhead is too large.
            // std::cout << "prev_time_ms: " << prev_time_ms << " sliced_time_ms: " << sliced_time_ms;
            if (sliced_time_ms > KERNEL_SLICE_THRESHOLD * prev_time_ms) {
                tracer.kernel_slice_map[func] = false;
            } else {
                tracer.kernel_slice_map[func] = true;
            }}


    } else {
        err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    return err;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    tracer.kernel_name_to_host_func_map[std::string(deviceFun)] = hostFun;
    tracer.host_func_to_kernel_name_map[hostFun] = demangleFunc(std::string(deviceFun));

    assert(l__cudaRegisterFunction);
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {
    
    auto *wp = (__fatBinC_Wrapper_t *) fatCubin;
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;

    // std::cout << "Processing cubin with size: " << fatCubin_data_size_bytes << std::endl;
    auto sliced_ptx_fatbin_strs = cubin_cache.get_sliced_ptx_fatbin_strs((const char *)wp->data, fatCubin_data_size_bytes);

    tracer.sliced_ptx_fatbin_strs.insert(
        tracer.sliced_ptx_fatbin_strs.end(),
        std::make_move_iterator(sliced_ptx_fatbin_strs.begin()),
        std::make_move_iterator(sliced_ptx_fatbin_strs.end())
    );

    assert(l__cudaRegisterFatBinary);
    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
	assert(l__cudaRegisterFatBinaryEnd);
	l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

}