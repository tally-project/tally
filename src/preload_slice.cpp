
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

#include <cuda_runtime.h>
#include <cuda.h>
#include <fatbinary_section.h>

#include <tally/util.h>
#include <tally/def.h>
#include <tally/const.h>
#include <tally/kernel_slice.h>
#include <tally/cuda_api.h>

class Preload {

public:

    std::map<std::string, const void *> kernel_name_map;
    std::map<const void *, std::pair<CUfunction, uint32_t>> kernel_map;
    std::vector<std::string> sliced_ptx_files;
    bool registered = false;

    void register_sliced_kernels()
    {
        assert(lcuModuleLoadDataEx);
        assert(lcuModuleGetFunction);

        for (auto &sliced_ptx_file : sliced_ptx_files) {
            auto ptx_kernel_map = register_ptx(sliced_ptx_file);

            for (auto &pair : ptx_kernel_map) {
                auto host_func = kernel_name_map[pair.first];
                kernel_map[host_func] = pair.second;
            }
        }

        registered = true;
    }

    Preload(){
        cuda_handle = dlopen(libcuda_path, RTLD_LAZY);
        assert(cuda_handle);
    }

    ~Preload(){}
};

Preload tracer;

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    assert(lcudaLaunchKernel);
    assert(lcuLaunchKernel);

    if (!tracer.registered) {
        tracer.register_sliced_kernels();
    }

    auto cu_func = tracer.kernel_map[func].first;
    auto num_args = tracer.kernel_map[func].second;
    assert(cu_func);

    dim3 new_grid_dim(4, 1, 1);
    dim3 blockOffset(0, 0, 0);

    CUresult res;

    while (blockOffset.x < gridDim.x && blockOffset.y < gridDim.y && blockOffset.z < gridDim.z) {

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 1; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 1] = &blockOffset;

        res = lcuLaunchKernel(cu_func, new_grid_dim.x, new_grid_dim.y, new_grid_dim.z,
                        blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (res != CUDA_SUCCESS) {
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

    return cudaSuccess;
    // return lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    tracer.kernel_name_map[std::string(deviceFun)] = hostFun;

    assert(l__cudaRegisterFunction);
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {
    auto *wp = (__fatBinC_Wrapper_t *) fatCubin;
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;

    auto file_name = write_cubin_to_file(reinterpret_cast<const char*>(wp->data), fatCubin_data_size_bytes);
    auto ptx_file_names = gen_ptx_from_cubin(file_name);
    
    for (const auto& ptx_file_name : ptx_file_names) {

        auto sliced_ptx_file_name = "sliced_" + ptx_file_name;
        gen_sliced_ptx(ptx_file_name, sliced_ptx_file_name);
        tracer.sliced_ptx_files.push_back(sliced_ptx_file_name);
    }

    assert(l__cudaRegisterFatBinary);
    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
	assert(l__cudaRegisterFatBinaryEnd);
	l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

}