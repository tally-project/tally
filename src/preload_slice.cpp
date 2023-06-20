
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

class Preload {

public:

    std::map<std::string, const void *> kernel_name_to_host_func_map;
    std::map<const void *, std::string> host_func_to_kernel_name_map;
    std::map<const void *, std::pair<CUfunction, uint32_t>> kernel_map;
    std::vector<std::pair<std::string, std::string>> sliced_ptx_fatbin_strs;
    bool kernels_registered = false;

    void register_kernels()
    {
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

    std::cout << tracer.host_func_to_kernel_name_map[func] << std::endl;

    if (!tracer.kernels_registered) {
        tracer.register_kernels();
    }

    auto cu_func = tracer.kernel_map[func].first;
    auto num_args = tracer.kernel_map[func].second;
    
    assert(cu_func);

    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t num_threads = gridDim.x * gridDim.y * gridDim.z * threads_per_block;

    if (num_threads > THREADS_PER_SLICE) {

        dim3 new_grid_dim;

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

        std::cout << "gridDim: (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
        std::cout << "new_grid_dim: (" << new_grid_dim.x << ", " << new_grid_dim.y << ", " << new_grid_dim.z << ")" << std::endl;

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
    } else {
        return lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }
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

    std::cout << "Processing cubin with size: " << fatCubin_data_size_bytes << std::endl;
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