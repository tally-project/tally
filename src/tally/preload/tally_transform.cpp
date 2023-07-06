
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

#include <tally/util.h>
#include <tally/msg_struct.h>
#include <tally/env.h>
#include <tally/transform.h>
#include <tally/daemon.h>
#include <tally/cache.h>
#include <tally/generated/cuda_api.h>

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    if (!TallyDaemon::daemon->kernels_registered) {
        TallyDaemon::daemon->register_kernels();
        TallyDaemon::daemon->register_measurements();
        // std::cout << "Finished registering kernels" << std::endl;
    }

    if (PROFILE_KERNEL_TO_KERNEL_PERF && PROFILE_WARMED_UP) {
        CudaLaunchConfig::profile_kernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    auto num_args = TallyDaemon::daemon->sliced_kernel_map[func].second;
    CudaLaunchCall launch_call(func, gridDim, blockDim);

    cudaError_t err;

    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total_threads = gridDim.x * gridDim.y * gridDim.z * threads_per_block;

    if (total_threads > TRANSFORM_THREADS_THRESHOLD) {

        CudaLaunchConfig config;

        if (TallyDaemon::daemon->has_launch_config(launch_call)) {
            config = TallyDaemon::daemon->get_launch_config(launch_call);
        } else {
            config = CudaLaunchConfig::tune(func, gridDim, blockDim, args, sharedMem, stream);
            assert(TallyDaemon::daemon->has_launch_config(launch_call));
        }

        CHECK_CUDA_ERROR(config.launch(func, gridDim, blockDim, args, sharedMem, stream));

    } else {
        CHECK_CUDA_ERROR(lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
    }

    return err;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    std::string mangled_kernel_name = std::string(deviceFun);
    std::string demangled_kernel_name = demangleFunc(mangled_kernel_name);
    TallyDaemon::daemon->mangled_kernel_name_to_host_func_map[mangled_kernel_name] = hostFun;
    TallyDaemon::daemon->demangled_kernel_name_to_host_func_map[demangled_kernel_name] = hostFun;
    TallyDaemon::daemon->host_func_to_demangled_kernel_name_map[hostFun] = demangled_kernel_name;

    assert(l__cudaRegisterFunction);
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {
    
    auto *wp = (__fatBinC_Wrapper_t *) fatCubin;
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;

    TallyDaemon::daemon->register_fat_binary((const char *)wp->data, fatCubin_data_size_bytes);

    assert(l__cudaRegisterFatBinary);
    return l__cudaRegisterFatBinary(fatCubin);
}

void __cudaRegisterFatBinaryEnd(void ** fatCubinHandle)
{
	assert(l__cudaRegisterFatBinaryEnd);
	l__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

}