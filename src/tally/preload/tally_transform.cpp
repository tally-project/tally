
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
#include <tally/const.h>
#include <tally/transform.h>
#include <tally/generated/cuda_api.h>

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    // std::cout << "cudaLaunchKernel" << std::endl;

    cudaError_t err;

    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total_threads = gridDim.x * gridDim.y * gridDim.z * threads_per_block;

    // std::cout << total_threads << std::endl;

    if (total_threads > TRANSFORM_THREADS_THRESHOLD) {

        if (!Transform::tracer->kernels_registered) {
            Transform::tracer->register_kernels();
        }

        if (Transform::tracer->kernel_profile_map.find(func) == Transform::tracer->kernel_profile_map.end()) {
            Transform::tracer->kernel_profile_map[func] = LaunchConfig::tune(func, gridDim, blockDim, args, sharedMem, stream);
        }

        auto &config = Transform::tracer->kernel_profile_map[func];
        err = config.launch(func, gridDim, blockDim, args, sharedMem, stream);

    } else {
        auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    return err;
}

void __cudaRegisterFunction(void ** fatCubinHandle, const char * hostFun, char * deviceFun, const char * deviceName, int  thread_limit, uint3 * tid, uint3 * bid, dim3 * bDim, dim3 * gDim, int * wSize)
{
    Transform::tracer->kernel_name_to_host_func_map[std::string(deviceFun)] = hostFun;
    Transform::tracer->host_func_to_kernel_name_map[hostFun] = demangleFunc(std::string(deviceFun));

    assert(l__cudaRegisterFunction);
    return l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void** __cudaRegisterFatBinary( void *fatCubin ) {
    
    // auto ptb_str = gen_ptb_ptx("./before.ptx");
    // auto sliced_str = gen_sliced_ptx("./before.ptx");

    // std::cout << "ptb_str:" << std::endl;
    // std::cout << ptb_str << std::endl;

    // std::cout << "sliced_str:" << std::endl;
    // std::cout << sliced_str << std::endl;

    // exit(1);

    auto *wp = (__fatBinC_Wrapper_t *) fatCubin;
    struct fatBinaryHeader *fbh = (struct fatBinaryHeader *) wp->data;
    size_t fatCubin_data_size_bytes = fbh->headerSize + fbh->fatSize;

    // std::cout << "Processing cubin with size: " << fatCubin_data_size_bytes << std::endl;
    auto ptx_fatbin_strs = CubinCache::cache->get_transform_ptx_fatbin_strs((const char *)wp->data, fatCubin_data_size_bytes);

    Transform::tracer->sliced_ptx_fatbin_strs.insert(
        Transform::tracer->sliced_ptx_fatbin_strs.end(),
        std::make_move_iterator(ptx_fatbin_strs.first.begin()),
        std::make_move_iterator(ptx_fatbin_strs.first.end())
    );

    Transform::tracer->ptb_ptx_fatbin_strs.insert(
        Transform::tracer->ptb_ptx_fatbin_strs.end(),
        std::make_move_iterator(ptx_fatbin_strs.second.begin()),
        std::make_move_iterator(ptx_fatbin_strs.second.end())
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