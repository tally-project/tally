
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
#include <tally/generated/cuda_api.h>

extern "C" { 

cudaError_t cudaLaunchKernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    if (!Transform::tracer->kernels_registered) {
        Transform::tracer->register_kernels();
        std::cout << "Finished registering kernels" << std::endl;
    }

    auto num_args = Transform::tracer->sliced_kernel_map[func].second;
    CudaLaunchCall launch_call(func, gridDim, blockDim);

    if (PROFILE_KERNEL_TO_KERNEL_PERF && PROFILE_WARMED_UP) {

        // Only care about those sliced/PTB
        if (Transform::tracer->kernel_baseline_performance.find(launch_call) != Transform::tracer->kernel_baseline_performance.end()) {

            // Use this to identify the idx of the kernel launch
            auto curr_kernel_idx = Transform::tracer->curr_kernel_idx;

            if (curr_kernel_idx == PROFILE_KERNEL_IDX) {

                bool use_original = PROFILE_USE_ORIGINAL;
                bool use_sliced = PROFILE_USE_SLICED;
                bool use_ptb = PROFILE_USE_PTB;
                bool use_cuda_graph = PROFILE_USE_CUDA_GRAPH;
                uint32_t threads_per_slice = PROFILE_THREADS_PER_SLICE;
                uint32_t num_blocks_per_sm = PROFILE_NUM_BLOCKS_PER_SM;

                LaunchConfig profile_config(use_original, use_sliced, use_ptb, use_cuda_graph, threads_per_slice, num_blocks_per_sm);
                std::cout << "Profiling config: " << profile_config << std::endl;
                auto time_iters = profile_config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, PROFILE_DURATION_SECONDS);

                std::cout << "Kernel: " << Transform::tracer->host_func_to_kernel_name_map[func] << std::endl;
                std::cout << "\tblockDim: " << blockDim.x << " " << blockDim.y << " " << blockDim.z << std::endl;
                std::cout << "\tgridDim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
                float baseline_tp = 1 / Transform::tracer->kernel_baseline_performance[launch_call];
                float new_tp = time_iters.second / time_iters.first;

                std::cout << "Time: " << time_iters.first << std::endl;
                std::cout << "Iters: " << time_iters.second << std::endl;
                std::cout << "baseline_tp: " << baseline_tp << std::endl;
                std::cout << "new_tp: " << new_tp << std::endl;

                std::cout << "Normalized throughput: " << new_tp / baseline_tp << std::endl;

                exit(0);

            } else {
                curr_kernel_idx++;
            }
        } else {
            std::cout << "Baseline performance does not exist, skipping.." << std::endl;
        }
    }

    cudaError_t err;

    uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    uint32_t total_threads = gridDim.x * gridDim.y * gridDim.z * threads_per_block;

    if (total_threads > TRANSFORM_THREADS_THRESHOLD) {

        if (Transform::tracer->kernel_profile_map.find(launch_call) == Transform::tracer->kernel_profile_map.end()) {
            Transform::tracer->kernel_profile_map[launch_call] = LaunchConfig::tune(func, gridDim, blockDim, args, sharedMem, stream);
        }

        auto &config = Transform::tracer->kernel_profile_map[launch_call];
        CHECK_CUDA_ERROR(config.launch(func, gridDim, blockDim, args, sharedMem, stream));

    } else {
        CHECK_CUDA_ERROR(lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
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