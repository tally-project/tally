#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <type_traits>

#include <cuda.h>

#include <tally/env.h>
#include <tally/cuda_launch.h>
#include <tally/transform.h>
#include <tally/generated/server.h>

const CudaLaunchConfig CudaLaunchConfig::default_config = get_original_config();

std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config)
{
    os << "CudaLaunchConfig: ";
    if (config.use_original) {
        os << "original";
    } else if (config.use_ptb) {
        os << "PTB: blocks_per_sm: " << config.blocks_per_sm;
    } else if (config.use_dynamic_ptb) {
        os << "Dynamic PTB: blocks_per_sm: " << config.blocks_per_sm;
    } else if (config.use_preemptive_ptb) {
        os << "Preemptive PTB: blocks_per_sm: " << config.blocks_per_sm;
    }
    return os;
}

std::vector<uint32_t> get_candidate_blocks_per_sm(uint32_t threads_per_block, uint32_t num_blocks, uint32_t max_threads_per_sm) {

    std::vector<uint32_t> candiates;

    uint32_t _blocks_per_sm = 1;
    while (true) {

        // One kernel should not take all the thread slots
        if (_blocks_per_sm * threads_per_block > max_threads_per_sm) {
            break;
        }
        
        // There is no point going over the total num of blocks
        // But we will keep the (_blocks_per_sm == 1) case
        if (_blocks_per_sm > 1 && (_blocks_per_sm - 1) * CUDA_NUM_SM > num_blocks) {
            break;
        }

        candiates.push_back(_blocks_per_sm);
        _blocks_per_sm++;
    }

    return candiates;
}

// =================== Used by Profile Scheduler ===========================
// Note that this tries to profile as many configs as possible
// Including ones that can take up all the thread slots
// This should only be used for profiling purposes
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_profile_configs(CudaLaunchCall &launch_call)
{
    uint32_t threads_per_block = launch_call.threads_per_block;
    uint32_t num_blocks = launch_call.num_blocks;

    std::vector<CudaLaunchConfig> configs;

    // PTB configs
    auto candiate_blocks_per_sm = get_candidate_blocks_per_sm(threads_per_block, num_blocks, CUDA_MAX_NUM_THREADS_PER_SM);
    for (auto blocks_per_sm : candiate_blocks_per_sm) {
        auto ptb_config = CudaLaunchConfig::get_ptb_config(blocks_per_sm);
        auto dynamic_ptb_config = CudaLaunchConfig::get_dynamic_ptb_config(blocks_per_sm);
        auto preemptive_ptb_config = CudaLaunchConfig::get_preemptive_ptb_config(blocks_per_sm);

        configs.push_back(ptb_config);
        configs.push_back(dynamic_ptb_config);
        configs.push_back(preemptive_ptb_config);

        // get corresponding kernel slicing config
        uint32_t blocks_per_slice = blocks_per_sm * CUDA_NUM_SM;
        uint32_t num_slices = (num_blocks + blocks_per_slice - 1) / blocks_per_slice;

        if (num_slices > 1) {
            auto sliced_config = CudaLaunchConfig::get_sliced_config(num_slices);
            configs.push_back(sliced_config);
        }
    }
    
    
    return configs;
}

// =================== Used by Workload Agnostic Sharing Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_workload_agnostic_sharing_configs(CudaLaunchCall &launch_call)
{
    uint32_t threads_per_block = launch_call.threads_per_block;
    uint32_t num_blocks = launch_call.num_blocks;

    std::vector<CudaLaunchConfig> configs;

    uint32_t blocks_per_sm = (num_blocks + CUDA_NUM_SM - 1) / CUDA_NUM_SM;
    uint32_t threads_per_sm = threads_per_block * blocks_per_sm;

    // do not need PTB or sliced kernel
    if (threads_per_sm <= SHARING_PTB_MAX_NUM_THREADS_PER_SM) {
        return configs;
    }

    auto candiate_blocks_per_sm = get_candidate_blocks_per_sm(threads_per_block, num_blocks, SHARING_PTB_MAX_NUM_THREADS_PER_SM);
    auto largest_blocks_per_sm = *std::max_element(candiate_blocks_per_sm.begin(), candiate_blocks_per_sm.end());

    // take the largest
    auto ptb_config = CudaLaunchConfig::get_ptb_config(largest_blocks_per_sm);
    auto dynamic_ptb_config = CudaLaunchConfig::get_dynamic_ptb_config(largest_blocks_per_sm);

    auto kernel_name = TallyServer::server->host_func_to_demangled_kernel_name_map[launch_call.func];

    if (containsSubstring(kernel_name, "DeviceSelectSweepKernel") ||
        containsSubstring(kernel_name, "DeviceScanKernel")
    ) {
        // for some reason these kernel hangs under PTB while they work fine for dynamic PTB
        // leave it unhandled for now.
    } else {
        configs.push_back(ptb_config);
    }

    configs.push_back(dynamic_ptb_config);

    // get corresponding kernel slicing config
    uint32_t blocks_per_slice = largest_blocks_per_sm * CUDA_NUM_SM;
    uint32_t num_slices = (num_blocks + blocks_per_slice - 1) / blocks_per_slice;

    if (num_slices > 1) {
        auto sliced_config = CudaLaunchConfig::get_sliced_config(num_slices);
        configs.push_back(sliced_config);
    }

    return configs;
}

// =================== Used by Priority Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_priority_configs(CudaLaunchCall &launch_call)
{
    uint32_t threads_per_block = launch_call.threads_per_block;
    uint32_t num_blocks = launch_call.num_blocks;

    if (num_blocks <= CUDA_NUM_SM) {
        return {};
    }

    std::vector<CudaLaunchConfig> configs;

    auto candiate_blocks_per_sm = get_candidate_blocks_per_sm(threads_per_block, num_blocks, PRIORITY_PTB_MAX_NUM_THREADS_PER_SM);

    // Sort by desc order
    std::sort(candiate_blocks_per_sm.begin(), candiate_blocks_per_sm.end(), std::greater<uint32_t>());
    
    for (auto blocks_per_sm : candiate_blocks_per_sm) {

        // preemptive config
        auto preemptive_config = CudaLaunchConfig::get_preemptive_ptb_config(blocks_per_sm);
        configs.push_back(preemptive_config);

        // get corresponding kernel slicing config
        uint32_t blocks_per_slice = blocks_per_sm * CUDA_NUM_SM;
        uint32_t num_slices = (num_blocks + blocks_per_slice - 1) / blocks_per_slice;

        if (num_slices > 1) {
            auto sliced_config = CudaLaunchConfig::get_sliced_config(num_slices);
            configs.push_back(sliced_config);
        }
    }

    return configs;
}

// return (time, iterations)
CUresult CudaLaunchConfig::repeat_launch(
    const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream,
    float dur_seconds, PTBKernelArgs *ptb_args, uint32_t *curr_idx_arr, SlicedKernelArgs *slice_args,
    float *time_ms, float *iters, int32_t max_count)
{
    float _time_ms;
    CUresult err;

    auto startTime = std::chrono::steady_clock::now();
    uint64_t ckpt_count = 0;
    uint64_t count = 0;
    uint64_t elapsed_ns = 0;

    while (true) {

        cudaStreamSynchronize(stream);

        if (use_dynamic_ptb || use_preemptive_ptb) {
            // Make Sure the previous kernel has finished
            cudaMemsetAsync(ptb_args, 0, sizeof(PTBKernelArgs), stream);
        }

        // Perform your steps here
        err = launch(func, gridDim, blockDim, args, sharedMem, stream, ptb_args, curr_idx_arr);
        count++;
        ckpt_count++;

        auto currentTime = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();

        if ((max_count > 0 && count >= max_count) || ((double) elapsed_ns) / 1e9 >= dur_seconds) {
            cudaStreamSynchronize(stream);
            auto currentTime = std::chrono::steady_clock::now();
            elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
            break;
        }
    }

    if (time_ms) *time_ms = (double)elapsed_ns / 1e6;
    if (iters) *iters = count;

    return err;
}

SlicedKernelArgs get_sliced_kernel_args(dim3 gridDim, uint32_t num_slices)
{
    SlicedKernelArgs slice_args;

    dim3 &sliced_gridDim = slice_args.sliced_gridDim;
    std::vector<dim3> &blockOffset_vec = slice_args.block_offsets;

    dim3 blockOffset(0, 0, 0);

    uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
    num_slices = std::min(total_blocks, num_slices);
    uint32_t blocks_per_slice = (total_blocks + num_slices - 1) / num_slices;

    if (blocks_per_slice <= gridDim.x) {
        sliced_gridDim = dim3(blocks_per_slice, 1, 1);
    } else {
        uint32_t num_blocks_y = (blocks_per_slice + gridDim.x - 1) / gridDim.x;
        if (num_blocks_y <= gridDim.y) {
            sliced_gridDim = dim3(gridDim.x, num_blocks_y, 1);
        } else {
            uint32_t num_blocks_z = (num_blocks_y + gridDim.y - 1) / gridDim.y;
            sliced_gridDim = dim3(gridDim.x, gridDim.y, std::min(num_blocks_z, gridDim.z));
        }
    }

    while (blockOffset.x < gridDim.x && blockOffset.y < gridDim.y && blockOffset.z < gridDim.z) {

        blockOffset_vec.push_back(blockOffset);

        blockOffset.x += sliced_gridDim.x;

        if (blockOffset.x >= gridDim.x) {
            blockOffset.x = 0;
            blockOffset.y += sliced_gridDim.y;

            if (blockOffset.y >= gridDim.y) {
                blockOffset.y = 0;
                blockOffset.z += sliced_gridDim.z;
            }
        }
    }

    return slice_args;
}

// ======== Use the following template to try out a hand-written ptx kernel for correctness ===========
// auto kernel_name = TallyServer::server->host_func_to_demangled_kernel_name_map[func];
// if (kernel_name == "kernel")
// {

//     CUmodule    cudaModule;
//     CUfunction  function;

//     std::ifstream t("kernel.ptx");
//     std::string str((std::istreambuf_iterator<char>(t)),
//                         std::istreambuf_iterator<char>());

//     cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0);
//     cuModuleGetFunction(&function, cudaModule, "kernel");

//     cu_func = function;
// }
// ======================================================================================================

CUresult CudaLaunchConfig::launch(
    const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream,
    PTBKernelArgs *ptb_args, uint32_t *curr_idx_arr, SlicedKernelArgs *slice_args)
{

    if (use_original) {

        auto cu_func = TallyServer::server->original_kernel_map[func].func;
        assert(cu_func);

        if (!cu_func) {
            throw std::runtime_error("Error: cu_func is NULL.");
        }

        auto err = lcuLaunchKernel(cu_func, gridDim.x, gridDim.y, gridDim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, NULL);

        return err;
        
    } else if (use_ptb) {

        auto cu_func = TallyServer::server->ptb_kernel_map[func].func;
        size_t num_args = TallyServer::server->ptb_kernel_map[func].num_args;
        assert(cu_func);
        
        uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
        uint32_t worker_blocks = std::min(total_blocks, CUDA_NUM_SM * blocks_per_sm);
        worker_blocks = std::min(max_worker_blocks, worker_blocks);

        dim3 PTB_grid_dim(worker_blocks);

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 1; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 1] = &gridDim;

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        return err;
        
    } else if (use_dynamic_ptb) {

        assert(ptb_args);
        auto global_idx = &(ptb_args->global_idx);

        auto cu_func = TallyServer::server->dynamic_ptb_kernel_map[func].func;
        size_t num_args = TallyServer::server->dynamic_ptb_kernel_map[func].num_args;
        
        assert(cu_func);

        uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
        uint32_t worker_blocks = std::min(total_blocks, CUDA_NUM_SM * blocks_per_sm);
        worker_blocks = std::min(max_worker_blocks, worker_blocks);

        dim3 PTB_grid_dim(worker_blocks);

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 3; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 3] = &gridDim;
        KernelParams[num_args - 2] = &global_idx;
        KernelParams[num_args - 1] = &curr_idx_arr;

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        return err;

    } else if (use_preemptive_ptb) { 
        assert(ptb_args);
        assert(curr_idx_arr);

        auto global_idx = &(ptb_args->global_idx);
        auto retreat = &(ptb_args->retreat);

        auto cu_func = TallyServer::server->preemptive_ptb_kernel_map[func].func;
        size_t num_args = TallyServer::server->preemptive_ptb_kernel_map[func].num_args;

        assert(cu_func);

        uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
        uint32_t worker_blocks = std::min(total_blocks, CUDA_NUM_SM * blocks_per_sm);
        worker_blocks = std::min(max_worker_blocks, worker_blocks);

        dim3 PTB_grid_dim(worker_blocks);

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 4; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 4] = &gridDim;
        KernelParams[num_args - 3] = &global_idx;
        KernelParams[num_args - 2] = &retreat;
        KernelParams[num_args - 1] = &curr_idx_arr;

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        return err;

    } else if (use_sliced) {
        auto cu_func = TallyServer::server->sliced_kernel_map[func].func;
        auto num_args = TallyServer::server->sliced_kernel_map[func].num_args;

        auto sliced_gridDim = slice_args->sliced_gridDim;
        auto blockOffset_vec = slice_args->block_offsets;

        // launch specific slice if specified
        // otherwise launch all slices together
        if (slice_args->launch_idx >= 0) {
            auto block_offset = blockOffset_vec[slice_args->launch_idx];
            blockOffset_vec = { block_offset };
        }

        CUresult err;
        for (auto &blockOffset : blockOffset_vec) {        

            void *KernelParams[num_args];
            for (size_t i = 0; i < num_args - 2; i++) {
                KernelParams[i] = args[i];
            }
            KernelParams[num_args - 2] = &gridDim;
            KernelParams[num_args - 1] = &blockOffset;

            // This ensure that you won't go over the original grid size
            dim3 curr_grid_dim (
                std::min(gridDim.x - blockOffset.x, sliced_gridDim.x),
                std::min(gridDim.y - blockOffset.y, sliced_gridDim.y),
                std::min(gridDim.z - blockOffset.z, sliced_gridDim.z)
            );

            err = lcuLaunchKernel(cu_func, curr_grid_dim.x, curr_grid_dim.y, curr_grid_dim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);
        }

        return err;
    } else {

        auto kernel_name = TallyServer::server->host_func_to_demangled_kernel_name_map[func];
        std::cout << kernel_name << std::endl;
        throw std::runtime_error("Invalid launch config.");
    }
}