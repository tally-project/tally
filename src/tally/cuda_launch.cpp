#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cassert>
#include <type_traits>

#include <cuda.h>

#include <tally/env.h>
#include <tally/cuda_launch.h>
#include <tally/transform.h>
#include <tally/generated/server.h>

const CudaLaunchConfig CudaLaunchConfig::default_config = CudaLaunchConfig();

std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config)
{
    os << "CudaLaunchConfig: ";
    if (config.use_original) {
        os << "original";
    } else if (config.use_ptb) {
        os << "PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    } else if (config.use_dynamic_ptb) {
        os << "Dynamic PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    } else if (config.use_preemptive_ptb) {
        os << "Preemptive PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    }
    return os;
}

// =================== Used by Profile Scheduler ===========================
// Note that this tries to profile as many configs as possible
// Including ones that can take up all the thread slots
// This should only be used for profiling purposes
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_profile_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks)
{
    std::vector<CudaLaunchConfig> configs;

    configs.push_back(CudaLaunchConfig::default_config);

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(true) {

        // One kernel should not take all the thread slots
        if (_num_blocks_per_sm * threads_per_block > CUDA_MAX_NUM_THREADS_PER_SM) {
            break;
        }
        
        // There is no point going over the total num of blocks
        // But we will keep the (_num_blocks_per_sm == 1) case
        if (_num_blocks_per_sm > 1 && (_num_blocks_per_sm - 1) * CUDA_NUM_SM > num_blocks) {
            break;
        }

        // regular PTB
        CudaLaunchConfig ptb_config(false, true, false, false, _num_blocks_per_sm);

        // dynamic PTB
        CudaLaunchConfig dynamic_ptb_config(false, false, true, false, _num_blocks_per_sm);

        // preemptive PTB
        CudaLaunchConfig preemptive_ptb_config(false, false, false, true, _num_blocks_per_sm);

        configs.push_back(ptb_config);
        configs.push_back(dynamic_ptb_config);
        configs.push_back(preemptive_ptb_config);

        _num_blocks_per_sm++;
    }
    
    return configs;
}

// =================== Used by Workload Agnostic Sharing Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_workload_agnostic_sharing_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks)
{
    std::vector<CudaLaunchConfig> configs;

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(true) {

        // One kernel should not take all the thread slots
        if (_num_blocks_per_sm * threads_per_block > PTB_MAX_NUM_THREADS_PER_SM) {
            break;
        }
        
        // There is no point going over the total num of blocks
        // But we will keep the (_num_blocks_per_sm == 1) case
        if (_num_blocks_per_sm > 1 && (_num_blocks_per_sm - 1) * CUDA_NUM_SM > num_blocks) {
            break;
        }

        // regular PTB
        CudaLaunchConfig ptb_config(false, true, false, false, _num_blocks_per_sm);

        // dynamic PTB
        CudaLaunchConfig dynamic_ptb_config(false, false, true, false, _num_blocks_per_sm);

        // preemptive PTB
        CudaLaunchConfig preemptive_ptb_config(false, false, false, true, _num_blocks_per_sm);

        auto kernel_name = TallyServer::server->host_func_to_demangled_kernel_name_map[launch_call.func];
    
        if (containsSubstring(kernel_name, "DeviceSelectSweepKernel") && _num_blocks_per_sm > 3) {
            // for some reason this kernel hangs indefinitely under PTB with _num_blocks_per_sm > 3
            // while it works fine for dynamic PTB
            // leave it unhandled for now.
        } else {
            configs.push_back(ptb_config);
        }

        configs.push_back(dynamic_ptb_config);
        // configs.push_back(preemptive_ptb_config);

        _num_blocks_per_sm++;
    }
    
    return configs;
}

// =================== Used by Workload Aware Sharing Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_preemptive_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks)
{
    std::vector<CudaLaunchConfig> configs;

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(true) {

        // One kernel should not take all the thread slots
        if (_num_blocks_per_sm * threads_per_block > PTB_MAX_NUM_THREADS_PER_SM) {
            break;
        }
        
        // There is no point going over the total num of blocks
        // But we will keep the (_num_blocks_per_sm == 1) case
        if (_num_blocks_per_sm > 1 && (_num_blocks_per_sm - 1) * CUDA_NUM_SM > num_blocks) {
            break;
        }

        // preemptive PTB
        CudaLaunchConfig preemptive_ptb_config(false, false, false, true, _num_blocks_per_sm);
        configs.push_back(preemptive_ptb_config);

        _num_blocks_per_sm++;
    }
    
    return configs;
}

// =================== Used by Priority Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_priority_preemptive_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks)
{
    std::vector<CudaLaunchConfig> configs;
    std::vector<uint32_t> _num_blocks_per_sm_candiates;

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(true) {

        // One kernel should not take all the thread slots
        if (_num_blocks_per_sm * threads_per_block > PTB_MAX_NUM_THREADS_PER_SM) {
        // if (_num_blocks_per_sm * threads_per_block > CUDA_MAX_NUM_THREADS_PER_SM) {
            break;
        }
        
        // There is no point going over the total num of blocks
        // But we will keep the (_num_blocks_per_sm == 1) case
        if (_num_blocks_per_sm > 1 && (_num_blocks_per_sm - 1) * CUDA_NUM_SM > num_blocks) {
            break;
        }

        _num_blocks_per_sm_candiates.push_back(_num_blocks_per_sm);
        _num_blocks_per_sm++;
    }

    // preemptive PTB
    for (auto _num_blocks_per_sm : _num_blocks_per_sm_candiates) {
        CudaLaunchConfig preemptive_ptb_config(false, false, false, true, _num_blocks_per_sm);
        configs.push_back(preemptive_ptb_config);
    }

    // // regular PTB
    // for (auto _num_blocks_per_sm : _num_blocks_per_sm_candiates) {
    //     CudaLaunchConfig ptb_config(false, true, false, false, _num_blocks_per_sm);
    //     configs.push_back(ptb_config);
    // }

    // // dynamic PTB
    // for (auto _num_blocks_per_sm : _num_blocks_per_sm_candiates) {
    //     CudaLaunchConfig dynamic_ptb_config(false, false, true, false, _num_blocks_per_sm);
    //     configs.push_back(dynamic_ptb_config);
    // }
    
    return configs;
}

// return (time, iterations)
CUresult CudaLaunchConfig::repeat_launch(
    const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream,
    float dur_seconds, PTBArgs *ptb_args, uint32_t *curr_idx_arr, float *time_ms, float *iters, int32_t max_count)
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
            cudaMemsetAsync(ptb_args, 0, sizeof(PTBArgs), stream);
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

CUresult CudaLaunchConfig::launch(
    const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream,
    PTBArgs *ptb_args, uint32_t *curr_idx_arr)
{
    if (use_original) {

        CUfunction cu_func = TallyServer::server->original_kernel_map[func].func;
        assert(cu_func);

        auto err = lcuLaunchKernel(cu_func, gridDim.x, gridDim.y, gridDim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, NULL);

        return err;
    } else if (use_ptb) {

        CUfunction cu_func = TallyServer::server->ptb_kernel_map[func].func;
        size_t num_args = TallyServer::server->ptb_kernel_map[func].num_args;
        assert(cu_func);

        dim3 PTB_grid_dim;
        
        uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Depend on number of PTBs/SM
        if (total_blocks < CUDA_NUM_SM) {
            PTB_grid_dim = dim3(total_blocks);
        } else {
            PTB_grid_dim = dim3(CUDA_NUM_SM * num_blocks_per_sm);
        }

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

        CUfunction cu_func = TallyServer::server->dynamic_ptb_kernel_map[func].func;
        size_t num_args = TallyServer::server->dynamic_ptb_kernel_map[func].num_args;
        
        assert(cu_func);

        dim3 PTB_grid_dim;
        
        uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Depend on number of PTBs/SM
        if (total_blocks < CUDA_NUM_SM) {
            PTB_grid_dim = dim3(total_blocks);
        } else {
            PTB_grid_dim = dim3(CUDA_NUM_SM * num_blocks_per_sm);
        }

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

        CUfunction cu_func = TallyServer::server->preemptive_ptb_kernel_map[func].func;
        size_t num_args = TallyServer::server->preemptive_ptb_kernel_map[func].num_args;

        assert(cu_func);

        dim3 PTB_grid_dim;
        
        uint32_t total_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Depend on number of PTBs/SM
        if (total_blocks < CUDA_NUM_SM) {
            PTB_grid_dim = dim3(total_blocks);
        } else {
            PTB_grid_dim = dim3(CUDA_NUM_SM * num_blocks_per_sm);
        }

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

    } else {
        throw std::runtime_error("Invalid launch config.");
    }
}