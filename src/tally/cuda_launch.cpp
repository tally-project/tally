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
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_profile_configs(uint32_t threads_per_block, uint32_t num_blocks)
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
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_workload_agnostic_sharing_configs(uint32_t threads_per_block, uint32_t num_blocks)
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

        configs.push_back(ptb_config);
        // configs.push_back(dynamic_ptb_config);
        // configs.push_back(preemptive_ptb_config);

        _num_blocks_per_sm++;
    }
    
    return configs;
}

// =================== Used by Workload Aware Sharing Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_preemptive_configs(uint32_t threads_per_block, uint32_t num_blocks)
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


// Instantiate template
template
CUresult CudaLaunchConfig::repeat_launch<const void *>(const void *, dim3, dim3, void **, size_t, cudaStream_t, float, uint32_t *, bool *, float *, float *, int32_t);

template
CUresult CudaLaunchConfig::repeat_launch<CUfunction>(CUfunction, dim3, dim3, void **, size_t, cudaStream_t, float, uint32_t *, bool *, float *, float *, int32_t);

// return (time, iterations)
template <typename T>
CUresult CudaLaunchConfig::repeat_launch(
    T func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream,
    float dur_seconds, uint32_t *global_idx, bool *retreat, float *time_ms, float *iters, int32_t max_count)
{
    float _time_ms;
    CUresult err;

    if (use_dynamic_ptb || use_preemptive_ptb) {
        // Make Sure the previous kernel has finished
        cudaStreamSynchronize(stream);
        cudaMemsetAsync(retreat, 0, sizeof(bool), stream);
        cudaMemsetAsync(global_idx, 0, sizeof(uint32_t), stream);
    }

    // get a rough estimate of the kernel duration
    err = launch(func, gridDim, blockDim, args, sharedMem, stream, global_idx, retreat, true, &_time_ms);

    uint64_t sync_interval = std::max((uint64_t)((dur_seconds * 1000.) / _time_ms) / 100, 1ul);

    auto startTime = std::chrono::steady_clock::now();
    uint64_t ckpt_count = 0;
    uint64_t count = 0;
    uint64_t elapsed_ns = 0;

    while (true) {

        if (use_dynamic_ptb || use_preemptive_ptb) {
            // Make Sure the previous kernel has finished
            cudaStreamSynchronize(stream);
            cudaMemsetAsync(retreat, 0, sizeof(bool), stream);
            cudaMemsetAsync(global_idx, 0, sizeof(uint32_t), stream);
        }

        // Perform your steps here
        err = launch(func, gridDim, blockDim, args, sharedMem, stream, global_idx, retreat);
        count++;
        ckpt_count++;

        // Avoid launching too many kernels
        if (ckpt_count == sync_interval) {
            cudaStreamSynchronize(stream);
            ckpt_count = 0;
        }

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

// Instantiate template
template
CUresult CudaLaunchConfig::launch<const void *>(const void *, dim3, dim3, void **, size_t, cudaStream_t, uint32_t *, bool *, bool, float *);

template
CUresult CudaLaunchConfig::launch<CUfunction>(CUfunction, dim3, dim3, void **, size_t, cudaStream_t, uint32_t *, bool *, bool, float *);

void checkCudaErrors(CUresult err) {
    assert(err == CUDA_SUCCESS);
}

template <typename T>
CUresult CudaLaunchConfig::launch(
    T func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream,
    uint32_t *global_idx, bool *retreat, bool run_profile, float *elapsed_time_ms)
{
    cudaEvent_t _start, _stop;

    if (use_original) {

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaStreamSynchronize(stream);

            cudaEventRecord(_start);
        }

        CUfunction cu_func;

        if constexpr (std::is_same<T, const void *>::value) {
            cu_func = TallyServer::server->original_kernel_map[func].func;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = func;
        } else {
            throw std::runtime_error("Unsupported typename");
        }

        assert(cu_func);

        auto err = lcuLaunchKernel(cu_func, gridDim.x, gridDim.y, gridDim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, NULL);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        return err;
    } else if (use_ptb) {

        CUfunction cu_func;
        size_t num_args;

        if (std::is_same<T, const void *>::value) {
            cu_func = TallyServer::server->ptb_kernel_map[func].func;
            num_args = TallyServer::server->ptb_kernel_map[func].num_args;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = TallyServer::server->jit_ptb_kernel_map[func];
            num_args = TallyServer::server->_jit_kernel_addr_to_args[func].size() + 1;
        } else {
            throw std::runtime_error("Unsupported typename");
        }

        assert(cu_func);

        dim3 PTB_grid_dim;
        
        uint32_t total_blocks = blockDim.x * blockDim.y * blockDim.z;
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

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaStreamSynchronize(stream);

            cudaEventRecord(_start);
        }

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        return err;
        
    } else if (use_dynamic_ptb) {

        assert(global_idx);
        
        CUfunction cu_func;
        size_t num_args;

        if (std::is_same<T, const void *>::value) {
            cu_func = TallyServer::server->dynamic_ptb_kernel_map[func].func;
            num_args = TallyServer::server->dynamic_ptb_kernel_map[func].num_args;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = TallyServer::server->jit_dynamic_ptb_kernel_map[func];
            num_args = TallyServer::server->_jit_kernel_addr_to_args[func].size() + 2;
        } else {
            throw std::runtime_error("Unsupported typename");
        }
        
        assert(cu_func);

        dim3 PTB_grid_dim;
        
        uint32_t total_blocks = blockDim.x * blockDim.y * blockDim.z;
        // Depend on number of PTBs/SM
        if (total_blocks < CUDA_NUM_SM) {
            PTB_grid_dim = dim3(total_blocks);
        } else {
            PTB_grid_dim = dim3(CUDA_NUM_SM * num_blocks_per_sm);
        }

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 2; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 2] = &gridDim;
        KernelParams[num_args - 1] = &global_idx;

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaStreamSynchronize(stream);

            cudaEventRecord(_start);
        }

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        return err;

    } else if (use_preemptive_ptb) { 
        assert(global_idx);
        assert(retreat);
        
        CUfunction cu_func;
        size_t num_args;

        if (std::is_same<T, const void *>::value) {
            cu_func = TallyServer::server->preemptive_ptb_kernel_map[func].func;
            num_args = TallyServer::server->preemptive_ptb_kernel_map[func].num_args;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = TallyServer::server->jit_preemptive_ptb_kernel_map[func];
            num_args = TallyServer::server->_jit_kernel_addr_to_args[func].size() + 3;
        } else {
            throw std::runtime_error("Unsupported typename");
        }
        
        assert(cu_func);

        dim3 PTB_grid_dim;
        
        uint32_t total_blocks = blockDim.x * blockDim.y * blockDim.z;
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
        KernelParams[num_args - 1] = &retreat;

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaStreamSynchronize(stream);

            cudaEventRecord(_start);
        }

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        return err;

    } else {
        throw std::runtime_error("Invalid launch config.");
    }
}