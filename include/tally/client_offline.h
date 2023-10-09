
#ifndef TALLY_OFFLINE_H
#define TALLY_OFFLINE_H

#include <string>
#include <map>
#include <iostream>
#include <functional>
#include <memory>
#include <atomic>
#include <cfloat>
#include <utility>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <nvrtc.h>
#include <cublasLt.h>

#include <tally/cache.h>
#include <tally/log.h>
#include <tally/cuda_launch.h>
#include <tally/msg_struct.h>
#include <tally/cuda_util.h>
#include <tally/cache_struct.h>

class TallyClientOffline {

public:
    static TallyClientOffline *client_offline;

    std::unordered_map<const void *, std::string> host_func_to_demangled_kernel_name_map;
    std::unordered_map<const void *, uint32_t> host_func_to_cubin_uid_map;

    // Map CUfunction to host func, similar to _kernel_client_addr_mapping
	std::unordered_map<CUfunction, const void *> cu_func_addr_mapping;

    std::map<std::pair<std::string, uint32_t>, const void *> demangled_kernel_name_and_cubin_uid_to_host_func_map;

    std::unordered_map<CUmodule, std::pair<const char *, size_t>> jit_module_to_cubin_map;

    std::unordered_map<const void *, WrappedCUfunction> original_kernel_map;
    std::unordered_map<const void *, WrappedCUfunction> ptb_kernel_map;
	std::unordered_map<const void *, WrappedCUfunction> dynamic_ptb_kernel_map;
	std::unordered_map<const void *, WrappedCUfunction> preemptive_ptb_kernel_map;

    std::unordered_map<uint32_t, std::unordered_map<std::string, const void *>> cubin_to_kernel_name_to_host_func_map;

    // Performance cache to use at runtime
	std::unordered_map<CudaLaunchCallConfig, CudaLaunchCallConfigResult> single_kernel_perf_map;
	std::unordered_map<CudaLaunchCall, CudaLaunchCallConfigResult> single_kernel_best_config_map;

	// Set and Get performance cache
	CudaLaunchCallConfigResult get_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, bool *found);
	void set_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, CudaLaunchMetadata meta_data, float norm_speed, float latency, uint32_t iters);

	CudaLaunchCallConfigResult get_single_kernel_best_config(CudaLaunchCall &launch_call, bool *found);
	void set_single_kernel_best_config(CudaLaunchCall &launch_call, CudaLaunchCallConfigResult &best_config);

	// Utility functions for measurement data
	CudaLaunchCall convert_key_to_call(CudaLaunchKey key);
	CudaLaunchKey convert_call_to_key(CudaLaunchCall call);

	CudaLaunchCallConfig convert_key_config_to_call_config(CudaLaunchKeyConfig key_config);
	CudaLaunchKeyConfig convert_call_config_to_key_config(CudaLaunchCallConfig call_config);

    void save_performance_cache();

    void register_ptx_transform(const char* cubin_data, size_t cubin_size)
    {
        using KERNEL_NAME_MAP_TYPE = std::unordered_map<std::string, const void *>;
        using KERNEL_MAP_TYPE = std::unordered_map<const void*, WrappedCUfunction>;

        auto original_data = TallyCache::cache->cubin_cache.get_original_data(cubin_data, cubin_size);
        auto ptb_data = TallyCache::cache->cubin_cache.get_ptb_data(cubin_data, cubin_size);
        auto dynamic_ptb_data = TallyCache::cache->cubin_cache.get_dynamic_ptb_data(cubin_data, cubin_size);
        auto preemptive_ptb_data = TallyCache::cache->cubin_cache.get_preemptive_ptb_data(cubin_data, cubin_size);

        auto cubin_uid = TallyCache::cache->cubin_cache.get_cubin_data_uid(cubin_data, cubin_size);
        auto &kernel_name_to_host_func_map = cubin_to_kernel_name_to_host_func_map[cubin_uid];

        register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(original_data, kernel_name_to_host_func_map, original_kernel_map);
        register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(ptb_data, kernel_name_to_host_func_map, ptb_kernel_map);
        register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(dynamic_ptb_data, kernel_name_to_host_func_map, dynamic_ptb_kernel_map);
        register_kernels_from_ptx_fatbin<KERNEL_NAME_MAP_TYPE, KERNEL_MAP_TYPE>(preemptive_ptb_data, kernel_name_to_host_func_map, preemptive_ptb_kernel_map);
    }

    CUresult launch_kernel(CudaLaunchConfig config, const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
    {
        if (config.use_original) {
            CUfunction cu_func = original_kernel_map[func].func;
            assert(cu_func);

            auto err = lcuLaunchKernel(cu_func, gridDim.x, gridDim.y, gridDim.z,
                                    blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, NULL);

            return err;
        } else if (config.use_ptb) {

            CUfunction cu_func = ptb_kernel_map[func].func;
            size_t num_args = ptb_kernel_map[func].num_args;
            assert(cu_func);

            dim3 PTB_grid_dim;
            
            uint32_t total_blocks = blockDim.x * blockDim.y * blockDim.z;
            // Depend on number of PTBs/SM
            if (total_blocks < CUDA_NUM_SM) {
                PTB_grid_dim = dim3(total_blocks);
            } else {
                PTB_grid_dim = dim3(CUDA_NUM_SM * config.num_blocks_per_sm);
            }

            void *KernelParams[num_args];
            for (size_t i = 0; i < num_args - 1; i++) {
                KernelParams[i] = args[i];
            }
            KernelParams[num_args - 1] = &gridDim;

            auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);
            return err;
        } else {
            throw std::runtime_error("Invalid launch config.");
        }
    }

    TallyClientOffline();
    ~TallyClientOffline(){}
};

#endif // TALLY_OFFLINE_H
