
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
    // static std::unique_ptr<TallyClientOffline> client_offline;

    static TallyClientOffline *client_offline;

    uint32_t *global_idx = nullptr;
    uint32_t *curr_idx_arr = nullptr;

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
	CudaLaunchCall convert_key_to_call(CudaLaunchKey key, bool *exists);
	CudaLaunchKey convert_call_to_key(CudaLaunchCall call);

    void register_ptx_transform(const char* cubin_data, size_t cubin_size);

    CUresult launch_kernel(CudaLaunchConfig config, const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream);
    CUresult launch_kernel_repeat(
        CudaLaunchConfig config, const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem,
        cudaStream_t  stream, float dur_seconds, float *time_ms, float *iters, int32_t max_count
    );

    void register_measurements();
    void tune_kernel_launch(std::vector<CudaLaunchConfig> &configs, const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream);

    void set_exit();

    TallyClientOffline(){}

    ~TallyClientOffline()
    {
        TallyCache::cache->save_transform_cache();
        TallyCache::cache->save_performance_cache();
    }
};

#endif // TALLY_OFFLINE_H
