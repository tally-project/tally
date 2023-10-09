#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>
#include <atomic>

#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cache.h>
#include <tally/cache_util.h>
#include <tally/client_offline.h>

// Load cache data into runtime data
void TallyClientOffline::register_measurements()
{
    // Register single-kernel perf cache
    auto &cached_single_kernel_perf_map = TallyCache::cache->performance_cache.single_kernel_perf_map;

    for (auto &pair : cached_single_kernel_perf_map) {
        auto &result = pair.second;

        bool exists;
        CudaLaunchCall launch_call = convert_key_to_call(result.key, &exists);

        if (!exists) {
            continue;
        }

        CudaLaunchCallConfig call_config(launch_call, result.config);
        CudaLaunchCallConfigResult new_result(launch_call, result.config, result.meta_data, result.metrics);

        single_kernel_perf_map[call_config] = new_result;
    }

    // Register single-kernel best config cache
    auto &cached_single_kernel_best_config_map = TallyCache::cache->performance_cache.single_kernel_best_config_map;;

    for (auto &pair : cached_single_kernel_best_config_map) {
        auto &result = pair.second;

        bool exists;
        CudaLaunchCall launch_call = convert_key_to_call(result.key, &exists);

        if (!exists) {
            continue;
        }

        CudaLaunchCallConfigResult new_result(launch_call, result.config, result.meta_data, result.metrics);

        single_kernel_best_config_map[launch_call] = new_result;
    }
}

CudaLaunchCallConfigResult
TallyClientOffline::get_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, bool *found)
{
    CudaLaunchCallConfig call_config(launch_call, launch_config);

    if (single_kernel_perf_map.find(call_config) != single_kernel_perf_map.end()) {
        *found = true;
        return single_kernel_perf_map[call_config];
    } 

    *found = false;
    return CudaLaunchCallConfigResult();
}

void TallyClientOffline::set_single_kernel_perf(
    CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, CudaLaunchMetadata meta_data,
    float norm_speed, float latency, uint32_t iters)
{
    CudaLaunchCallConfig call_config(
        launch_call,
        launch_config
    );

    CudaLaunchCallConfigResult result(
        launch_call,
        launch_config,
        meta_data,
        KernelProfileMetrics(latency, norm_speed, iters)
    );

    single_kernel_perf_map[call_config] = result;

    auto launch_key = convert_call_to_key(launch_call);
    CudaLaunchKeyConfigResult cache_res(
        launch_key,
        launch_config,
        meta_data,
        KernelProfileMetrics(latency, norm_speed, iters)
    );

    TallyCache::cache->performance_cache.set_single_kernel_perf(launch_key, launch_config, cache_res);
    TallyCache::cache->perf_cache_changed = true;
    // save_performance_cache();
}

CudaLaunchCallConfigResult TallyClientOffline::get_single_kernel_best_config(CudaLaunchCall &launch_call, bool *found)
{
    if (single_kernel_best_config_map.find(launch_call) != single_kernel_best_config_map.end()) {
        *found = true;
        return single_kernel_best_config_map[launch_call];
    }

    *found = false;
    return CudaLaunchCallConfigResult();
}

void TallyClientOffline::set_single_kernel_best_config(CudaLaunchCall &launch_call, CudaLaunchCallConfigResult &best_config)
{
    single_kernel_best_config_map[launch_call] = best_config;

    auto launch_key = convert_call_to_key(launch_call);

    CudaLaunchKeyConfigResult cache_res(
        launch_key,
        best_config.config,
        best_config.meta_data,
        best_config.metrics
    );

    TallyCache::cache->performance_cache.set_single_kernel_best_config(launch_key, cache_res);
    TallyCache::cache->perf_cache_changed = true;
    // save_performance_cache();
}

// ======= Utility functions below =======

CudaLaunchCall TallyClientOffline::convert_key_to_call(CudaLaunchKey key, bool *exists)
{
    auto demangled_kernel_name_and_cubin_uid = std::pair<std::string, size_t>(key.kernel_name, key.cubin_uid);

    if (demangled_kernel_name_and_cubin_uid_to_host_func_map.find(demangled_kernel_name_and_cubin_uid) == demangled_kernel_name_and_cubin_uid_to_host_func_map.end()) {
        *exists = false;
        return CudaLaunchCall(0, 0, 0);
    }

    *exists = true;

    auto host_func = demangled_kernel_name_and_cubin_uid_to_host_func_map[demangled_kernel_name_and_cubin_uid];

    return CudaLaunchCall(
        host_func,
        key.gridDim,
        key.blockDim
    );
}

CudaLaunchKey TallyClientOffline::convert_call_to_key(CudaLaunchCall call)
{
    return CudaLaunchKey(
        host_func_to_demangled_kernel_name_map[call.func],
        call.gridDim,
        call.blockDim,
        host_func_to_cubin_uid_map[call.func]
    );
}

// ======= End Utility functions =======