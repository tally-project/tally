#include <cstring>
#include <dlfcn.h>
#include <cassert>
#include <unordered_set>
#include <atomic>

#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cache.h>
#include <tally/cache_util.h>
#include <tally/generated/server.h>

// Load cache data into runtime data
void TallyServer::load_measurements(uint32_t cubin_uid)
{
    // Register single-kernel perf cache
    auto &performance_cache = TallyCache::cache->get_performance_cache();
    auto &cached_single_kernel_perf_map = performance_cache.single_kernel_perf_map;;

    for (auto &pair : cached_single_kernel_perf_map) {
        auto &result = pair.second;
        auto &key = result.key;

        auto res_kernel_name = key.kernel_name;
        auto res_cubin_uid = key.cubin_uid;

        if (res_cubin_uid != cubin_uid) {
            continue;
        }

        auto kernel_name_cubin_uid_pair = std::make_pair(res_kernel_name, res_cubin_uid);
        auto registered = kernel_cubin_uid_to_host_func_map.find(kernel_name_cubin_uid_pair) !=
                          kernel_cubin_uid_to_host_func_map.end();

        if (!registered) {
            continue;
        }

        CudaLaunchCall launch_call = convert_key_to_call(result.key);
        CudaLaunchCallConfig call_config(launch_call, result.config);

        CudaLaunchCallConfigResult new_result(launch_call, result.config, result.meta_data, result.metrics);

        single_kernel_perf_map[call_config] = new_result;
    }
}

CudaLaunchCallConfigResult
TallyServer::get_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, bool *found)
{
    CudaLaunchCallConfig call_config(launch_call, launch_config);

    if (single_kernel_perf_map.find(call_config) != single_kernel_perf_map.end()) {
        *found = true;
        return single_kernel_perf_map[call_config];
    } 

    *found = false;
    return CudaLaunchCallConfigResult();
}

void TallyServer::delete_single_kernel_perf(CudaLaunchCall &launch_call, CudaLaunchConfig launch_config)
{
    CudaLaunchCallConfig call_config(launch_call, launch_config);
    single_kernel_perf_map.erase(call_config);
}

void TallyServer::set_single_kernel_perf(
    CudaLaunchCall &launch_call, CudaLaunchConfig launch_config, CudaLaunchMetadata meta_data,
    float norm_speed, float latency, uint32_t iters, float preempt_latency_ms_est)
{
    CudaLaunchCallConfig call_config(
        launch_call,
        launch_config
    );

    CudaLaunchCallConfigResult result(
        launch_call,
        launch_config,
        meta_data,
        KernelProfileMetrics(latency, norm_speed, iters, preempt_latency_ms_est)
    );

    single_kernel_perf_map[call_config] = result;

    auto launch_key = convert_call_to_key(launch_call);
    CudaLaunchKeyConfigResult cache_res(
        launch_key,
        launch_config,
        meta_data,
        result.metrics
    );

    auto &performance_cache = TallyCache::cache->get_performance_cache();
    performance_cache.set_single_kernel_perf(launch_key, launch_config, cache_res);
}

CudaLaunchCallConfigResult TallyServer::get_single_kernel_chosen_config(CudaLaunchCall &launch_call, bool *found)
{
    if (single_kernel_chosen_config_map.find(launch_call) != single_kernel_chosen_config_map.end()) {
        *found = true;
        return single_kernel_chosen_config_map[launch_call];
    }

    *found = false;
    return CudaLaunchCallConfigResult();
}

void TallyServer::set_single_kernel_chosen_config(CudaLaunchCall &launch_call, CudaLaunchCallConfigResult &best_config)
{
    single_kernel_chosen_config_map[launch_call] = best_config;

    auto launch_key = convert_call_to_key(launch_call);

    CudaLaunchKeyConfigResult cache_res(
        launch_key,
        best_config.config,
        best_config.meta_data,
        best_config.metrics
    );
}

void TallyServer::clear_single_kernel_chosen_configs()
{
    single_kernel_chosen_config_map.clear();
}

void TallyServer::save_performance_cache()
{
    TallyCache::cache->save_performance_cache();
}

// ======= Utility functions below =======

CudaLaunchCall TallyServer::convert_key_to_call(CudaLaunchKey key)
{

    auto kernel_name_cubin_uid_pair = std::make_pair(key.kernel_name, key.cubin_uid);

    auto host_func = kernel_cubin_uid_to_host_func_map[
        kernel_name_cubin_uid_pair
    ];

    return CudaLaunchCall(
        host_func,
        key.gridDim,
        key.blockDim
    );
}

CudaLaunchKey TallyServer::convert_call_to_key(CudaLaunchCall call)
{
    return CudaLaunchKey(
        host_func_to_demangled_kernel_name_map[call.func],
        call.gridDim,
        call.blockDim,
        host_func_to_cubin_uid_map[call.func]
    );
}

CudaLaunchCallConfig TallyServer::convert_key_config_to_call_config(CudaLaunchKeyConfig key_config)
{
    auto launch_call = convert_key_to_call(key_config.key);
    return CudaLaunchCallConfig(
        launch_call,
        key_config.config
    );
}

CudaLaunchKeyConfig TallyServer::convert_call_config_to_key_config(CudaLaunchCallConfig call_config)
{
    auto launch_key = convert_call_to_key(call_config.call);
    return CudaLaunchKeyConfig(
        launch_key,
        call_config.config
    );
}

// ======= End Utility functions =======