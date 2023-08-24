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
void TallyServer::register_measurements()
{
    // Register single-kernel perf cache
    auto &cached_single_kernel_perf_map = TallyCache::cache->performance_cache.single_kernel_perf_map;;

    for (auto &pair : cached_single_kernel_perf_map) {
        auto &result = pair.second;

        CudaLaunchCall launch_call = convert_key_to_call(result.key);
        CudaLaunchCallConfig call_config(launch_call, result.config);

        CudaLaunchCallConfigResult new_result(launch_call, result.config, result.meta_data, result.metrics);

        single_kernel_perf_map[call_config] = new_result;
    }

    // Register single-kernel best config cache
    auto &cached_single_kernel_best_config_map = TallyCache::cache->performance_cache.single_kernel_best_config_map;;

    for (auto &pair : cached_single_kernel_best_config_map) {
        auto &result = pair.second;

        CudaLaunchCall launch_call = convert_key_to_call(result.key);
        CudaLaunchCallConfigResult new_result(launch_call, result.config, result.meta_data, result.metrics);

        single_kernel_best_config_map[launch_call] = new_result;
    }

    // Register kernel-pair perf cache
    auto &cached_kernel_pair_perf_map = TallyCache::cache->performance_cache.kernel_pair_perf_map;;

    for (auto &pair : cached_kernel_pair_perf_map) {
        auto &key_pair = pair.first;
        auto &config_res_map = pair.second;

        auto &key_1 = key_pair.launch_key_1;
        auto &key_2 = key_pair.launch_key_2;

        CudaLaunchCall launch_call_1 = convert_key_to_call(key_1);
        CudaLaunchCall launch_call_2 = convert_key_to_call(key_2);

        CudaLaunchCallPair call_pair(launch_call_1, launch_call_2);

        for (auto &config_res_pair : config_res_map) {

            auto &key_config_pair = config_res_pair.first;
            auto &res = config_res_pair.second;

            CudaLaunchCallConfigPair call_config_pair = convert_key_config_pair_to_call_config_pair(key_config_pair);
            CudaLaunchCallConfigPairResult runtime_res = convert_pair_res_to_runtime_res(res);

            kernel_pair_perf_map[call_pair][call_config_pair] = runtime_res;
        }
    }

    // Register kernel-pair best config cache
    auto &cached_kernel_pair_best_config_map = TallyCache::cache->performance_cache.kernel_pair_best_config_map;;

    for (auto &pair : cached_kernel_pair_best_config_map) {
        auto &key_pair = pair.first;
        auto &res = pair.second;

        auto &key_1 = key_pair.launch_key_1;
        auto &key_2 = key_pair.launch_key_2;

        CudaLaunchCall launch_call_1 = convert_key_to_call(key_1);
        CudaLaunchCall launch_call_2 = convert_key_to_call(key_2);

        CudaLaunchCallPair call_pair(launch_call_1, launch_call_2);
        CudaLaunchCallConfigPairResult new_result = convert_pair_res_to_runtime_res(res);
       
        kernel_pair_best_config_map[call_pair] = new_result;
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

void TallyServer::set_single_kernel_perf(
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
    save_performance_cache();
}

CudaLaunchCallConfigResult TallyServer::get_single_kernel_best_config(CudaLaunchCall &launch_call, bool *found)
{
    if (single_kernel_best_config_map.find(launch_call) != single_kernel_best_config_map.end()) {
        *found = true;
        return single_kernel_best_config_map[launch_call];
    }

    *found = false;
    return CudaLaunchCallConfigResult();
}

void TallyServer::set_single_kernel_best_config(CudaLaunchCall &launch_call, CudaLaunchCallConfigResult &best_config)
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
    save_performance_cache();
}

CudaLaunchCallConfigPairResult
TallyServer::get_kernel_pair_perf(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2, CudaLaunchConfig &launch_config_1, CudaLaunchConfig &launch_config_2, bool *found)
{
    CudaLaunchCallPair call_pair(launch_call_1, launch_call_2);

    if (kernel_pair_perf_map.find(call_pair) != kernel_pair_perf_map.end()) {
        auto &config_res_map = kernel_pair_perf_map[call_pair];

        CudaLaunchCallConfig call_config_1(launch_call_1, launch_config_1);
        CudaLaunchCallConfig call_config_2(launch_call_2, launch_config_2);

        CudaLaunchCallConfigPair call_config_pair(call_config_1, call_config_2);

         if (config_res_map.find(call_config_pair) != config_res_map.end()) {
            *found = true;
            return config_res_map[call_config_pair];
        }
    } 

    *found = false;
    return CudaLaunchCallConfigPairResult();
}

void TallyServer::set_kernel_pair_perf(
    CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2,
    CudaLaunchConfig &launch_config_1, CudaLaunchConfig &launch_config_2,
    CudaLaunchMetadata meta_data_1, CudaLaunchMetadata meta_data_2,
    float norm_speed_1, float norm_speed_2, float latency_1, float latency_2,
    float fixed_workload_latency, float fixed_workload_speedup,
    float unfair_workload_latency, float unfair_workload_speedup
)
{
    CudaLaunchCallPair call_pair(launch_call_1, launch_call_2);

    CudaLaunchCallConfig call_config_1(launch_call_1, launch_config_1);
    CudaLaunchCallConfig call_config_2(launch_call_2, launch_config_2);
    CudaLaunchCallConfigPair call_config_pair(call_config_1, call_config_2);

    CudaLaunchCallConfigPairResult result(
        std::make_pair<CudaLaunchCallConfig, KernelProfileMetrics>(std::move(call_config_1), KernelProfileMetrics(latency_1, norm_speed_1)),
        std::make_pair<CudaLaunchCallConfig, KernelProfileMetrics>(std::move(call_config_2), KernelProfileMetrics(latency_2, norm_speed_2)),
        std::make_pair<CudaLaunchCallConfig, CudaLaunchMetadata>(std::move(call_config_1), std::move(meta_data_1)),
        std::make_pair<CudaLaunchCallConfig, CudaLaunchMetadata>(std::move(call_config_2), std::move(meta_data_2)),
        WorkloadPerformance(fixed_workload_latency, fixed_workload_speedup),
        WorkloadPerformance(unfair_workload_latency, unfair_workload_speedup)
    );

    kernel_pair_perf_map[call_pair][call_config_pair] = result;

    CudaLaunchKey launch_key_1 = convert_call_to_key(launch_call_1);
    CudaLaunchKey launch_key_2 = convert_call_to_key(launch_call_2);

    CudaLaunchKeyConfigPairResult cache_res = convert_pair_res_to_cache_res(result);

    TallyCache::cache->performance_cache.set_kernel_pair_perf(launch_key_1, launch_key_2, launch_config_1, launch_config_2, cache_res);
    save_performance_cache();
}

CudaLaunchCallConfigPairResult TallyServer::get_kernel_pair_best_config(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2, bool *found)
{
    CudaLaunchCallPair call_pair(launch_call_1, launch_call_2);

    if (kernel_pair_best_config_map.find(call_pair) != kernel_pair_best_config_map.end()) {
        *found = true;
        return kernel_pair_best_config_map[call_pair];
    }

    *found = false;
    return CudaLaunchCallConfigPairResult();
}

void TallyServer::set_kernel_pair_best_config(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2, CudaLaunchCallConfigPairResult best_config)
{
    CudaLaunchCallPair call_pair(launch_call_1, launch_call_2);

    kernel_pair_best_config_map[call_pair] = best_config;

    CudaLaunchKey launch_key_1 = convert_call_to_key(launch_call_1);
    CudaLaunchKey launch_key_2 = convert_call_to_key(launch_call_2);

    CudaLaunchKeyConfigPairResult cache_res = convert_pair_res_to_cache_res(best_config);

    TallyCache::cache->performance_cache.set_kernel_pair_best_config(launch_key_1, launch_key_2, cache_res);
    save_performance_cache();
}

void TallyServer::save_performance_cache()
{
    TallyCache::cache->save_performance_cache();
}

// ======= Utility functions below =======

CudaLaunchCall TallyServer::convert_key_to_call(CudaLaunchKey key)
{

    auto host_func = demangled_kernel_name_and_cubin_hash_to_host_func_map[
        std::make_pair<std::string, size_t>(std::move(key.kernel_name), std::move(key.cubin_hash))
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
        host_func_to_cubin_hash_map[call.func]
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

CudaLaunchCallConfigPair TallyServer::convert_key_config_pair_to_call_config_pair(CudaLaunchKeyConfigPair key_config_pair)
{
    auto call_config_1 = convert_key_config_to_call_config(key_config_pair.key_config_1);
    auto call_config_2 = convert_key_config_to_call_config(key_config_pair.key_config_2);

    return CudaLaunchCallConfigPair(
        call_config_1,
        call_config_2
    );
}

CudaLaunchKeyConfigPair TallyServer::convert_call_config_pair_to_key_config_pair(CudaLaunchCallConfigPair call_config_pair)
{
    auto key_config_1 = convert_call_config_to_key_config(call_config_pair.call_config_1);
    auto key_config_2 = convert_call_config_to_key_config(call_config_pair.call_config_2);

    return CudaLaunchKeyConfigPair(
        key_config_1,
        key_config_2
    );
}

CudaLaunchCallConfigPairResult TallyServer::convert_pair_res_to_runtime_res(CudaLaunchKeyConfigPairResult res)
{
    CudaLaunchCallConfig call_config_1 = convert_key_config_to_call_config(res.config_key_norm_speed_1.first);
    CudaLaunchCallConfig call_config_2 = convert_key_config_to_call_config(res.config_key_norm_speed_2.first);
    
    CudaLaunchCallConfigPairResult runtime_res(
        std::make_pair<CudaLaunchCallConfig, KernelProfileMetrics>(std::move(call_config_1), std::move(res.config_key_norm_speed_1.second)),
        std::make_pair<CudaLaunchCallConfig, KernelProfileMetrics>(std::move(call_config_2), std::move(res.config_key_norm_speed_2.second)),
        std::make_pair<CudaLaunchCallConfig, CudaLaunchMetadata>(std::move(call_config_1), std::move(res.config_key_meta_data_1.second)),
        std::make_pair<CudaLaunchCallConfig, CudaLaunchMetadata>(std::move(call_config_2), std::move(res.config_key_meta_data_2.second)),
        res.fixed_workload_perf,
        res.unfair_workload_perf
    );

    return runtime_res;
}

CudaLaunchKeyConfigPairResult TallyServer::convert_pair_res_to_cache_res(CudaLaunchCallConfigPairResult res)
{
    CudaLaunchKeyConfig key_config_1 = convert_call_config_to_key_config(res.call_config_norm_speed_1.first);
    CudaLaunchKeyConfig key_config_2 = convert_call_config_to_key_config(res.call_config_norm_speed_2.first);
    
    CudaLaunchKeyConfigPairResult cache_res(
        std::make_pair<CudaLaunchKeyConfig, KernelProfileMetrics>(std::move(key_config_1), std::move(res.call_config_norm_speed_1.second)),
        std::make_pair<CudaLaunchKeyConfig, KernelProfileMetrics>(std::move(key_config_2), std::move(res.call_config_norm_speed_2.second)),
        std::make_pair<CudaLaunchKeyConfig, CudaLaunchMetadata>(std::move(key_config_1), std::move(res.call_config_meta_data_1.second)),
        std::make_pair<CudaLaunchKeyConfig, CudaLaunchMetadata>(std::move(key_config_2), std::move(res.call_config_meta_data_2.second)),
        res.fixed_workload_perf,
        res.unfair_workload_perf
    );

    return cache_res;
}

// ======= End Utility functions =======