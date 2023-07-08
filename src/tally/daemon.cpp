#include <memory>

#include <tally/daemon.h>
#include <tally/cuda_util.h>
#include <tally/generated/cuda_api.h>

std::shared_ptr<TallyDaemon> TallyDaemon::daemon = std::make_shared<TallyDaemon>();

float TallyDaemon::get_execution_time(CudaLaunchCallConfig &call_config)
{
    if (config_latency_map.find(call_config) != config_latency_map.end()) {
        return config_latency_map[call_config];
    }

    return -1.;
}

void TallyDaemon::set_execution_time(CudaLaunchCallConfig &call_config, float time_ms)
{
    config_latency_map[call_config] = time_ms;

    CudaLaunchKey launch_key(
        host_func_to_demangled_kernel_name_map[call_config.call.func],
        call_config.call.gridDim,
        call_config.call.blockDim
    );

    CudaLaunchKeyConfig key_config(launch_key, call_config.config);

    TallyCache::cache->performance_cache.set_execution_time(key_config, time_ms);
}

bool TallyDaemon::has_launch_config(CudaLaunchCall &call)
{
    return kernel_config_map.find(call) != kernel_config_map.end();
}

CudaLaunchConfig TallyDaemon::get_launch_config(CudaLaunchCall &call)
{
    return kernel_config_map[call];
}

void TallyDaemon::set_launch_config(CudaLaunchCall &call, CudaLaunchConfig &config)
{
    kernel_config_map[call] = config;

    CudaLaunchKey launch_key(
        host_func_to_demangled_kernel_name_map[call.func],
        call.gridDim,
        call.blockDim
    );

    TallyCache::cache->performance_cache.set_launch_config(launch_key, config);
}

void TallyDaemon::save_performance_cache()
{
    TallyCache::cache->save_performance_cache();
}

void TallyDaemon::register_kernels()
{
    lcudaStreamCreate(&stream);

    ptb_kernel_map = register_kernels_from_ptx_fatbin(ptb_ptx_fatbin_strs, mangled_kernel_name_to_host_func_map);
    sliced_kernel_map = register_kernels_from_ptx_fatbin(sliced_ptx_fatbin_strs, mangled_kernel_name_to_host_func_map);
    kernels_registered = true;
}

void TallyDaemon::register_measurements()
{
    auto &cache_kernel_config_map = TallyCache::cache->performance_cache.kernel_config_map;
    auto &cache_config_latency_map = TallyCache::cache->performance_cache.config_latency_map;

    for (auto &pair : cache_kernel_config_map) {
        auto &key = pair.first;
        auto &config = pair.second;
        auto &kernel_name = key.kernel_name;

        const void *host_func = demangled_kernel_name_to_host_func_map[kernel_name];
        CudaLaunchCall call(host_func, key.gridDim, key.blockDim);

        kernel_config_map[call] = config;
    }

    for (auto &pair : cache_config_latency_map) {
        auto &key_config = pair.first;
        auto time_ms = pair.second;
        auto &kernel_name = key_config.key.kernel_name;

        auto *host_func = demangled_kernel_name_to_host_func_map[kernel_name];
        CudaLaunchCallConfig call_config(
            CudaLaunchCall(host_func, key_config.key.gridDim, key_config.key.blockDim),
            key_config.config
        );

        config_latency_map[call_config] = time_ms;
    }
}

void TallyDaemon::register_fat_binary(const char* cubin_data, size_t cubin_size)
{
    auto sliced_data = TallyCache::cache->cubin_cache.get_sliced_data(cubin_data, cubin_size);
    auto ptb_data = TallyCache::cache->cubin_cache.get_ptb_data(cubin_data, cubin_size);

    // Cannot compile those generated code yet because CUDA is not yet initialized
    // So accumulate them first
    merge_vec(sliced_ptx_fatbin_strs, sliced_data);
    merge_vec(ptb_ptx_fatbin_strs, ptb_data);
}