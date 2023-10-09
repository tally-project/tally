#include <tally/client_offline.h>
#include <tally/cuda_launch.h>

// std::unique_ptr<TallyClientOffline> TallyClientOffline::client_offline;

TallyClientOffline *TallyClientOffline::client_offline;

bool done = false;

__attribute__((__constructor__)) void init_client()
{
    // TallyClientOffline::client_offline = std::make_unique<TallyClientOffline>();

    TallyClientOffline::client_offline =  new TallyClientOffline();
}

void TallyClientOffline::set_exit()
{
    static bool set;

    if (!set) {
        std::atexit([]{ delete TallyClientOffline::client_offline;  });
        set = true;
    }
}

void TallyClientOffline::tune_kernel_launch(std::vector<CudaLaunchConfig> &configs, const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    auto launch_call = CudaLaunchCall(func, gridDim, blockDim);
    auto kernel_name = host_func_to_demangled_kernel_name_map[func];

    spdlog::info("Launch config not found for: " + kernel_name + "_" + launch_call.dim_str());

    auto start = std::chrono::high_resolution_clock::now();

    // Otherwise tune and look for best config
    auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
    auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

    // Measure single-run time
    float time_elapsed;
    float iters;

    cudaDeviceSynchronize();

    launch_kernel_repeat(CudaLaunchConfig::default_config, func, gridDim, blockDim, args, sharedMem, stream, 1000, &time_elapsed, nullptr, 1);

    // In seconds
    float profile_duration = (100 * time_elapsed) / 1000.f;

    // At least run for 0.1 sec
    profile_duration = std::max(profile_duration, 0.5f);

    // Maybe don't exceed 1 minute;
    profile_duration = std::min(profile_duration, 60.f);

    // Run default config first
    CudaLaunchConfig base_config = CudaLaunchConfig::default_config;

    launch_kernel_repeat(base_config, func, gridDim, blockDim, args, sharedMem, stream, profile_duration, &time_elapsed, &iters, -1);

    float base_latency_ms = time_elapsed / iters;

    // Save result to cache
    set_single_kernel_perf(launch_call, base_config, original_kernel_map[launch_call.func].meta_data, 1., base_latency_ms, iters);

    CudaLaunchConfig best_config;
    float best_latency_ms = FLT_MAX;

    for (auto &config : configs) {

        launch_kernel_repeat(config, func, gridDim, blockDim, args, sharedMem, stream, profile_duration, &time_elapsed, &iters, -1);

        float latency_ms = time_elapsed / iters;

        if (latency_ms < best_latency_ms) {
            best_config = config;
            best_latency_ms = latency_ms;
        }

        float norm_speed = base_latency_ms / latency_ms;

        set_single_kernel_perf(launch_call, config, ptb_kernel_map[launch_call.func].meta_data, norm_speed, base_latency_ms, iters);
    }

    float best_norm_speed = base_latency_ms / best_latency_ms;
    if (best_norm_speed < 0.5) {
        best_config = base_config;
        best_norm_speed = 1.;
    }

    bool found_in_cache;
    auto res = get_single_kernel_perf(launch_call, best_config, &found_in_cache);
    set_single_kernel_best_config(launch_call, res);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    spdlog::info("Tuning complete ("+ std::to_string(elapsed.count()) + " ms). Launch config: " + best_config.str() + ". Norm speed: " + std::to_string(best_norm_speed));
}

void TallyClientOffline::register_ptx_transform(const char* cubin_data, size_t cubin_size)
{
    set_exit();

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

CUresult TallyClientOffline::launch_kernel(CudaLaunchConfig config, const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
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

CUresult TallyClientOffline::launch_kernel_repeat(
    CudaLaunchConfig config, const void *func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem,
    cudaStream_t  stream, float dur_seconds, float *time_ms, float *iters, int32_t max_count
)
{
    float _time_ms;
    CUresult err;

    auto startTime = std::chrono::steady_clock::now();
    uint64_t ckpt_count = 0;
    uint64_t count = 0;
    uint64_t elapsed_ns = 0;

    while (true) {

        cudaStreamSynchronize(stream);

        // Perform your steps here
        err = launch_kernel(config, func, gridDim, blockDim, args, sharedMem, stream);
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