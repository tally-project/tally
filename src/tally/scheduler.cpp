#include <cassert>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>

void TallyServer::start_scheduler()
{
    implicit_init_cuda_ctx();

    while (!iox::posix::hasTerminationRequested()) {

#ifdef PROFILE_KERNEL_WISE
        // Run profile kernel-wise experiment
        profile_kernel_wise();

#else
        // Currently the scheduling decision is to launch kernel as long as they arrive
        for (auto &pair : client_data_all) {

            auto &client_data = pair.second;

            if (client_data.has_kernel) {
                client_data.err = (*client_data.kernel_to_dispatch)(CudaLaunchConfig::default_config, false, 0, nullptr, nullptr);
                client_data.has_kernel = false;
            }

        }
#endif

    }
}

void TallyServer::profile_kernel_wise()
{
    // Wait until there are two kernels from two clients
    while (true) {
        int kernel_count = 0;

        for (auto &pair : client_data_all) {
            auto &client_data = pair.second;
            if (client_data.has_kernel) {
                kernel_count++;
            }
        }

        if (kernel_count >= 2) {
            assert(kernel_count == 2);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Launch both kernel - to warm up and in case all experiments are cached already
    for (auto &pair : client_data_all) {
        auto &client_data = pair.second;
        (*client_data.kernel_to_dispatch)(CudaLaunchConfig::default_config, false, 0, nullptr, nullptr);
    }

    CudaLaunchCall launch_calls[2];
    std::function<CUresult(CudaLaunchConfig, bool, float, float*, float*)> kernel_partials[2];
    std::string kernel_names[2];

    int index = 0;
    for (auto &pair : client_data_all) {

        auto &client_data = pair.second;
        launch_calls[index] = client_data.launch_call; 
        kernel_partials[index] = *client_data.kernel_to_dispatch;
        kernel_names[index] = host_func_to_demangled_kernel_name_map[client_data.launch_call.func];
        index++;
    }

    float iters[2];
    float time_elapsed[2];
    CUresult errs[2];

    // This will be called per thread
    auto launch_kernel_func = [kernel_partials, &iters, &time_elapsed, &errs](int idx, CudaLaunchConfig config) {
        errs[idx] = (kernel_partials[idx])(config, true, PROFILE_DURATION, &(time_elapsed[idx]), &(iters[idx]));
    };

    bool has_run_baseline = false;
    float baseline[2];

    // Launch once
    errs[0] = (kernel_partials[0])(CudaLaunchConfig::default_config, false, 0, nullptr, nullptr);
    errs[1] = (kernel_partials[1])(CudaLaunchConfig::default_config, false, 0, nullptr, nullptr);

    auto k1_blockDim = launch_calls[0].blockDim;
    auto k2_blockDim = launch_calls[1].blockDim;

    auto k1_gridDim = launch_calls[0].gridDim;
    auto k2_gridDim = launch_calls[1].gridDim;

    auto k1_configs = CudaLaunchConfig::get_configs(k1_blockDim.x * k1_blockDim.y * k1_blockDim.z, k1_gridDim.x * k1_gridDim.y * k1_gridDim.z);
    auto k2_configs = CudaLaunchConfig::get_configs(k2_blockDim.x * k2_blockDim.y * k2_blockDim.z, k2_gridDim.x * k2_gridDim.y * k2_gridDim.z);

    float best_sum_thrupt = -1.;
    CudaLaunchCallConfigPairResult best_config;
    bool has_new_config = false;

    for (auto &k1_config : k1_configs) {
        for (auto &k2_config : k2_configs) {

            bool found_in_cache = false;
            auto res = get_kernel_pair_perf(launch_calls[0], launch_calls[1], k1_config, k2_config, &found_in_cache);

            if (found_in_cache) {

                cudaDeviceSynchronize();

                std::thread launch_t_1(launch_kernel_func, 0, k1_config);
                std::thread launch_t_2(launch_kernel_func, 1, k2_config);

                launch_t_1.join();
                launch_t_2.join();

                float k1_thrupt = iters[0] / time_elapsed[0];
                float k2_thrupt = iters[1] / time_elapsed[1];

                // Lazily run baseline - can skip if all experiments are cached
                if (!has_run_baseline) {

                    cudaDeviceSynchronize();

                    launch_kernel_func(0, CudaLaunchConfig::default_config);
                    launch_kernel_func(1, CudaLaunchConfig::default_config);

                    baseline[0] = iters[0] / time_elapsed[0];
                    baseline[1] = iters[1] / time_elapsed[1];

                    has_run_baseline = true;
                }

                float k1_norm_speed = k1_thrupt / baseline[0];
                float k2_norm_speed = k2_thrupt / baseline[1];

                // Save the results
                set_kernel_pair_perf(launch_calls[0], launch_calls[1], k1_config, k2_config, k1_norm_speed, k2_norm_speed);
                
                res = get_kernel_pair_perf(launch_calls[0], launch_calls[1], k1_config, k2_config, &found_in_cache);
                assert(found_in_cache);

                has_new_config = true;
            }

            float sum_norm_thrupt = res.get_sum_norm_speed();
            if (sum_norm_thrupt > best_sum_thrupt) {
                best_sum_thrupt = sum_norm_thrupt;
                best_config = res;
            }
        }
    }

    if (has_new_config) {
        set_kernel_pair_best_config(launch_calls[0], launch_calls[1], best_config);
    }

    // clear the flags
    index = 0;
    for (auto &pair : client_data_all) {
        auto &client_data = pair.second;
        client_data.err = errs[index];
        client_data.has_kernel = false;
        index++;
    }
}