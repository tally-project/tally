#include <cassert>
#include <cfloat>
#include <random>
#include <unordered_set>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_launch.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

struct DispatchedKernel {
public:
    KernelLaunchWrapper kernel_wrapper;

    // If true, this kernel is running
    // If false, this kernel has been signaled to stop (may not have stopped)
    bool running = false;

    CudaLaunchConfig config;

    SlicedKernelArgs slice_args;
};

struct cudaEventWrapper {
    cudaEvent_t event;
    bool active = false;
};

void TallyServer::priority_launch_and_measure_kernel(KernelLaunchWrapper &kernel_wrapper, int32_t client_id)
{
    // Store temporary data here
    static std::unordered_map<CudaLaunchCallConfig, TempKernelProfileMetrics> temp_perf_data;
    static std::unordered_map<CudaLaunchCallConfig, uint32_t> warmup_perf_data;
    static std::unordered_map<CudaLaunchCall, bool> abnormal_launch_calls;

    std::vector<CudaLaunchConfig> profiled_configs;

    auto append_to_profiled_configs = [&](CudaLaunchConfig &config) {

        auto it = std::find(profiled_configs.begin(), profiled_configs.end(), config);
        if (it == profiled_configs.end()) {
            profiled_configs.push_back(config);
        }
    };

    auto &launch_call = kernel_wrapper.launch_call;
    auto &client_data = client_data_all[client_id];
    auto kernel_name = host_func_to_demangled_kernel_name_map[launch_call.func];
    auto cubin_uid = host_func_to_cubin_uid_map[launch_call.func];
    auto kernel_str = kernel_name + "_" + launch_call.dim_str() + "_" + std::to_string(cubin_uid);

    float time_elapsed;
    float iters;

    auto base_config = CudaLaunchConfig::default_config;
    append_to_profiled_configs(base_config);

    // First profile the original kernel
    bool found;
    auto original_res = get_single_kernel_perf(launch_call, base_config, &found);

    if (!found) {
        CudaLaunchCallConfig original_call_config(launch_call, base_config);
        auto &metrics = temp_perf_data[original_call_config];

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        kernel_wrapper.kernel_to_dispatch(base_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, 0, true);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        time_elapsed = elapsed.count();

        if (warmup_perf_data.find(original_call_config) == warmup_perf_data.end()) {
            warmup_perf_data[original_call_config] = 0;
        }

        warmup_perf_data[original_call_config]++;

        // first 10 is warm up
        if (warmup_perf_data[original_call_config] < 10) {
            return;
        }

        metrics.add_measurement(time_elapsed);

        if (metrics.count >= KERNEL_PROFILE_ITERATIONS) {
            set_single_kernel_perf(launch_call, base_config, original_kernel_map[launch_call.func].meta_data, 1., metrics.avg_latency_ms, 0, metrics.avg_latency_ms);
        }

        return;
    }

    auto &original_metrics = original_res.metrics;
    float has_executed = false;
    auto priority_configs = CudaLaunchConfig::get_priority_configs(launch_call);

    // maximum number of slices from priority_configs
    uint32_t max_num_slices = 1;
    for (auto &config : priority_configs) {
        if (config.use_sliced) {
            if (config.num_slices > max_num_slices) {
                max_num_slices = config.num_slices;
            }
        }
    }

    auto launch_kernel_with_config = [&](CudaLaunchConfig &config) {
        // prepare ptb args
        auto ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];

        if (config.use_sliced) {
            
            // prepare sliced args
            auto slice_args = get_sliced_kernel_args(launch_call.gridDim, config.num_slices);

            // We want to provision the overhead to launch sliced kernel one by one
            for (size_t i = 0; i < slice_args.block_offsets.size(); i++) {
                slice_args.launch_idx = i;
                kernel_wrapper.kernel_to_dispatch(config, nullptr, nullptr, &slice_args, false, 0, nullptr, nullptr, -1, true);
                cudaDeviceSynchronize();
            }

        } else {

            if (config.use_dynamic_ptb || config.use_preemptive_ptb) {
                cudaMemsetAsync(ptb_args, 0, sizeof(PTBKernelArgs), kernel_wrapper.launch_stream);
            }

            kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, nullptr, false, 0, nullptr, nullptr, -1, true);
        }

        has_executed = true;
    };

    auto profile_config = [&](CudaLaunchConfig &config) {

        CudaLaunchCallConfig call_config(launch_call, config);
        auto &metrics = temp_perf_data[call_config];

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        launch_kernel_with_config(config);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        time_elapsed = elapsed.count();

        metrics.add_measurement(time_elapsed);

        if (metrics.count >= KERNEL_PROFILE_ITERATIONS) {
            float norm_speed = original_metrics.latency_ms / metrics.avg_latency_ms;

            // Likely the base config is not profiled correctly, so we delete it and do it again.
            if (norm_speed >= 1.8 && (abnormal_launch_calls.find(launch_call) == abnormal_launch_calls.end())) {
                delete_single_kernel_perf(launch_call, base_config);

                CudaLaunchCallConfig original_call_config(launch_call, base_config);
                temp_perf_data.erase(original_call_config);
                warmup_perf_data.erase(original_call_config);

                TALLY_SPD_WARN("Detecting abnormal norm_speed of " + std::to_string(norm_speed) + " of kernel " + kernel_str + "\n");
                
                // Book mark so if it is still the case again, will leave it as it is
                abnormal_launch_calls[launch_call] = true;
            } else {

                float preemption_latency_ms = 0.;
                if (config.use_preemptive_ptb) {

                    if (config.max_worker_blocks < CUDA_NUM_SM) {
                        preemption_latency_ms = PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS;
                    } else {
                        auto latency_ms = metrics.avg_latency_ms;
                        auto batch_size = config.blocks_per_sm * std::min((uint32_t)CUDA_NUM_SM, launch_call.num_blocks);
                        auto num_batches = (launch_call.num_blocks + batch_size - 1) / batch_size;
                        preemption_latency_ms =  latency_ms / (float) num_batches;

                        // NOTE: division by a factor is heuristical
                        // default is 2 - let's suppose at any given time,
                        // we will wait in average half of the execution time
                        preemption_latency_ms = preemption_latency_ms / PRIORITY_PTB_PREEMPTION_LATENCY_CALCULATION_FACTOR;
                    }

                } else if (config.use_dynamic_ptb) {

                    preemption_latency_ms = metrics.avg_latency_ms;

                } else if (config.use_sliced) {

                    if (config.num_slices > max_num_slices) {
                        preemption_latency_ms = PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS;
                    } else {
                        preemption_latency_ms = metrics.avg_latency_ms / config.num_slices;
                    }

                }
                
                set_single_kernel_perf(launch_call, config, ptb_kernel_map[launch_call.func].meta_data, norm_speed, metrics.avg_latency_ms, 0, preemption_latency_ms);
            }
        }
    };

    auto set_chosen_config = [&](CudaLaunchCallConfigResult &res) {

        auto config = res.config;
        auto &config_metrics = res.metrics;
        auto latency_ms = config_metrics.latency_ms;
        auto norm_speed = config_metrics.norm_speed;
        auto preemption_latency_ms = config_metrics.preemption_latency_ms;

        if (has_executed) {
            TALLY_SPD_LOG_ALWAYS("Tuning complete for: " + kernel_str);
            if (config.use_preemptive_ptb && config.max_worker_blocks < CUDA_NUM_SM) {
                TALLY_SPD_WARN(
                    "Setting max_worker_blocks to " + std::to_string(config.max_worker_blocks) +
                    " to mitigate longer preemption latency than max allowed."
                ); 
            } else if (config.use_sliced && config.num_slices > max_num_slices) {
                TALLY_SPD_WARN(
                    "Setting num_slices to " + std::to_string(config.num_slices) +
                    " as opposed to " + std::to_string(max_num_slices) +
                    " to mitigate longer preemption latency than max allowed."
                ); 
            } else if (config.use_original) {
                TALLY_SPD_WARN("Fall back to original kernel");
            }

            if (preemption_latency_ms > PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS) {
                TALLY_SPD_WARN("Estimated preemption latency of " + std::to_string(preemption_latency_ms) + " is greater than max allowed.");
            }

            for (auto &config : profiled_configs) {
                auto res = get_single_kernel_perf(launch_call, config, &found);
                auto norm_speed = res.metrics.norm_speed;
                auto latency_ms = res.metrics.latency_ms;
                auto preemption_latency_ms = res.metrics.preemption_latency_ms;
                TALLY_SPD_LOG_ALWAYS(
                    "    Launch config: " + config.str() + ". " +
                    "Estimated Preemption Latency: " + std::to_string(preemption_latency_ms) + " ms " + 
                    "Latency: " + std::to_string(latency_ms) + " ms " +
                    "Norm speed: " + std::to_string(norm_speed)
                );
            }

            TALLY_SPD_LOG_ALWAYS(
                "Chosen config: " + config.str() + ". " +
                "Estimated Preemption Latency: " + std::to_string(preemption_latency_ms) + " ms " + 
                "Latency: " + std::to_string(latency_ms) + " ms " +
                "Norm speed: " + std::to_string(norm_speed) + "\n"
            );
        }
        
        set_single_kernel_chosen_config(launch_call, res);
    };

    // Certain kernels do not have transformed kernel available
    auto has_transform = preemptive_ptb_kernel_map.find(launch_call.func) != preemptive_ptb_kernel_map.end();
    if (!has_transform)
    {
        set_chosen_config(original_res);
        if (!has_executed) {
            launch_kernel_with_config(base_config);
        }
        return;
    }

    // gradually lower the required threshold 
    for ( float perf_threshold = 1.0;
          perf_threshold >= PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD;
          perf_threshold -= 0.1
    ) {

        float effective_perf_threshold = perf_threshold;

        // put a higher threshold for short kernels
        if (original_metrics.latency_ms <= PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS) {
            effective_perf_threshold = PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD_FOR_SHORT_KERNEL;
        }

        // Favor config from priority_configs over original config
        for (auto &config : priority_configs) {

            append_to_profiled_configs(config);

            auto res = get_single_kernel_perf(launch_call, config, &found);
            if (!found) {
                profile_config(config);
            }
            res = get_single_kernel_perf(launch_call, config, &found);

            // still collecting
            if (!found) {
                return;
            }

            auto &config_metrics = res.metrics;
            auto preemption_latency_ms = config_metrics.preemption_latency_ms;

            // If both norm speed and estimated preemption latency are accepted
            // set this config as the chosen config
            if (config_metrics.norm_speed >= effective_perf_threshold) {
                
                if (preemption_latency_ms <= PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS) {

                    set_chosen_config(res);
                    if (!has_executed) {
                        launch_kernel_with_config(config);
                    }
                    return;
                }
            }

            // has_executed but not satisfied
            if (has_executed) {
                return;
            }
        }
    }

    // Now if original kernel latency is within PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS
    // Choose it
    if (original_metrics.latency_ms <= PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS) {
        set_chosen_config(original_res);
        if (!has_executed) {
            launch_kernel_with_config(base_config);
        }
        return;
    }

    // Meaning num blocks <= NUM_SMs
    // since the original kernel latency is beyond limit
    // We will use these two as the basis to further limit resources
    if (priority_configs.empty()) {
        priority_configs = {
            CudaLaunchConfig::get_preemptive_ptb_config(1),
            CudaLaunchConfig::get_sliced_config(2)
        };
    }

    auto get_last_norm_speed = [&](bool use_preemptive_ptb, bool use_sliced) {
        float norm_speed = -1.;

        for (int i = priority_configs.size() - 1; i >= 0; i--) {
            auto &config = priority_configs[i];
            if ((use_preemptive_ptb && config.use_preemptive_ptb) || (use_sliced && config.use_sliced)) {
                auto res = get_single_kernel_perf(launch_call, config, &found);
                if (found) {
                    norm_speed = res.metrics.norm_speed;
                    break;
                }
            }
        }

        return norm_speed;
    };

    float preemptive_norm_speed = get_last_norm_speed(true, false);
    float sliced_norm_speed = get_last_norm_speed(false, true);

    // Otherwise, we will launch (< CUDA_NUM_SM) thread blocks to further restrict the preemption latency
    if (preemptive_norm_speed >= PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD ||
        sliced_norm_speed >= PRIORITY_FALL_BACK_TO_ORIGINAL_THRESHOLD
    ) {

        CudaLaunchConfig config;

        if (preemptive_norm_speed >= sliced_norm_speed) {
            config = CudaLaunchConfig::get_preemptive_ptb_config(1);
            auto preemptive_res = get_single_kernel_perf(launch_call, config, &found);

            auto max_worker_blocks = (PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS / preemptive_res.metrics.preemption_latency_ms) *
                                     (float) std::min((uint32_t)CUDA_NUM_SM, launch_call.num_blocks);
            config.max_worker_blocks = std::max((uint32_t)std::floor(max_worker_blocks), PRIORITY_MIN_WORKER_BLOCKS);

            if (config.max_worker_blocks >= launch_call.num_blocks) {
                config = base_config;
            }
        }

        else {
            // provision a 30% overhead
            uint32_t num_slices = std::ceil(original_metrics.latency_ms * 1.3 / PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS);
            num_slices = std::max(num_slices, max_num_slices);

            // let's at least allow PRIORITY_MIN_WORKER_BLOCKS blocks per slice
            uint32_t num_slices_lower_bound = (launch_call.num_blocks + PRIORITY_MIN_WORKER_BLOCKS - 1) / PRIORITY_MIN_WORKER_BLOCKS;
            num_slices = std::min(num_slices, num_slices_lower_bound);

            num_slices = std::min(num_slices, launch_call.num_blocks);
            if (num_slices > 1) {
                config = CudaLaunchConfig::get_sliced_config(num_slices);
            } else {
                config = base_config;
            }
        }

        append_to_profiled_configs(config);

        auto res = get_single_kernel_perf(launch_call, config, &found);
        if (!found) {
            profile_config(config);
        }
        res = get_single_kernel_perf(launch_call, config, &found);

        // still collecting
        if (!found) {
            return;
        }

        set_chosen_config(res);
        if (!has_executed) {
            launch_kernel_with_config(config);
        }
        return;
    }

    // Finally, if nothing works, fall back to use the original kernel
    set_chosen_config(original_res);
    if (!has_executed) {
        launch_kernel_with_config(base_config);
    }
}

// Always prioritize to execute kernels from high-priority job.
// Run low-priority job only when there is no high-priority kernel.
// Preempt low-priority kernel when high-priority kernel arrives.

// For high-priority, we launch as many kernels as there are in the queue
// For low-priority, we at most launch one kernel
void TallyServer::run_priority_scheduler()
{
    TALLY_SPD_LOG_ALWAYS("Running priority scheduler ...");
    TALLY_SPD_LOG_ALWAYS("PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS=" + std::to_string(PRIORITY_MAX_ALLOWED_PREEMPTION_LATENCY_MS));
    TALLY_SPD_LOG_ALWAYS("PRIORITY_MIN_WAIT_TIME_MS=" + std::to_string(PRIORITY_MIN_WAIT_TIME_MS));
    TALLY_SPD_LOG_ALWAYS("PRIORITY_USE_ORIGINAL_CONFIGS=" + std::to_string(PRIORITY_USE_ORIGINAL_CONFIGS));
    TALLY_SPD_LOG_ALWAYS("PRIORITY_USE_SPACE_SHARE=" + std::to_string(PRIORITY_USE_SPACE_SHARE));
    TALLY_SPD_LOG_ALWAYS("PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS=" + std::to_string(PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS));

    // Keep in track low-priority kernels that are in progress
    // The boolean indicates whether the kernel is running/stopped
    std::map<ClientPriority, DispatchedKernel, std::greater<ClientPriority>> in_progress_kernels;
    std::map<uint32_t, cudaEventWrapper> client_events;

    cudaStream_t retreat_stream;
    cudaStreamCreateWithFlags(&retreat_stream, cudaStreamNonBlocking);

    auto highest_priority_idle_start_t = std::chrono::high_resolution_clock::now();
    bool has_warned = false;
    KernelLaunchWrapper kernel_wrapper;

    if (PRIORITY_USE_SPACE_SHARE) {

        while (!iox::posix::hasTerminationRequested()) {

            bool is_highest_priority = true;

            for (auto &pair : client_priority_map) {

                auto &client_priority = pair.first;
                auto client_id = client_priority.client_id;
                auto priority = client_priority.priority;
                auto &client_data = client_data_all[client_id];

                if (client_data.has_exit) {
                    client_data_all.erase(client_id);
                    client_priority_map.erase(client_priority);
                    break;
                }

                if (is_highest_priority) {

                    while (true) {

                        bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

                        if (!succeeded) {
                            break;
                        }

                        kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
                        kernel_wrapper.free_args();
                        client_data.queue_size--;
                    }
                } else {

                    bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);
                    if (succeeded) {
                        kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
                        kernel_wrapper.free_args();
                        client_data.queue_size--;
                    }

                }

                is_highest_priority = false;
            }
        }

    } else if (PRIORITY_DISABLE_TRANSFORMATION) {

        while (!iox::posix::hasTerminationRequested()) {

            bool is_highest_priority = true;

            for (auto &pair : client_priority_map) {

                auto &client_priority = pair.first;
                auto client_id = client_priority.client_id;
                auto &client_data = client_data_all[client_id];

                if (client_events.find(client_id) == client_events.end()) {
                    auto &event = client_events[client_id].event;
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                }

                auto &client_event_wrapper = client_events[client_id];
                auto client_event = client_event_wrapper.event;

                if (client_data.has_exit) {
                    cudaEventDestroy(client_event);
                    
                    client_data_all.erase(client_id);
                    client_events.erase(client_id);
                    client_priority_map.erase(client_priority);
                    break;
                }

                bool succeeded;
            
                if (is_highest_priority) {

                    // this job may have been promoted from low to high so it has a kernel in in_progress_kernels
                    if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {
                        auto &dispatched_kernel = in_progress_kernels[client_priority];
                        auto &dispatched_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                        auto running = dispatched_kernel.running;

                        // wait for it to complete
                        cudaStreamSynchronize(dispatched_kernel_wrapper.launch_stream);

                        // Erase it from in-progress
                        in_progress_kernels.erase(client_priority);

                        // mark as launched
                        dispatched_kernel_wrapper.free_args();
                        client_data.queue_size--;
                    }

                    // Always try to fetch and launch kernel
                    succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

                    // all kernels have been launched
                    if (!succeeded) {

                        // query if there is active kernel
                        if (client_event_wrapper.active) {

                            // then we wait until all kernels have completed execution to
                            // potentially consider launching low-priority kernels
                            if (cudaEventQuery(client_event) != cudaSuccess) {
                                break;
                            }
                            
                            // mark event as inactive as all kernels have finished
                            else {
                                client_event_wrapper.active = false;
                            }

                        }
                    }

                } else {

                    // For non-highest-priority client, we only launch at most 1 kernel at any time
                    // therefore we will query whether the kernel has finished

                    // First check whether this client already has a kernel running
                    if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {

                        auto &dispatched_kernel = in_progress_kernels[client_priority];
                        auto &dispatched_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                        auto running = dispatched_kernel.running;
                        auto &config = dispatched_kernel.config;

                        // First check if the kernel is still running
                        // note that the running flag only tells us whether the kernel has been signaled to stop
                        // it is possible that it has not yet terminated
                        if (cudaEventQuery(client_event) != cudaSuccess) {
                            break;
                        }

                        // Erase from in-progress
                        client_event_wrapper.active = false;
                        dispatched_kernel_wrapper.free_args();
                        in_progress_kernels.erase(client_priority);
                        client_data.queue_size--;
                    }

                    // Try to fetch kernel from low-priority queue
                    succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);
                }

                // Successfully fetched a kernel from the launch queue
                if (succeeded) {

                    // Stop any running kernel
                    for (auto &in_progress_kernel : in_progress_kernels) {

                        auto &dispatched_kernel = in_progress_kernel.second;
                        auto running = dispatched_kernel.running;

                        if (!running) {
                            continue;
                        }

                        auto in_progress_client_id = in_progress_kernel.first.client_id;
                        auto &in_progress_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                        auto &in_progress_client_data = client_data_all[in_progress_client_id];
                        auto &in_progress_kernel_event_wrapper = client_events[in_progress_client_id];
                        auto &in_progress_kernel_event = in_progress_kernel_event_wrapper.event;

                        // First check whether this kernel has already finished
                        if (cudaEventQuery(in_progress_kernel_event) == cudaSuccess) {

                            // Erase if finished
                            in_progress_kernel_event_wrapper.active = false;
                            in_progress_kernel_wrapper.free_args();
                            in_progress_kernels.erase(in_progress_kernel.first);
                            in_progress_client_data.queue_size--;
                            break;
                        }

                        // Mark that this kernel has been signaled to stop
                        dispatched_kernel.running = false;

                        // We know there are at most one kernel running, so break
                        break;
                    }

                    // Now launch the high priority kernel
                    auto config = CudaLaunchConfig::default_config;

                    if (!is_highest_priority) {

                        // bookkeep kernel launch if it is not highest-priority 
                        in_progress_kernels[client_priority] = DispatchedKernel(kernel_wrapper, true, config);
                    }

                    // Launch the kernel
                    kernel_wrapper.kernel_to_dispatch(config, NULL, NULL, NULL, false, 0, nullptr, nullptr, -1, true);
                    
                    // record event
                    cudaEventRecord(client_event, kernel_wrapper.launch_stream);
                    client_event_wrapper.active = true;

                    // For highest priority, we can directly mark the kernel as launched as we won't ever preempt it
                    if (is_highest_priority) {

                        // Mark as launched
                        kernel_wrapper.free_args();
                        client_data.queue_size--;

                    }

                    break;
                }

                // Did not fetch a kernel
                // Proceed to check the next-priority client

                is_highest_priority = false;
            }
        }

    } else {

        while (!iox::posix::hasTerminationRequested()) {

            bool is_highest_priority = true;

            for (auto &pair : client_priority_map) {

                auto &client_priority = pair.first;
                auto client_id = client_priority.client_id;
                auto &client_data = client_data_all[client_id];

                if (client_events.find(client_id) == client_events.end()) {
                    auto &event = client_events[client_id].event;
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                }

                auto &client_event_wrapper = client_events[client_id];
                auto client_event = client_event_wrapper.event;

                if (client_data.has_exit) {
                    cudaEventDestroy(client_event);
                    
                    client_data_all.erase(client_id);
                    client_events.erase(client_id);
                    client_priority_map.erase(client_priority);
                    break;
                }

                bool succeeded;
            
                if (is_highest_priority) {

                    // this job may have been promoted from low to high so it has a kernel in in_progress_kernels
                    if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {
                        auto &dispatched_kernel = in_progress_kernels[client_priority];
                        auto &dispatched_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                        auto running = dispatched_kernel.running;

                        if (dispatched_kernel.config.use_original) {
                            // do nothing
                        } else if (dispatched_kernel.config.use_preemptive_ptb) {

                            // if this kernel has been signaled to stop, we will launch it again
                            if (!running) {
                                
                                // Make sure there is no pending event in retreat stream
                                cudaStreamSynchronize(retreat_stream);

                                // set retreat to 0
                                auto ptb_args = client_data.stream_to_ptb_args[dispatched_kernel_wrapper.launch_stream];
                                auto retreat = &(ptb_args->retreat);
                                cudaMemsetAsync(retreat, 0, sizeof(bool), dispatched_kernel_wrapper.launch_stream);
            
                                // Launch the kernel again
                                dispatched_kernel_wrapper.kernel_to_dispatch(dispatched_kernel.config, ptb_args, client_data.curr_idx_arr, nullptr, false, 0, nullptr, nullptr, -1, true);
                            }

                        } else if (dispatched_kernel.config.use_sliced) {

                            // Launch all remaining slices
                            dispatched_kernel.slice_args.launch_idx++;

                            while (dispatched_kernel.slice_args.launch_idx < dispatched_kernel.slice_args.block_offsets.size()) {
                                dispatched_kernel_wrapper.kernel_to_dispatch(dispatched_kernel.config, nullptr, nullptr, &dispatched_kernel.slice_args, false, 0, nullptr, nullptr, -1, true);
                                dispatched_kernel.slice_args.launch_idx++;
                            }

                        }

                        // wait for it to complete
                        cudaStreamSynchronize(dispatched_kernel_wrapper.launch_stream);

                        // Erase it from in-progress
                        in_progress_kernels.erase(client_priority);

                        // mark as launched
                        dispatched_kernel_wrapper.free_args();
                        client_data.queue_size--;
                    }

                    // Always try to fetch and launch kernel
                    succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

                    // all kernels have been launched
                    if (!succeeded) {

                        // query if there is active kernel
                        if (client_event_wrapper.active) {

                            // then we wait until all kernels have completed execution to
                            // potentially consider launching low-priority kernels
                            if (cudaEventQuery(client_event) != cudaSuccess) {
                                break;
                            }
                            
                            // mark event as inactive as all kernels have finished
                            else {
                                client_event_wrapper.active = false;
                                highest_priority_idle_start_t = std::chrono::high_resolution_clock::now();
                            }

                        }
                    }

                } else {

                    // If PRIORITY_MIN_WAIT_TIME_MS > 0, we need to check whether enough idle time is observed
                    if (PRIORITY_MIN_WAIT_TIME_MS > 0.) {
                        auto curr_t = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> elapsed = curr_t - highest_priority_idle_start_t;
                        float idle_time_elapsed = elapsed.count();
                        if (idle_time_elapsed < PRIORITY_MIN_WAIT_TIME_MS) {
                            break;
                        }
                    }

                    // For non-highest-priority client, we only launch at most 1 kernel at any time
                    // therefore we will query whether the kernel has finished

                    // First check whether this client already has a kernel running
                    if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {

                        auto &dispatched_kernel = in_progress_kernels[client_priority];
                        auto &dispatched_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                        auto running = dispatched_kernel.running;
                        auto &config = dispatched_kernel.config;

                        // First check if the kernel is still running
                        // note that the running flag only tells us whether the kernel has been signaled to stop
                        // it is possible that it has not yet terminated
                        if (cudaEventQuery(client_event) != cudaSuccess) {
                            break;
                        }

                        bool finished = false;

                        if (config.use_original) {

                            finished = true;

                        } else if (config.use_preemptive_ptb) {

                            // preemptive kernel was running and now it is finished
                            if (running) {
                                finished = true;
                            }

                        } else if (config.use_sliced) {

                            // all slices have been launched
                            if (dispatched_kernel.slice_args.launch_idx >= dispatched_kernel.slice_args.block_offsets.size() - 1) {
                                finished = true;
                            }

                        }

                        // it was running and now it is finished
                        if (finished) {

                            // Erase from in-progress
                            client_event_wrapper.active = false;
                            dispatched_kernel_wrapper.free_args();
                            in_progress_kernels.erase(client_priority);
                            client_data.queue_size--;

                        // we have told it to stop so we will launch it again
                        } else {

                            if (config.use_sliced) {

                                // point to next slice
                                dispatched_kernel.slice_args.launch_idx++;

                                // Launch the kernel again
                                dispatched_kernel_wrapper.kernel_to_dispatch(config, nullptr, nullptr, &dispatched_kernel.slice_args, false, 0, nullptr, nullptr, -1, true);

                            } else if (config.use_preemptive_ptb) {

                                // Make sure there is no pending event in retreat stream
                                // think about a previous cudaMemsetAsync(&retreat) has not yet completed,
                                // then there may be conflicted writes to retreat
                                cudaStreamSynchronize(retreat_stream);

                                // set retreat to 0
                                auto ptb_args = client_data.stream_to_ptb_args[dispatched_kernel_wrapper.launch_stream];
                                auto retreat = &(ptb_args->retreat);
                                cudaMemsetAsync(retreat, 0, sizeof(bool), dispatched_kernel_wrapper.launch_stream);
            
                                // Launch the kernel again
                                dispatched_kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, nullptr, false, 0, nullptr, nullptr, -1, true);
                            }

                            // Monitor the launched kernel
                            cudaEventRecord(client_event, dispatched_kernel_wrapper.launch_stream);

                            // Flip the flag
                            dispatched_kernel.running = true;

                            client_event_wrapper.active = true;

                            // done
                            break;
                        }
                    }

                    // Try to fetch kernel from low-priority queue
                    succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);
                }

                // Successfully fetched a kernel from the launch queue
                if (succeeded) {

                    // Stop any running kernel
                    for (auto &in_progress_kernel : in_progress_kernels) {

                        auto &dispatched_kernel = in_progress_kernel.second;
                        auto running = dispatched_kernel.running;

                        if (!running) {
                            continue;
                        }

                        auto in_progress_client_id = in_progress_kernel.first.client_id;
                        auto &in_progress_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                        auto &in_progress_client_data = client_data_all[in_progress_client_id];
                        auto &in_progress_kernel_event_wrapper = client_events[in_progress_client_id];
                        auto &in_progress_kernel_event = in_progress_kernel_event_wrapper.event;
                        auto &in_progress_kernel_config = dispatched_kernel.config;

                        // First check whether this kernel has already finished
                        if (!in_progress_kernel_config.use_sliced && cudaEventQuery(in_progress_kernel_event) == cudaSuccess) {

                            // Erase if finished
                            in_progress_kernel_event_wrapper.active = false;
                            in_progress_kernel_wrapper.free_args();
                            in_progress_kernels.erase(in_progress_kernel.first);
                            in_progress_client_data.queue_size--;
                            break;
                        }

                        if (in_progress_kernel_config.use_preemptive_ptb) {

                            // Set retreat flag
                            auto ptb_args = in_progress_client_data.stream_to_ptb_args[in_progress_kernel_wrapper.launch_stream];
                            cudaMemsetAsync(&(ptb_args->retreat), 1, sizeof(bool), retreat_stream);

    #if defined(MEASURE_PREEMPTION_LATENCY)
                            auto start = std::chrono::high_resolution_clock::now();

                            // Fetch progress - this will block as memcpy from device to host
                            uint32_t progress = 0;
                            cudaMemcpyAsync(&progress, &(ptb_args->global_idx), sizeof(uint32_t), cudaMemcpyDeviceToHost, in_progress_kernel_wrapper.launch_stream);

                            auto end = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double, std::milli> duration = end - start;
                            auto preemption_latency_ms = duration.count();

                            auto &launch_call = in_progress_kernel_wrapper.launch_call;
                            auto kernel_name = host_func_to_demangled_kernel_name_map[launch_call.func];
                            auto num_thread_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

                            TALLY_SPD_LOG_ALWAYS("Preempted Kernel " + kernel_name);
                            TALLY_SPD_LOG_ALWAYS("Number thread blocks: " + std::to_string(num_thread_blocks));
                            TALLY_SPD_LOG_ALWAYS("Latency: " + std::to_string(preemption_latency_ms) + "ms");
    #endif
                        }

                        // Mark that this kernel has been signaled to stop
                        dispatched_kernel.running = false;

                        // We know there are at most one kernel running, so break
                        break;
                    }

                    // Now launch the high priority kernel
                    auto config = CudaLaunchConfig::default_config;
                    PTBKernelArgs *ptb_args = nullptr;
                    SlicedKernelArgs *sliced_args = nullptr;

                    if (!is_highest_priority && kernel_wrapper.is_library_call) {
                        TALLY_SPD_WARN("Found library call from low priority job");
                    }

                    auto &launch_call = kernel_wrapper.launch_call;

                    bool use_original_configs = PRIORITY_USE_ORIGINAL_CONFIGS;

                    // PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS let us use original configs for the
                    // best-effort job when a long time has elapsed since the the high priority job is active
                    // Consider a high-priority server which there may be long period of idle time
                    if (PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS > 0.) {
                        auto curr_t = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double, std::milli> elapsed = curr_t - highest_priority_idle_start_t;
                        float idle_time_elapsed = elapsed.count();
                        if (idle_time_elapsed >= PRIORITY_WAIT_TIME_MS_TO_USE_ORIGINAL_CONFIGS) {
                            use_original_configs = true;
                        }
                    }

                    if (!kernel_wrapper.is_library_call && !use_original_configs) {

                        auto is_colocate = client_priority_map.size() > 1;

                        // when not colocating, i.e. profiling, or the job is low-priority
                        // we need to profile or launch preemptive configs
                        if (!is_colocate || !is_highest_priority) {

                            // Do some profiling of the preemptive kernels
                            bool found_in_cache;
                            auto res = get_single_kernel_chosen_config(launch_call, &found_in_cache);

                            if (!found_in_cache) {

                                if (!has_warned && is_colocate) {
                                    TALLY_SPD_WARN("Certain launch configs were not found during job co-location, which might impact experiment accuracy!");
                                    has_warned = true;
                                }

                                priority_launch_and_measure_kernel(kernel_wrapper, client_id);

                                kernel_wrapper.free_args();
                                client_data.queue_size--;
                                break;
                            }

                            // if profiling is done, only launch preemptive configs
                            // when low priority
                            if (!is_highest_priority) {
                                config = res.config;
                            }
                        }
                    }

                    if (!is_highest_priority) {

                        // bookkeep kernel launch if it is not highest-priority 
                        in_progress_kernels[client_priority] = DispatchedKernel(kernel_wrapper, true, config);

                        if (config.use_preemptive_ptb) {
                            // set retreat ang global_idx to 0
                            ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];
                            cudaMemsetAsync(ptb_args, 0, sizeof(PTBKernelArgs), kernel_wrapper.launch_stream);
                        }

                        // Only launch the first slice
                        if (config.use_sliced) {
                            in_progress_kernels[client_priority].slice_args = get_sliced_kernel_args(launch_call.gridDim, config.num_slices);
                            sliced_args = &(in_progress_kernels[client_priority].slice_args);
                            sliced_args->launch_idx = 0;
                        }
                    }

                    // Launch the kernel
                    kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, sliced_args, false, 0, nullptr, nullptr, -1, true);
                    
                    // record event
                    cudaEventRecord(client_event, kernel_wrapper.launch_stream);
                    client_event_wrapper.active = true;

                    // For highest priority, we can directly mark the kernel as launched as we won't ever preempt it
                    if (is_highest_priority) {

                        // Mark as launched
                        kernel_wrapper.free_args();
                        client_data.queue_size--;

                    }

                    break;
                }

                // Did not fetch a kernel
                // Proceed to check the next-priority client

                is_highest_priority = false;
            }
        }
    }
}