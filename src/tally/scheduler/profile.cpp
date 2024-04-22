#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

void TallyServer::run_profile_scheduler()
{
    TALLY_SPD_LOG_ALWAYS("Running profile scheduler ...");

    KernelLaunchWrapper kernel_wrapper;

    cudaStream_t retreat_stream;
    cudaStreamCreate(&retreat_stream);

    std::vector<float> kernel_durations;
    int count = 0;

    while (!iox::posix::hasTerminationRequested()) {

        for (auto &pair : client_data_all) {

            auto &client_data = pair.second;
            auto client_id = pair.first;

            if (client_data.has_exit) {
                client_data_all.erase(client_id);
                break;
            }

            bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

            if (succeeded) {
                cudaDeviceSynchronize();
                auto start = std::chrono::high_resolution_clock::now();

                kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
                client_data.queue_size--;

                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> elapsed = end - start;
                float time_elapsed = elapsed.count();

                // skip the first 10000 kernels for warmup
                if (count > 10000) {
                    kernel_durations.push_back(time_elapsed);
                } else {
                    count++;
                }
            }
        }
    }

    std::ofstream result_file("kernel_durations.txt");

    if (result_file.is_open()) {
        for (float value : kernel_durations) {
            result_file << value << "\n";
        }
        result_file.close();
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
}

// void TallyServer::run_profile_scheduler()
// {
//     TALLY_SPD_LOG_ALWAYS("Running profile scheduler ...");

//     KernelLaunchWrapper kernel_wrapper;

//     cudaStream_t retreat_stream;
//     cudaStreamCreate(&retreat_stream);

//     while (!iox::posix::hasTerminationRequested()) {

//         for (auto &pair : client_data_all) {

//             auto &client_data = pair.second;
//             auto client_id = pair.first;

//             if (client_data.has_exit) {
//                 client_data_all.erase(client_id);
//                 break;
//             }

//             bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

//             if (succeeded) {
//                 CudaLaunchConfig config = CudaLaunchConfig::default_config;

//                 if (!kernel_wrapper.is_library_call) {
                    
//                     auto launch_call = kernel_wrapper.launch_call;
//                     auto kernel_name = host_func_to_demangled_kernel_name_map[launch_call.func];

//                     // Some kernels will fail if launched many times
//                     if (containsSubstring(kernel_name, "DeviceRadixSortOnesweepKernel") ||
//                         containsSubstring(kernel_name, "krn_partial_segment_offset") ||
//                         containsSubstring(kernel_name, "compute_grad_weight") ||
//                         containsSubstring(kernel_name, "sum_and_scatter")
//                     ) {

//                     } else {

//                         // Look up cache for best-performance config
//                         bool found_in_cache;
//                         auto res = get_single_kernel_chosen_config(launch_call, &found_in_cache);

//                         if (!found_in_cache) {

//                             auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
//                             auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;
//                             auto configs = CudaLaunchConfig::get_profile_configs(launch_call);

//                             tune_kernel_launch(kernel_wrapper, client_id, configs);
//                             res = get_single_kernel_chosen_config(launch_call, &found_in_cache);
//                         }

//                         config = res.config;
//                     }
//                 }

//                 if (config.use_dynamic_ptb || config.use_preemptive_ptb) {
//                     auto ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];
//                     cudaMemsetAsync(ptb_args, 0, sizeof(PTBKernelArgs), kernel_wrapper.launch_stream);
//                     kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, nullptr, false, 0, nullptr, nullptr, -1, true);
//                 } else {
//                     kernel_wrapper.kernel_to_dispatch(config, nullptr, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
//                 }

//                 client_data.queue_size--;
//             }
//         }
//     }
// }

// void TallyServer::run_profile_scheduler()
// {
//     TALLY_SPD_LOG_ALWAYS("Running profile scheduler ...");

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, 1);

//     while (!iox::posix::hasTerminationRequested()) {

//         // Wait until there are two kernels from two clients
//         while (!iox::posix::hasTerminationRequested()) {
//             int kernel_count = 0;

//             for (auto &pair : client_data_all) {
//                 auto &client_data = pair.second;
//                 if (client_data.kernel_dispatch_queue.peek()) {
//                     kernel_count++;
//                 }
//             }

//             if (kernel_count >= 2) {
//                 assert(kernel_count == 2);

//                 for (auto &pair : client_data_all) {
//                     auto &client_data = pair.second;
//                     bool random_skip = dis(gen);

//                     // Add some randomness to shuffle the kernel pairs
//                     if (random_skip) {
//                         KernelLaunchWrapper kernel_wrapper;
//                         bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

//                         if (!succeeded) {
//                             throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": try_dequeue failed.");
//                         }

//                         kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
//                         client_data.queue_size--;
//                         kernel_count--;
//                     }
//                 }

//                 if (kernel_count == 2) {
//                     break;
//                 }
//             }

//             std::this_thread::sleep_for(std::chrono::milliseconds(10));
//         }

//         int index = 0;

//         PTBKernelArgs *ptb_args_arrs[2];
//         uint32_t *curr_idx_arrs[2];
//         KernelLaunchWrapper kernel_wrappers[2];
//         CudaLaunchCall launch_calls[2];
//         std::string kernel_names[2];

//         // Kernel Metadata (num registers, shared memory, etc.)
//         CudaLaunchMetadata launch_calls_meta_original[2];
//         CudaLaunchMetadata launch_calls_meta_ptb[2];

//         // Will set this to the max(100 * single run duration) for each kernel
//         // This makes sure the measurement error is within 1%
//         // This is in seconds
//         float profile_duration = 0.;
//         bool found_library_call = false;

//         // Launch both kernel - to warm up and in case all experiments are cached already
//         for (auto &pair : client_data_all) {
//             auto &client_data = pair.second;

//             bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrappers[index]);

//             if (!succeeded) {
//                 throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": try_dequeue failed.");
//             }

//             float time_elapsed;
//             kernel_wrappers[index].kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, true, 1000, &time_elapsed, nullptr, 1, true);

//             if (kernel_wrappers[index].is_library_call) {
//                 found_library_call = true;
//                 break;
//             }

//             auto ptb_args = client_data.stream_to_ptb_args[kernel_wrappers[index].launch_stream];
//             ptb_args_arrs[index] = ptb_args;
//             curr_idx_arrs[index] = client_data.curr_idx_arr;

//             launch_calls[index] = kernel_wrappers[index].launch_call;
//             auto func_addr = launch_calls[index].func;
            
//             kernel_names[index] = host_func_to_demangled_kernel_name_map[func_addr];

//             launch_calls_meta_original[index] = original_kernel_map[func_addr].meta_data;
//             launch_calls_meta_ptb[index] = ptb_kernel_map[func_addr].meta_data;

//             launch_calls_meta_original[index].dynamic_shmem_size_bytes = kernel_wrappers[index].dynamic_shmem_size_bytes;
//             launch_calls_meta_ptb[index].dynamic_shmem_size_bytes = kernel_wrappers[index].dynamic_shmem_size_bytes;

//             profile_duration = std::max(profile_duration, (100 * time_elapsed) / 1000.f);

//             index++;
//         }

//         if (found_library_call) {
//             continue;
//         }

//         // At least run for 1 sec
//         profile_duration = std::max(profile_duration, 1.f);

//         // Maybe don't exceed 1 minute;
//         profile_duration = std::min(profile_duration, 60.f);

//         // Preparation
//         auto k1_blockDim = launch_calls[0].blockDim;
//         auto k2_blockDim = launch_calls[1].blockDim;

//         auto k1_gridDim = launch_calls[0].gridDim;
//         auto k2_gridDim = launch_calls[1].gridDim;

//         auto k1_configs = CudaLaunchConfig::get_profile_configs(launch_calls[0], k1_blockDim.x * k1_blockDim.y * k1_blockDim.z, k1_gridDim.x * k1_gridDim.y * k1_gridDim.z);
//         auto k2_configs = CudaLaunchConfig::get_profile_configs(launch_calls[1], k2_blockDim.x * k2_blockDim.y * k2_blockDim.z, k2_gridDim.x * k2_gridDim.y * k2_gridDim.z);
//         auto k1_k2_configs = std::vector<std::vector<CudaLaunchConfig>> {k1_configs, k2_configs};

//         auto launch_kernel_func = [kernel_wrappers, ptb_args_arrs, curr_idx_arrs](int idx, CudaLaunchConfig config, float dur_seconds, float *time_elapsed, float *iters, CUresult *err, int32_t total_iters) {
//             *err = (kernel_wrappers[idx].kernel_to_dispatch)(config, ptb_args_arrs[idx], curr_idx_arrs[idx], true, dur_seconds, time_elapsed, iters, total_iters, true);
//         };

//         CUresult errs[2];

//         // Launch once, in case all results are cached already
//         errs[0] = (kernel_wrappers[0].kernel_to_dispatch)(CudaLaunchConfig::default_config, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
//         errs[1] = (kernel_wrappers[1].kernel_to_dispatch)(CudaLaunchConfig::default_config, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);

//         // We will be collecting two things:
//         //    1. single-kernel performance under different launch configs
//         //    2. kernel-pair performance under different launch configs

//         // Step 1: get the single kernel performance baselines
//         float base_latency_ms[2];

//         // This fixed workload is from counting how many iterations of both kernel run for a duration
//         // Namely, the workload consists of a number of iterations for both kernels respectively
//         // We will use this artifical workload to compute a speedup number of pair-wise performance against MPS
//         int32_t fixed_workload_iters[2] = { 0, 0 };

//         for (int i = 0; i < 2; i++) {

//             bool found_in_cache = false;

//             auto res = get_single_kernel_perf(launch_calls[i], CudaLaunchConfig::default_config, &found_in_cache);
//             if (!found_in_cache) {

//                 float iters;
//                 float time_elapsed;

//                 cudaDeviceSynchronize();

//                 launch_kernel_func(i, CudaLaunchConfig::default_config, profile_duration, &time_elapsed, &iters, &(errs[i]), -1);

//                 // compute latency
//                 float latency_ms = time_elapsed / iters;

//                 // save result
//                 set_single_kernel_perf(launch_calls[i], CudaLaunchConfig::default_config, launch_calls_meta_original[i], 1., latency_ms, iters);

//                 // query again
//                 res = get_single_kernel_perf(launch_calls[i], CudaLaunchConfig::default_config, &found_in_cache);

//                 assert(found_in_cache);
//             }

//             auto metrics = res.metrics;
//             base_latency_ms[i] = metrics.latency_ms;
//             fixed_workload_iters[i] = metrics.iters;
//         }

//         // Step 2: get the single kernel performance for various configs

//         // The best config will not include the baseline config
//         // Otherwise the baseline is always the best

//         bool skip_colocate_exp = false;
//         float best_latency_ms[2] = { FLT_MAX, FLT_MAX };
//         bool has_new_config[2] = { false, false };
//         CudaLaunchCallConfigResult best_configs[2];

//         for (int i = 0; i < 2; i++) {

//             auto configs = k1_k2_configs[i];

//             for (auto &config : configs) {

//                 if (config == CudaLaunchConfig::default_config) {
//                     continue;
//                 }

//                 bool found_in_cache = false;

//                 auto res = get_single_kernel_perf(launch_calls[i], config, &found_in_cache);
//                 if (!found_in_cache) {

//                     float iters;
//                     float time_elapsed;

//                     cudaDeviceSynchronize();

//                     launch_kernel_func(i, config, profile_duration, &time_elapsed, &iters, &(errs[i]), -1);

//                     // compute latency
//                     float latency_ms = time_elapsed / iters;
//                     float norm_speed = base_latency_ms[i] / latency_ms;

//                     // Some kernels experiment 100x slowdown after applying PTB
//                     // for those we cannot run fixed workload experiment, so will skip them
//                     if (norm_speed < 0.05) {
//                         std::cerr << "Kernel " << kernel_names[i] << " speed after applying PTB is less than 20% of the original speed" << "\n";
//                         std::cerr << "Skipping kernel-pair co-located experiments" << std::endl;
//                         skip_colocate_exp = true;
//                     }

//                     // save result
//                     set_single_kernel_perf(launch_calls[i], config, launch_calls_meta_ptb[i], norm_speed, latency_ms, iters);

//                     // query again
//                     res = get_single_kernel_perf(launch_calls[i], config, &found_in_cache);

//                     assert(found_in_cache);
//                     has_new_config[i] = true;
//                 }

//                 auto metrics = res.metrics;
//                 float latency_ms = metrics.latency_ms;

//                 if (latency_ms < best_latency_ms[i]) {
//                     best_configs[i] = res;
//                     best_latency_ms[i] = latency_ms;
//                 }
//             }

//             if (has_new_config[i]) {
//                 set_single_kernel_chosen_config(launch_calls[i], best_configs[i]);
//             }
//         }

//         // Step 3: get the kernel pair performance for various configs
//         float best_sum_norm_speed = -1.;
//         CudaLaunchCallConfigPairResult best_pair_config;
//         bool has_new_pair_config = false;
//         float fixed_workload_latency_mps = -1.;

//         if (!skip_colocate_exp) {
        
//             for (auto &k1_config : k1_configs) {
//                 for (auto &k2_config : k2_configs) {
//                     CudaLaunchMetadata meta_data_1;
//                     CudaLaunchMetadata meta_data_2;

//                     if (k1_config == CudaLaunchConfig::default_config) {
//                         meta_data_1 = launch_calls_meta_original[0];
//                     } else {
//                         meta_data_1 = launch_calls_meta_ptb[0];
//                     }

//                     if (k2_config == CudaLaunchConfig::default_config) {
//                         meta_data_2 = launch_calls_meta_original[1];
//                     } else {
//                         meta_data_2 = launch_calls_meta_ptb[1];
//                     }

//                     bool found_in_cache = false;
//                     auto res = get_kernel_pair_perf(launch_calls[0], launch_calls[1], k1_config, k2_config, &found_in_cache);

//                     if (!found_in_cache) {

//                         // First experiment - Get colocated latency and norm speed for each kernel
//                         cudaDeviceSynchronize();

//                         float iters[2];
//                         float time_elapsed[2];

//                         std::thread launch_t_1(launch_kernel_func, 0, k1_config, profile_duration, &(time_elapsed[0]), &(iters[0]), &(errs[0]), -1);
//                         std::thread launch_t_2(launch_kernel_func, 1, k2_config, profile_duration, &(time_elapsed[1]), &(iters[1]), &(errs[1]), -1);

//                         launch_t_1.join();
//                         launch_t_2.join();

//                         if (std::abs(time_elapsed[0] - time_elapsed[1]) > 0.03 * std::min(time_elapsed[0], time_elapsed[1])) {
//                             std::cerr << "Warning: two jobs do not finish at around the same time" << "\n";
//                             std::cerr << "time_elapsed_1: " << time_elapsed[0] << " time_elapsed_2: " << time_elapsed[1] << std::endl;
//                         }

//                         float k1_latency_ms = time_elapsed[0] / iters[0];
//                         float k2_latency_ms = time_elapsed[1] / iters[1];

//                         float k1_norm_speed = base_latency_ms[0] / k1_latency_ms;
//                         float k2_norm_speed = base_latency_ms[1] / k2_latency_ms;

//                         // Second experiment - Run fixed workload experiment
//                         cudaDeviceSynchronize();

//                         auto start = std::chrono::high_resolution_clock::now();

//                         std::thread fixed_workload_t_1(launch_kernel_func, 0, k1_config, 10000, nullptr, nullptr, &(errs[0]), fixed_workload_iters[0]);
//                         std::thread fixed_workload_t_2(launch_kernel_func, 1, k2_config, 10000, nullptr, nullptr, &(errs[1]), fixed_workload_iters[1]);

//                         fixed_workload_t_1.join();
//                         fixed_workload_t_2.join();

//                         auto end = std::chrono::high_resolution_clock::now();
//                         std::chrono::duration<double, std::milli> duration = end - start;

//                         float fixed_workload_latency = duration.count();
//                         float fixed_workload_speedup = fixed_workload_latency_mps / fixed_workload_latency;

//                         // Set the fixed workload baseline if it is MPS
//                         if (k1_config == CudaLaunchConfig::default_config && k2_config == CudaLaunchConfig::default_config) {
//                             fixed_workload_latency_mps = fixed_workload_latency;
//                         }

//                         assert(fixed_workload_latency_mps > 0);

//                         // Third Experiment: Let's run an unfair workload
//                         // Use the transformed config's finished iterations as the workload

//                         float unfair_workload_speedup = 0.;
//                         float unfair_workload_latency = 0.;

//                         // Only do for non-MPS, since MPS is the baseline
//                         if ((k1_config != CudaLaunchConfig::default_config) || (k2_config != CudaLaunchConfig::default_config)) {
                            
//                             cudaDeviceSynchronize();

//                             auto start = std::chrono::high_resolution_clock::now();

//                             // Run on the default config
//                             std::thread unfair_workload_t_1(launch_kernel_func, 0, CudaLaunchConfig::default_config, 10000, nullptr, nullptr, &(errs[0]), iters[0]);
//                             std::thread unfair_workload_t_2(launch_kernel_func, 1, CudaLaunchConfig::default_config, 10000, nullptr, nullptr, &(errs[1]), iters[1]);

//                             unfair_workload_t_1.join();
//                             unfair_workload_t_2.join();

//                             auto end = std::chrono::high_resolution_clock::now();
//                             std::chrono::duration<double, std::milli> duration = end - start;

//                             unfair_workload_latency = duration.count();
//                             unfair_workload_speedup = unfair_workload_latency / (std::max(time_elapsed[0], time_elapsed[1]));
//                         }

//                         // Save the results
//                         set_kernel_pair_perf(
//                             launch_calls[0], launch_calls[1], k1_config, k2_config, meta_data_1, meta_data_2,
//                             k1_norm_speed, k2_norm_speed, k1_latency_ms, k2_latency_ms,
//                             fixed_workload_latency, fixed_workload_speedup,
//                             unfair_workload_latency, unfair_workload_speedup
//                         );
                        
//                         res = get_kernel_pair_perf(launch_calls[0], launch_calls[1], k1_config, k2_config, &found_in_cache);
//                         assert(found_in_cache);

//                         has_new_pair_config = true;
//                     }

//                     // Set the fixed workload baseline if it is MPS
//                     if (k1_config == CudaLaunchConfig::default_config && k2_config == CudaLaunchConfig::default_config) {
//                         fixed_workload_latency_mps = res.fixed_workload_perf.latency_ms;
//                     }

//                     float sum_norm_speed = res.get_sum_norm_speed();
//                     if (sum_norm_speed > best_sum_norm_speed) {
//                         best_sum_norm_speed = sum_norm_speed;
//                         best_pair_config = res;
//                     }
//                 }
//             }

//             if (has_new_pair_config) {
//                 set_kernel_pair_best_config(launch_calls[0], launch_calls[1], best_pair_config);
//             }
//         }
        
//         // Mark kernel as launched
//         for (auto &pair : client_data_all) {
//             auto &client_data = pair.second;
//             client_data.queue_size--;
//         }
//     }
// }