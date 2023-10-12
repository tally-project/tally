#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

struct KernelInProgress {
public:
    KernelLaunchWrapper kernel_wrapper;
    bool launched = false;
    CudaLaunchConfig config = CudaLaunchConfig::default_config;
};

// Always launch kernel pairs of the best config
void TallyServer::run_workload_aware_sharing_scheduler()
{
    spdlog::info("Running workload aware sharing scheduler ...");

    srand (time(NULL));

    // Keep in track kernels that are in progress
    std::map<int32_t, KernelInProgress> in_progress_kernels;

    cudaStream_t retreat_stream;
    cudaStreamCreate(&retreat_stream);

    CudaLaunchConfig preemptive_config(false, false, false, true, 4);
    CudaLaunchConfig original_config = CudaLaunchConfig::default_config;

    auto get_single_kernel_result = [this](KernelLaunchWrapper &kernel_wrapper, int32_t client_id) {
        auto &launch_call = kernel_wrapper.launch_call;
        CudaLaunchCallConfigResult res;

        if (kernel_wrapper.is_library_call) {
            res.config = CudaLaunchConfig::default_config;
        } else {
            // Look up cache for best-performance config
            bool found_in_cache;
            res = get_single_kernel_best_config(launch_call, &found_in_cache);

            if (!found_in_cache) {
                auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
                auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

                // Check the latency of this kernel, if it is short, then we fall back to the non-preemtive version
                float latency_ms;
                kernel_wrapper.kernel_to_dispatch(CudaLaunchConfig::default_config, nullptr, nullptr, true, 1000, &latency_ms, nullptr, 1, true);

                auto &client_data = client_data_all[client_id];

                // Some preemptive kernel launch fails with 'invalid argument', fall back to the non-preemtive version for those
                CudaLaunchConfig preemptive_config(false, false, false, true, 4);
                auto err = kernel_wrapper.kernel_to_dispatch(preemptive_config, client_data.global_idx, client_data.retreat, true, 1000, nullptr, nullptr, 1, false);

                if (err) {
                    auto kernel_name = host_func_to_demangled_kernel_name_map[launch_call.func];
                    spdlog::info("Fail to launch preemptive version of kernel " + kernel_name + ". Falling back to non-preemptive versions");
                }

                if (latency_ms < 1 || err) {
                    auto non_preemptive_ptb_configs = CudaLaunchConfig::get_workload_agnostic_sharing_configs(threads_per_block, num_blocks);
                    tune_kernel_launch(kernel_wrapper, client_id, non_preemptive_ptb_configs);
                    res = get_single_kernel_best_config(launch_call, &found_in_cache);
                } else {
                    auto preemptive_ptb_configs = CudaLaunchConfig::get_preemptive_configs(threads_per_block, num_blocks);
                    tune_kernel_launch(kernel_wrapper, client_id, preemptive_ptb_configs);
                    res = get_single_kernel_best_config(launch_call, &found_in_cache);
                }
            }
        }

        return res;
    };

    auto send_preempt_signal_to_kernel = [this, retreat_stream](KernelInProgress &kernel, CudaLaunchConfig const &new_config, int32_t client_id) {
        auto &client_data = client_data_all[client_id];
        auto &kernel_wrapper = kernel.kernel_wrapper;

        // Check if kernel is already launched
        if (kernel.launched) {
            // Only preemptive kernel can be re-launched
            if (!kernel.config.use_preemptive_ptb) {
                throw std::runtime_error("Trying to launch a kernel which has been launched non-preemptively");
            }

            // Same exact config, no need to preempt
            if (kernel.config == new_config) {
                return;
            }

            // Set retreat flag
            cudaMemsetAsync(client_data.retreat, 1, sizeof(bool), retreat_stream);
        }
    };

    auto launch_kernel_with_config = [this, retreat_stream](KernelInProgress &kernel, CudaLaunchConfig const &config, int32_t client_id) {

        auto &client_data = client_data_all[client_id];
        auto &kernel_wrapper = kernel.kernel_wrapper;

        // Check if kernel is already launched
        if (kernel.launched) {

            // Only preemptive kernel can be re-launched
            if (!kernel.config.use_preemptive_ptb) {
                throw std::runtime_error("Trying to launch a kernel which has been launched non-preemptively");
            }

            // Same exact config, simply return
            if (kernel.config == config) {
                return;
            }

            // Fetch the progress
            uint32_t progress = 0;
            cudaMemcpyAsync(&progress, client_data.global_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, kernel_wrapper.launch_stream);

            // Use this to check whether kernel has finished
            auto &launch_call = kernel_wrapper.launch_call;
            auto num_thread_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

            // Wait for kernel to stop
            cudaStreamSynchronize(kernel_wrapper.launch_stream);

            if (progress >= num_thread_blocks) {
                return;
            }
        
        // If never launched before, set global_idx to 0
        } else {
            if (config.use_preemptive_ptb) {
                cudaMemsetAsync(client_data.global_idx, 0, sizeof(uint32_t), kernel_wrapper.launch_stream);
            }
        }

        // Always set retreat to 0 before launch preemptive kernel
        if (config.use_preemptive_ptb) {
            cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), kernel_wrapper.launch_stream);
        }

        // Create a event to monitor the kernel execution
        cudaEventCreateWithFlags(&kernel_wrapper.event, cudaEventDisableTiming);

        // Launch the kernel again
        kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, false, 0, nullptr, nullptr, -1, true);

        // Monitor the launched kernel
        cudaEventRecord(kernel_wrapper.event, kernel_wrapper.launch_stream);

        kernel.launched = true;
        kernel.config = config;
    };

    while (!iox::posix::hasTerminationRequested()) {

        // Flag indicating whether there is new activity
        bool has_change = false;

        for (auto &pair : client_data_all) {

            auto client_id = pair.first;
            auto &client_data = pair.second;

            if (client_data.has_exit) {
                auto client_id = pair.first;
                client_data_all.erase(client_id);
                break;
            }

            bool fetch_new_kernel = false;

            bool has_kernel = in_progress_kernels.find(client_id) != in_progress_kernels.end();

            if (has_kernel) {

                auto &kernel_wrapper = in_progress_kernels[client_id].kernel_wrapper;

                // Check whether has finished
                if (cudaEventQuery(kernel_wrapper.event) == cudaSuccess) {

                    // Erase if finished
                    in_progress_kernels.erase(client_id);
                    client_data.queue_size--;
                    fetch_new_kernel = true;
                    has_change = true;
                }
            } else {
                fetch_new_kernel = true;
            }

            if (fetch_new_kernel) {
                // Try to fetch a new kernel if previous kernel has finished
                KernelLaunchWrapper kernel_wrapper;
                bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

                if (succeeded) {
                    in_progress_kernels[client_id] = KernelInProgress(kernel_wrapper);
                    has_change = true;
                }
            }
            
        }

        if (!has_change) {
            continue;
        }

        int num_kernels = in_progress_kernels.size();

        if (num_kernels == 0) {
            continue;
        }

        // When there is only one kernel to be launched,
        // If the kernel is already running, leave it as it is
        // (Potentially we can pre-empt and re-launch but not sure whether it is worth it)
        // Otherwise, find the best single-kernel performance config and launch it
        else if (num_kernels == 1) {

            auto client_id = in_progress_kernels.begin()->first;
            auto &client_data = client_data_all[client_id];
            auto &kernel = in_progress_kernels.begin()->second;

            if (!kernel.launched) {
                auto &kernel_wrapper = kernel.kernel_wrapper;
                auto res = get_single_kernel_result(kernel_wrapper, client_id);
                auto config = res.config;
                launch_kernel_with_config(kernel, config, client_id);
            }
        }

        // When there are two kernels in the pool,
        // Pre-empt both kernels and re-launch them with the best pair-wise config
        else if (num_kernels == 2) {
            
            // First kernel
            auto first_kernel_ptr = in_progress_kernels.begin();
            auto first_client_id = first_kernel_ptr->first;
            auto &first_kernel = first_kernel_ptr->second;
            auto &first_client_data = client_data_all[first_client_id];
            auto &first_kernel_wrapper = first_kernel.kernel_wrapper;
            auto &first_launch_call = first_kernel_wrapper.launch_call;

            // Second kernel
            auto second_kernel_ptr = std::next(first_kernel_ptr);
            auto second_client_id = second_kernel_ptr->first;
            auto &second_kernel = second_kernel_ptr->second;
            auto &second_client_data = client_data_all[second_client_id];
            auto &second_kernel_wrapper = second_kernel.kernel_wrapper;
            auto &second_launch_call = second_kernel_wrapper.launch_call;

            auto first_kernel_best_res = get_single_kernel_result(first_kernel_wrapper, first_client_id);
            auto second_kernel_best_res = get_single_kernel_result(second_kernel_wrapper, second_client_id);

            auto first_kernel_best_config = first_kernel_best_res.config;
            auto second_kernel_best_config = second_kernel_best_res.config;

            // 1. check whether any of the two kernels has already been launched with non-preemptive config
            // 2. check whether any of the kernel's best config is not preemptive
            //    that happens when either 1. PTB performance is really bad 2. it is short, so best config is non-preemptive
            bool skip_workload_aware_launch = false;

            if ((first_kernel.launched && !first_kernel.config.use_preemptive_ptb) ||
                (second_kernel.launched && !second_kernel.config.use_preemptive_ptb) ||
                !first_kernel_best_config.use_preemptive_ptb ||
                !second_kernel_best_config.use_preemptive_ptb)
            {
                skip_workload_aware_launch = true;
            }

            if (skip_workload_aware_launch) {

                // Launch both kernels if not launched already
                if (!first_kernel.launched) {
                    launch_kernel_with_config(first_kernel, first_kernel_best_config, first_client_id);
                }

                if (!second_kernel.launched) {
                    launch_kernel_with_config(second_kernel, second_kernel_best_config, second_client_id);
                }

                continue;
            }

            // Now, we know both kernels have reasonably good performance with preemptive-PTB
            // Try to find kernel pair best config
            bool found_in_cache;
            auto res = get_kernel_pair_best_config(first_launch_call, second_launch_call, &found_in_cache);

            if (!found_in_cache) {
                tune_kernel_pair_launch(first_kernel_wrapper, second_kernel_wrapper, first_client_id, second_client_id);
                res = get_kernel_pair_best_config(first_launch_call, second_launch_call, &found_in_cache);
            }

            auto pair_launch_config = res.get_configs(first_launch_call, second_launch_call);

            // Check whether time-share
            bool time_share = std::get<2>(pair_launch_config);
            if (time_share) {

                // Choose the shorter kernel to launch with the original config
                auto first_kernel_latency = first_kernel_best_res.metrics.latency_ms;
                auto second_kernel_latency = second_kernel_best_res.metrics.latency_ms;

                if (first_kernel_latency < second_kernel_latency) {
                    launch_kernel_with_config(first_kernel, CudaLaunchConfig::default_config, first_client_id);
                    cudaStreamSynchronize(first_kernel_wrapper.launch_stream);
                } else {
                    launch_kernel_with_config(second_kernel, CudaLaunchConfig::default_config, second_client_id);
                    cudaStreamSynchronize(second_kernel_wrapper.launch_stream);
                }

                continue;
            }

            // Space share
            auto config_1 = std::get<0>(pair_launch_config);
            auto config_2 = std::get<1>(pair_launch_config);

            // set retreat signal
            send_preempt_signal_to_kernel(first_kernel, config_1, first_client_id);
            send_preempt_signal_to_kernel(second_kernel, config_2, second_client_id);

            // Launch with the best pair-wise config
            launch_kernel_with_config(first_kernel, config_1, first_client_id);
            launch_kernel_with_config(second_kernel, config_2, second_client_id);
        }
        
        else {
            throw std::runtime_error("not supported for more than 2 kernels at a time.");
        }
    }
}