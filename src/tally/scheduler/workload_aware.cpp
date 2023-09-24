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

    auto get_single_kernel_config = [this](KernelLaunchWrapper &kernel_wrapper, int32_t client_id) {
        auto &launch_call = kernel_wrapper.launch_call;

        // Look up cache for best-performance config
        bool found_in_cache;
        auto res = get_single_kernel_best_config(launch_call, &found_in_cache);

        if (!found_in_cache) {
            auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
            auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;
            auto configs = CudaLaunchConfig::get_preemptive_configs(threads_per_block, num_blocks);

            tune_kernel_launch(kernel_wrapper, client_id, configs);
            res = get_single_kernel_best_config(launch_call, &found_in_cache);
        }

        auto config = res.config;

        return config;
    };

    auto launch_kernel_with_config = [this, retreat_stream](KernelInProgress &kernel, CudaLaunchConfig const &config, int32_t client_id) {

        auto &client_data = client_data_all[client_id];
        auto &kernel_wrapper = kernel.kernel_wrapper;

        // Check if kernel is already launched
        if (kernel.launched) {
            if (!kernel.config.use_preemptive_ptb) {
                throw std::runtime_error("Trying to launch a kernel which has been launched non-preemptively");
            }

            // Set retreat flag
            cudaMemsetAsync(client_data.retreat, 1, sizeof(bool), retreat_stream);
            
            uint32_t progress = 0;
            cudaMemcpyAsync(&progress, client_data.global_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, kernel_wrapper.launch_stream);

            // Wait for kernel to stop
            cudaStreamSynchronize(kernel_wrapper.launch_stream);
        }

        if (config.use_preemptive_ptb) {
            // cudaStreamSynchronize(kernel_wrapper.launch_stream);
            cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), kernel_wrapper.launch_stream);
            cudaMemsetAsync(client_data.global_idx, 0, sizeof(uint32_t), kernel_wrapper.launch_stream);
        }

        // Create a event to monitor the kernel execution
        cudaEventCreateWithFlags(&kernel_wrapper.event, cudaEventDisableTiming);

        // Launch the kernel again
        kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, false, 0, nullptr, nullptr, -1);

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

        if (!has_change ) {
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
                auto config = get_single_kernel_config(kernel_wrapper, client_id);
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

            // auto first_kernel_best_config = get_single_kernel_config(first_kernel_wrapper, first_client_id);
            // auto second_kernel_best_config = get_single_kernel_config(second_kernel_wrapper, second_client_id);

            // First, check whether any of the two kernels has already been launched with original config
            // Then there is nothing we can do much
            bool found_original_launch = false;

            if ((first_kernel.launched && first_kernel.config.use_original) ||
                (second_kernel.launched && second_kernel.config.use_original))
            {
                found_original_launch = true;
            }

            if (found_original_launch) {

                // Launch both kernels if not launched already
                if (!first_kernel.launched) {
                    auto best_config = get_single_kernel_config(first_kernel_wrapper, first_client_id);
                    launch_kernel_with_config(first_kernel, best_config, first_client_id);
                }

                if (!second_kernel.launched) {
                    auto best_config = get_single_kernel_config(second_kernel_wrapper, second_client_id);
                    launch_kernel_with_config(second_kernel, best_config, second_client_id);
                }

                continue;
            }

            // Then, check whether any of the kernel has original config
            // that is, the performance of preemptive-PTB is really bad

            auto first_kernel_best_config = get_single_kernel_config(first_kernel_wrapper, first_client_id);
            auto second_kernel_best_config = get_single_kernel_config(second_kernel_wrapper, second_client_id);

            if (first_kernel_best_config.use_original || second_kernel_best_config.use_original) {

                // Launch both kernels if not launched already
                if (!first_kernel.launched) {
                    launch_kernel_with_config(first_kernel, first_kernel_best_config, first_client_id);
                }

                if (!second_kernel.launched) {
                    launch_kernel_with_config(second_kernel, second_kernel_best_config, second_client_id);
                }

                continue;
            }

            // Now, we know both kernels 
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
                // Choose one of the two kernels to launch
                int index = rand() % 2;

                if (index == 0) {
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

            launch_kernel_with_config(first_kernel, config_1, first_client_id);
            launch_kernel_with_config(second_kernel, config_2, second_client_id);
        }
        
        else {
            throw std::runtime_error("not supported for more than 2 kernels at a time.");
        }
    }
}