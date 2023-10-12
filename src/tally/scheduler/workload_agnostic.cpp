#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

// Launch PTB-based kernel to always allow sharing
// Does not consider pair-wise performance
// simply launch the single-kernel best config
void TallyServer::run_workload_agnostic_sharing_scheduler()
{
    spdlog::info("Running workload agnostic sharing scheduler ...");

    KernelLaunchWrapper kernel_wrapper;

    cudaStream_t retreat_stream;
    cudaStreamCreate(&retreat_stream);

    while (!iox::posix::hasTerminationRequested()) {
        
        for (auto &pair : client_data_all) {

            auto &client_data = pair.second;
            auto client_id = pair.first;

            if (client_data.has_exit) {
                client_data_all.erase(client_id);
                break;
            }

            // Try fetch kernel from queue
            bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

            if (succeeded) {

                CudaLaunchConfig config = CudaLaunchConfig::default_config;

                if (!kernel_wrapper.is_library_call) {
                    
                    auto launch_call = kernel_wrapper.launch_call;

                    // Look up cache for best-performance config
                    bool found_in_cache;
                    auto res = get_single_kernel_best_config(launch_call, &found_in_cache);

                    if (!found_in_cache) {

                        auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
                        auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;
                        auto configs = CudaLaunchConfig::get_workload_agnostic_sharing_configs(threads_per_block, num_blocks);

                        tune_kernel_launch(kernel_wrapper, client_id, configs);
                        res = get_single_kernel_best_config(launch_call, &found_in_cache);
                        assert(found_in_cache);
                    }

                    config = res.config;

                    if (config.use_dynamic_ptb || config.use_preemptive_ptb) {
                        // Make Sure the previous kernel has finished
                        cudaStreamSynchronize(kernel_wrapper.launch_stream);
                        cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), kernel_wrapper.launch_stream);
                        cudaMemsetAsync(client_data.global_idx, 0, sizeof(uint32_t), kernel_wrapper.launch_stream);
                    }
                }

                kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, false, 0, nullptr, nullptr, -1, true);
                client_data.queue_size--;
            }
        }
    }
}
