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
    TALLY_SPD_LOG_ALWAYS("Running workload agnostic sharing scheduler ...");

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
                        auto configs = CudaLaunchConfig::get_workload_agnostic_sharing_configs(launch_call, threads_per_block, num_blocks);

                        launch_and_measure_kernel(kernel_wrapper, client_id, configs, USE_PTB_THRESHOLD);
                        client_data.queue_size--;
                        continue;
                    }

                    config = res.config;
                }

                if (config.use_dynamic_ptb || config.use_preemptive_ptb) {
                    auto ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];
                    cudaMemsetAsync(ptb_args, 0, sizeof(PTBArgs), kernel_wrapper.launch_stream);
                    kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);
                } else {
                    kernel_wrapper.kernel_to_dispatch(config, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
                }

                client_data.queue_size--;
            }
        }
    }
}
