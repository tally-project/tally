#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

// Launch kernel as soon as they arrive
void TallyServer::run_naive_scheduler()
{
    spdlog::info("Running naive scheduler ...");

    CudaLaunchConfig config = CudaLaunchConfig::default_config;
    // CudaLaunchConfig config = CudaLaunchConfig(false, true, false, false, 4);

    KernelLaunchWrapper kernel_wrapper;

    cudaStream_t retreat_stream;
    cudaStreamCreate(&retreat_stream);

    while (!iox::posix::hasTerminationRequested()) {

        for (auto &pair : client_data_all) {

            auto &client_data = pair.second;

            if (client_data.has_exit) {
                auto client_id = pair.first;
                client_data_all.erase(client_id);
                break;
            }

            bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

            if (succeeded) {

                if (config.use_dynamic_ptb || config.use_preemptive_ptb) {
                    // Make Sure the previous kernel has finished
                    cudaStreamSynchronize(kernel_wrapper.launch_stream);
                    cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), kernel_wrapper.launch_stream);
                    cudaMemsetAsync(client_data.global_idx, 0, sizeof(uint32_t), kernel_wrapper.launch_stream);
                }
                
                kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);
                client_data.queue_size--;
            }
        }
    }
}