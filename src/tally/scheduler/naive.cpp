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
    TALLY_SPD_LOG_ALWAYS("Running naive scheduler ...");

    CudaLaunchConfig config = CudaLaunchConfig::default_config;
    // CudaLaunchConfig config = CudaLaunchConfig(false, false, false, true, 4);

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
                    auto ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];
                    cudaMemsetAsync(ptb_args, 0, sizeof(PTBArgs), kernel_wrapper.launch_stream);
                    kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);
                } else {
                    kernel_wrapper.kernel_to_dispatch(config, nullptr, nullptr, false, 0, nullptr, nullptr, -1, true);
                }

                kernel_wrapper.free_args();
                
                client_data.queue_size--;
            }
        }
    }
}