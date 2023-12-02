#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

struct DispatchedKernel {
public:
    KernelLaunchWrapper kernel_wrapper;

    // If true, this kernel is running
    // If false, this kernel has been signaled to stop (may not have stopped)
    bool running = false;
};

// Always prioritize to execute kernels from high-priority job.
// Run low-priority job only when there is no high-priority kernel.
// Preempt low-priority kernel when high-priority kernel arrives.
void TallyServer::run_priority_scheduler()
{
    spdlog::info("Running priority scheduler ...");

    // Keep in track kernels that are in progress
    // The boolean indicates whether the kernel is running/stopped
    std::map<ClientPriority, DispatchedKernel, std::greater<ClientPriority>> in_progress_kernels;

    cudaStream_t retreat_stream;
    cudaStreamCreateWithFlags(&retreat_stream, cudaStreamNonBlocking);

    CudaLaunchConfig preemptive_config(false, false, false, true, 1);
    CudaLaunchConfig original_config = CudaLaunchConfig::default_config;

    while (!iox::posix::hasTerminationRequested()) {

        bool is_highest_priority = true;

        for (auto &pair : client_priority_map) {

            auto &client_priority = pair.first;
            auto client_id = client_priority.client_id;
            auto &client_data = client_data_all[client_id];

            if (client_data.has_exit) {
                client_data_all.erase(client_id);
                client_priority_map.erase(client_priority);
                break;
            }

            // First check whether this client already has a kernel running
            if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {

                auto &dispatched_kernel = in_progress_kernels[client_priority];
                auto &in_progress_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                auto running = dispatched_kernel.running;

                // is running
                if (running) {

                    // Check whether has finished
                    if (cudaEventQuery(in_progress_kernel_wrapper.event) == cudaSuccess) {

                        // Erase if finished
                        in_progress_kernels.erase(client_priority);
                        client_data.queue_size--;

                    }

                // has been attempted to stop
                } else {

                    CudaLaunchConfig config = original_config;

                    if (!is_highest_priority) {

                        // Make sure there is no pending event in retreat stream
                        // think about a previous cudaMemsetAsync(&retreat) has not yet completed,
                        // then there may be conflicted writes to retreat
                        cudaStreamSynchronize(retreat_stream);

                        // set retreat to 0
                        cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), in_progress_kernel_wrapper.launch_stream);

                        config = preemptive_config;
                    }

                    // Create a event to monitor the kernel execution
                    cudaEventCreateWithFlags(&in_progress_kernel_wrapper.event, cudaEventDisableTiming);

                    // Launch the kernel again
                    in_progress_kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);

                    // Monitor the launched kernel
                    cudaEventRecord(in_progress_kernel_wrapper.event, in_progress_kernel_wrapper.launch_stream);

                    // Flip the flag
                    in_progress_kernels[client_priority].running = true;

                }

                // Either way, break the loop
                break;

            }

            // If this client does not have a kernel in progress, try fetching from the queue
            KernelLaunchWrapper kernel_wrapper;
            bool succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

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

                    // First check whether this kernel has already finished
                    if (cudaEventQuery(in_progress_kernel_wrapper.event) == cudaSuccess) {

                        // Erase if finished
                        in_progress_kernels.erase(in_progress_kernel.first);
                        in_progress_client_data.queue_size--;
                        break;
                    }

                    // Set retreat flag
                    cudaMemsetAsync(in_progress_client_data.retreat, 1, sizeof(bool), retreat_stream);

                    // Mark that this kernel has been signaled to stop
                    dispatched_kernel.running = false;

                    // We know there are at most one kernel running, so break
                    break;
                }

                // Now launch the high priority kernel
                CudaLaunchConfig config = original_config;

                if (!is_highest_priority) {

                    // set retreat ang global_idx to 0
                    cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), kernel_wrapper.launch_stream);
                    cudaMemsetAsync(client_data.global_idx, 0, sizeof(uint32_t), kernel_wrapper.launch_stream);
                    
                    config = preemptive_config;
                }

                // std::cout << "Launched a new kernel" << std::endl;

                // Create a event to monitor the kernel execution
                cudaEventCreateWithFlags(&kernel_wrapper.event, cudaEventDisableTiming);

                kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);

                cudaEventRecord(kernel_wrapper.event, kernel_wrapper.launch_stream);

                // Bookkeep this kernel launch
                in_progress_kernels[client_priority].kernel_wrapper = kernel_wrapper;
                in_progress_kernels[client_priority].running = true;

                break;
            }

            // Did not fetch a kernel
            // Proceed to check the next-priority client

            is_highest_priority = false;
        }
    }
}