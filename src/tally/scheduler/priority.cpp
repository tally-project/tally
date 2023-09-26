#include <cassert>
#include <cfloat>
#include <random>

#include "spdlog/spdlog.h"

#include <tally/generated/server.h>
#include <tally/cuda_util.h>
#include <tally/util.h>
#include <tally/env.h>

// Always prioritize to execute kernels from high-priority job.
// Run low-priority job only when there is no high-priority kernel.
// Preempt low-priority kernel when high-priority kernel arrives.
void TallyServer::run_priority_scheduler()
{
    spdlog::info("Running priority scheduler ...");

    // Keep in track kernels that are in progress
    // The boolean indicates whether the kernel is running/stopped
    std::map<ClientPriority, std::pair<KernelLaunchWrapper, bool>, std::greater<ClientPriority>> in_progress_kernels;

    cudaStream_t retreat_stream;
    cudaStreamCreate(&retreat_stream);

    CudaLaunchConfig preemptive_config(false, false, false, true, 4);
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

                auto &in_progress_kernel_wrapper = in_progress_kernels[client_priority].first;
                auto running = in_progress_kernels[client_priority].second;

                // is running
                if (running) {

                    // Check whether has finished
                    if (cudaEventQuery(in_progress_kernel_wrapper.event) == cudaSuccess) {

                        // Erase if finished
                        in_progress_kernels.erase(client_priority);
                        client_data.queue_size--;

                    }

                // is stopped
                } else {

                    CudaLaunchConfig config = original_config;

                    if (!is_highest_priority) {
                        // set retreat to 0
                        cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), in_progress_kernel_wrapper.launch_stream);

                        config = preemptive_config;
                    }

                    // Create a event to monitor the kernel execution
                    cudaEventCreateWithFlags(&in_progress_kernel_wrapper.event, cudaEventDisableTiming);

                    // Launch the kernel again
                    in_progress_kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, false, 0, nullptr, nullptr, -1);

                    // Monitor the launched kernel
                    cudaEventRecord(in_progress_kernel_wrapper.event, in_progress_kernel_wrapper.launch_stream);

                    // Flip the flag
                    in_progress_kernels[client_priority].second = true;

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

                    auto running = in_progress_kernel.second.second;

                    if (!running) {
                        continue;
                    }

                    auto in_progress_client_id = in_progress_kernel.first.client_id;
                    auto &in_progress_kernel_wrapper = in_progress_kernel.second.first;
                    auto &in_progress_client_data = client_data_all[in_progress_client_id];

                    // Set retreat flag
                    cudaMemsetAsync(in_progress_client_data.retreat, 1, sizeof(bool), retreat_stream);
                    
                    // Fetch progress
                    uint32_t progress = 0;
                    cudaMemcpyAsync(&progress, in_progress_client_data.global_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, in_progress_kernel_wrapper.launch_stream);

                    auto &launch_call = in_progress_kernel_wrapper.launch_call;
                    auto num_thread_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

                    // Wait for kernel to stop
                    cudaStreamSynchronize(in_progress_kernel_wrapper.launch_stream);

                    if (progress >= num_thread_blocks) {

                        // Remove from bookkeeping
                        in_progress_kernels.erase(in_progress_kernel.first);
                        in_progress_client_data.queue_size--;

                    } else {

                        // Mark as stopped
                        in_progress_kernel.second.second = false;
                    }

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

                kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, false, 0, nullptr, nullptr, -1);

                cudaEventRecord(kernel_wrapper.event, kernel_wrapper.launch_stream);

                // Bookkeep this kernel launch
                in_progress_kernels[client_priority].first = kernel_wrapper;
                in_progress_kernels[client_priority].second = true;

                break;
            }

            // Did not fetch a kernel
            // Proceed to check the next-priority client

            is_highest_priority = false;
        }
    }
}