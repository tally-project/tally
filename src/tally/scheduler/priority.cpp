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

    CudaLaunchConfig config;
};

// Always prioritize to execute kernels from high-priority job.
// Run low-priority job only when there is no high-priority kernel.
// Preempt low-priority kernel when high-priority kernel arrives.

// For high-priority, we launch as many kernels as there are in the queue
// For low-priority, we at most launch one kernel
void TallyServer::run_priority_scheduler()
{
    TALLY_SPD_LOG_ALWAYS("Running priority scheduler ...");

    // Keep in track low-priority kernels that are in progress
    // The boolean indicates whether the kernel is running/stopped
    std::map<ClientPriority, DispatchedKernel, std::greater<ClientPriority>> in_progress_kernels;

    cudaEvent_t highest_priority_event = nullptr;

    cudaStream_t retreat_stream;
    cudaStreamCreateWithFlags(&retreat_stream, cudaStreamNonBlocking);

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

            KernelLaunchWrapper kernel_wrapper;
            bool succeeded;
        
            if (is_highest_priority) {

                // Always try to fetch and launch kernel
                succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

                // all kernels have been launched
                if (!succeeded) {

                    // then we wait until all kernels have completed execution to
                    // potentially consider launching low-priority kernels
                    if (cudaEventQuery(highest_priority_event) != cudaSuccess) {
                        break;
                    }

                }

            } else {
                // For non-highest-priority client, we only launch at most 1 kernel at any time
                // therefore we will query whether the kernel has finished

                // First check whether this client already has a kernel running
                if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {

                    auto &dispatched_kernel = in_progress_kernels[client_priority];
                    auto &in_progress_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                    auto running = dispatched_kernel.running;

                    // First check if the kernel is still running
                    // note that the running flag only tells us whether the kernel has been signaled to stop
                    // it is possible that it has yet terminated
                    if (cudaEventQuery(in_progress_kernel_wrapper.event) != cudaSuccess) {
                        break;
                    }

                    // Destroy the previously created event since the kernel has completed
                    cudaEventDestroy(in_progress_kernel_wrapper.event);

                    // it was running and now it is finished
                    if (running || dispatched_kernel.config.use_original) {

                        // Erase from in-progress
                        in_progress_kernels.erase(client_priority);
                        client_data.queue_size--;

                    // we have told it to stop so we will launch it again
                    } else {

                        auto &launch_call = in_progress_kernel_wrapper.launch_call;

                        bool found_in_cache;
                        auto res = get_single_kernel_best_config(launch_call, &found_in_cache);

                        if (!found_in_cache) {
                            throw std::runtime_error("must be found in cache");
                        }

                        auto config = CudaLaunchConfig::default_config;

                        if (!is_highest_priority) {

                            // Make sure there is no pending event in retreat stream
                            // think about a previous cudaMemsetAsync(&retreat) has not yet completed,
                            // then there may be conflicted writes to retreat
                            cudaStreamSynchronize(retreat_stream);

                            // set retreat to 0
                            cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), in_progress_kernel_wrapper.launch_stream);

                            config = res.config;
                        }

                        // Create a event to monitor the kernel execution
                        cudaEventCreateWithFlags(&in_progress_kernel_wrapper.event, cudaEventDisableTiming);

                        // Launch the kernel again
                        in_progress_kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);

                        // Monitor the launched kernel
                        cudaEventRecord(in_progress_kernel_wrapper.event, in_progress_kernel_wrapper.launch_stream);

                        // Flip the flag
                        in_progress_kernels[client_priority].running = true;
                        in_progress_kernels[client_priority].config = config;

                    }

                    // Either way, break the loop
                    break;
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

                    // First check whether this kernel has already finished
                    if (cudaEventQuery(in_progress_kernel_wrapper.event) == cudaSuccess) {

                        // Erase if finished
                        in_progress_kernels.erase(in_progress_kernel.first);
                        in_progress_client_data.queue_size--;
                        break;
                    }

                    // Set retreat flag
                    cudaMemsetAsync(in_progress_client_data.retreat, 1, sizeof(bool), retreat_stream);

#if defined(MEASURE_PREEMPTION_LATENCY)
                    auto start = std::chrono::high_resolution_clock::now();

                    // Fetch progress - this will block as memcpy from device to host
                    uint32_t progress = 0;
                    cudaMemcpyAsync(&progress, in_progress_client_data.global_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, in_progress_kernel_wrapper.launch_stream);

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
                    // Mark that this kernel has been signaled to stop
                    dispatched_kernel.running = false;

                    // We know there are at most one kernel running, so break
                    break;
                }

                // Now launch the high priority kernel
                auto config = CudaLaunchConfig::default_config;

                if (is_highest_priority) {

                    if (highest_priority_event) {
                        cudaEventDestroy(highest_priority_event);
                    }

                    // Create a new event to monitor the highest-priority kernel execution
                    cudaEventCreateWithFlags(&highest_priority_event, cudaEventDisableTiming);

                } else {

                    auto &launch_call = kernel_wrapper.launch_call;

                    // Do some profiling of the preemptive kernels
                    bool found_in_cache;
                    auto res = get_single_kernel_best_config(launch_call, &found_in_cache);

                    if (!found_in_cache) {
                        auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
                        auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

                        auto preemptive_ptb_configs = CudaLaunchConfig::get_preemptive_configs(threads_per_block, num_blocks);
                        tune_kernel_launch(kernel_wrapper, client_id, preemptive_ptb_configs);
                        res = get_single_kernel_best_config(launch_call, &found_in_cache);
                    }

                    if (config.use_preemptive_ptb) {
                        // set retreat ang global_idx to 0
                        cudaMemsetAsync(client_data.retreat, 0, sizeof(bool), kernel_wrapper.launch_stream);
                        cudaMemsetAsync(client_data.global_idx, 0, sizeof(uint32_t), kernel_wrapper.launch_stream);
                    }

                    config = res.config;

                    // Create a event to monitor the kernel execution
                    cudaEventCreateWithFlags(&kernel_wrapper.event, cudaEventDisableTiming);
                }



                // Launch the kernel
                kernel_wrapper.kernel_to_dispatch(config, client_data.global_idx, client_data.retreat, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);
                
                // For highest priority, we can directly mark the kernel as launched as we won't ever preempt it
                if (is_highest_priority) {

                    client_data.queue_size--;
                    cudaEventRecord(highest_priority_event, kernel_wrapper.launch_stream);

                } else {
                    // Otherwise, bookkeep kernel launch if it is not highest-priority 
                    cudaEventRecord(kernel_wrapper.event, kernel_wrapper.launch_stream);

                    in_progress_kernels[client_priority] = DispatchedKernel(kernel_wrapper, true, config);
                }

                break;
            }

            // Did not fetch a kernel
            // Proceed to check the next-priority client

            is_highest_priority = false;
        }
    }
}