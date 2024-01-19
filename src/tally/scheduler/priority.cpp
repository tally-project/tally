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
    std::map<uint32_t, cudaEvent_t> client_events;

    cudaStream_t retreat_stream;
    cudaStreamCreateWithFlags(&retreat_stream, cudaStreamNonBlocking);

    while (!iox::posix::hasTerminationRequested()) {

        bool is_highest_priority = true;

        for (auto &pair : client_priority_map) {

            auto &client_priority = pair.first;
            auto client_id = client_priority.client_id;
            auto &client_data = client_data_all[client_id];

            if (client_events.find(client_id) == client_events.end()) {
                cudaEventCreateWithFlags(&client_events[client_id], cudaEventDisableTiming);
            }

            auto client_event = client_events[client_id];

            if (client_data.has_exit) {
                cudaEventDestroy(client_event);
                
                client_data_all.erase(client_id);
                client_events.erase(client_id);
                client_priority_map.erase(client_priority);
                break;
            }

            KernelLaunchWrapper kernel_wrapper;
            bool succeeded;
        
            if (is_highest_priority) {

                // this job may have been promoted from low to high so it has a kernel in in_progress_kernels
                if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {
                    auto &dispatched_kernel = in_progress_kernels[client_priority];
                    auto &dispatched_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                    auto running = dispatched_kernel.running;

                    // if this kernel has been signaled to stop, we will launch it again
                    if (!running && !dispatched_kernel.config.use_original) {

                        // Make sure there is no pending event in retreat stream
                        cudaStreamSynchronize(retreat_stream);

                        // set retreat to 0
                        auto ptb_args = client_data.stream_to_ptb_args[dispatched_kernel_wrapper.launch_stream];
                        auto retreat = &(ptb_args->retreat);
                        cudaMemsetAsync(retreat, 0, sizeof(bool), dispatched_kernel_wrapper.launch_stream);
    
                        // Launch the kernel again
                        dispatched_kernel_wrapper.kernel_to_dispatch(dispatched_kernel.config, ptb_args, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);
                    }

                    // wait for it to complete
                    cudaStreamSynchronize(dispatched_kernel_wrapper.launch_stream);

                    // Erase it from in-progress
                    in_progress_kernels.erase(client_priority);

                    // mark as launched
                    dispatched_kernel_wrapper.free_args();
                    client_data.queue_size--;
                }

                // Always try to fetch and launch kernel
                succeeded = client_data.kernel_dispatch_queue.try_dequeue(kernel_wrapper);

                // all kernels have been launched
                if (!succeeded) {

                    // then we wait until all kernels have completed execution to
                    // potentially consider launching low-priority kernels
                    if (cudaEventQuery(client_event) != cudaSuccess) {
                        break;
                    }

                }

            } else {
                // For non-highest-priority client, we only launch at most 1 kernel at any time
                // therefore we will query whether the kernel has finished

                // First check whether this client already has a kernel running
                if (in_progress_kernels.find(client_priority) != in_progress_kernels.end()) {

                    auto &dispatched_kernel = in_progress_kernels[client_priority];
                    auto &dispatched_kernel_wrapper = dispatched_kernel.kernel_wrapper;
                    auto running = dispatched_kernel.running;

                    // First check if the kernel is still running
                    // note that the running flag only tells us whether the kernel has been signaled to stop
                    // it is possible that it has yet terminated
                    if (cudaEventQuery(client_event) != cudaSuccess) {
                        break;
                    }

                    // it was running and now it is finished
                    if (running || dispatched_kernel.config.use_original) {

                        // Erase from in-progress
                        dispatched_kernel_wrapper.free_args();
                        in_progress_kernels.erase(client_priority);
                        client_data.queue_size--;

                    // we have told it to stop so we will launch it again
                    } else {

                        auto config = dispatched_kernel.config;

                        // Make sure there is no pending event in retreat stream
                        // think about a previous cudaMemsetAsync(&retreat) has not yet completed,
                        // then there may be conflicted writes to retreat
                        cudaStreamSynchronize(retreat_stream);

                        // set retreat to 0
                        auto ptb_args = client_data.stream_to_ptb_args[dispatched_kernel_wrapper.launch_stream];
                        auto retreat = &(ptb_args->retreat);
                        cudaMemsetAsync(retreat, 0, sizeof(bool), dispatched_kernel_wrapper.launch_stream);
    
                        // Launch the kernel again
                        dispatched_kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);

                        // Monitor the launched kernel
                        cudaEventRecord(client_event, dispatched_kernel_wrapper.launch_stream);

                        // Flip the flag
                        dispatched_kernel.running = true;

                        // done
                        break;
                    }
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
                    auto in_progress_kernel_event = client_events[in_progress_client_id];

                    // First check whether this kernel has already finished
                    if (cudaEventQuery(in_progress_kernel_event) == cudaSuccess) {

                        // Erase if finished
                        in_progress_kernel_wrapper.free_args();
                        in_progress_kernels.erase(in_progress_kernel.first);
                        in_progress_client_data.queue_size--;
                        break;
                    }

                    // Set retreat flag
                    auto ptb_args = in_progress_client_data.stream_to_ptb_args[in_progress_kernel_wrapper.launch_stream];
                    cudaMemsetAsync(&(ptb_args->retreat), 1, sizeof(bool), retreat_stream);

#if defined(MEASURE_PREEMPTION_LATENCY)
                    auto start = std::chrono::high_resolution_clock::now();

                    // Fetch progress - this will block as memcpy from device to host
                    uint32_t progress = 0;
                    cudaMemcpyAsync(&progress, &(ptb_args->global_idx), sizeof(uint32_t), cudaMemcpyDeviceToHost, in_progress_kernel_wrapper.launch_stream);

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

                auto &launch_call = kernel_wrapper.launch_call;

                // Do some profiling of the preemptive kernels
                bool found_in_cache;
                auto res = get_single_kernel_best_config(launch_call, &found_in_cache);

                if (!found_in_cache) {
                    auto threads_per_block = launch_call.blockDim.x * launch_call.blockDim.y * launch_call.blockDim.z;
                    auto num_blocks = launch_call.gridDim.x * launch_call.gridDim.y * launch_call.gridDim.z;

                    auto preemptive_ptb_configs = CudaLaunchConfig::get_preemptive_configs(launch_call, threads_per_block, num_blocks);
                    launch_and_measure_kernel(kernel_wrapper, client_id, preemptive_ptb_configs, USE_PREEMPTIVE_LATENCY_THRESHOLD);

                    kernel_wrapper.free_args();
                    client_data.queue_size--;
                    break;
                }

                PTBArgs *ptb_args = nullptr;

                if (!is_highest_priority) {

                    config = res.config;

                    if (config.use_preemptive_ptb) {
                        // set retreat ang global_idx to 0
                        ptb_args = client_data.stream_to_ptb_args[kernel_wrapper.launch_stream];
                        cudaMemsetAsync(ptb_args, 0, sizeof(PTBArgs), kernel_wrapper.launch_stream);
                    }
                }

                // Launch the kernel
                kernel_wrapper.kernel_to_dispatch(config, ptb_args, client_data.curr_idx_arr, false, 0, nullptr, nullptr, -1, true);
                
                cudaEventRecord(client_event, kernel_wrapper.launch_stream);

                // For highest priority, we can directly mark the kernel as launched as we won't ever preempt it
                if (is_highest_priority) {

                    // Mark as launched
                    kernel_wrapper.free_args();
                    client_data.queue_size--;

                } else {
                    // Otherwise, bookkeep kernel launch if it is not highest-priority 
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