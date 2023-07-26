#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cassert>

#include <cuda.h>

#include <tally/env.h>
#include <tally/cuda_launch.h>
#include <tally/transform.h>
#include <tally/daemon.h>

const CudaLaunchConfig CudaLaunchConfig::default_config = CudaLaunchConfig();

std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config)
{
    os << "CudaLaunchConfig: ";
    if (config.use_original) {
        os << "original";
    } else if (config.use_sliced) {
        os << "sliced: use_cuda_graph: ";
        if (config.use_cuda_graph) {
            os << "true ";
        } else {
            os << "false ";
        }
        os << "threads_per_slice: " << config.threads_per_slice;
    } else if (config.use_ptb) {
        os << "PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    }
    return os;
}

// return (time, iterations)
std::pair<float, float> CudaLaunchConfig::repeat_launch(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream, float dur_seconds, uint32_t max_count)
{
    cudaDeviceSynchronize();
    float time_ms;

    // get a rough estimate of the kernel duration
    launch(func, gridDim, blockDim, args, sharedMem, stream, true, &time_ms);

    uint64_t sync_interval = std::max((uint64_t)((dur_seconds * 1000.) / time_ms), 1ul);

    auto startTime = std::chrono::steady_clock::now();
    uint64_t ckpt_count = 0;
    uint64_t count = 0;
    uint64_t elapsed_ns = 0;

    while (true) {

        // Perform your steps here
        launch(func, gridDim, blockDim, args, sharedMem, stream);
        count++;
        ckpt_count++;

        // Avoid launching too many kernels
        if (ckpt_count == sync_interval) {
            cudaDeviceSynchronize();
            ckpt_count = 0;
        }

        auto currentTime = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
        if ((max_count > 0 && count >= max_count) || ((double) elapsed_ns) / 1e9 >= dur_seconds) {
            cudaDeviceSynchronize();
            auto currentTime = std::chrono::steady_clock::now();
            elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
            break;
        }

    }

    return std::make_pair(((double)elapsed_ns / 1e6), (double)count);
}

void profile_kernel(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream)
{
    CudaLaunchCall launch_call(func, gridDim, blockDim);
    CudaLaunchCallConfig base_call_config(
        launch_call,
        CudaLaunchConfig()
    );
    float baseline_time_ms = TallyDaemon::daemon->get_execution_time(base_call_config);

    // Skip if base performance not exists
    if (baseline_time_ms > 0) {

        // Use this to identify the idx of the kernel launch
        auto curr_kernel_idx = TallyDaemon::daemon->curr_kernel_idx;

        if (curr_kernel_idx == PROFILE_KERNEL_IDX) {

            bool use_original = PROFILE_USE_ORIGINAL;
            bool use_sliced = PROFILE_USE_SLICED;
            bool use_ptb = PROFILE_USE_PTB;
            bool use_cuda_graph = PROFILE_USE_CUDA_GRAPH;
            uint32_t threads_per_slice = PROFILE_THREADS_PER_SLICE;
            uint32_t num_blocks_per_sm = PROFILE_NUM_BLOCKS_PER_SM;

            CudaLaunchConfig profile_config(use_original, use_sliced, use_ptb, use_cuda_graph, threads_per_slice, num_blocks_per_sm);
            std::cout << "Profiling config: " << profile_config << std::endl;
            auto time_iters = profile_config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, PROFILE_DURATION_SECONDS);

            std::cout << "Kernel: " << TallyDaemon::daemon->host_func_to_demangled_kernel_name_map[func] << std::endl;
            std::cout << "\tblockDim: " << blockDim.x << " " << blockDim.y << " " << blockDim.z << std::endl;
            std::cout << "\tgridDim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;
            float baseline_tp = 1 / baseline_time_ms;
            float new_tp = time_iters.second / time_iters.first;

            std::cout << "Time: " << time_iters.first << std::endl;
            std::cout << "Iters: " << time_iters.second << std::endl;
            std::cout << "baseline_tp: " << baseline_tp << std::endl;
            std::cout << "new_tp: " << new_tp << std::endl;

            std::cout << "Normalized throughput: " << new_tp / baseline_tp << std::endl;

            exit(0);

        } else {
            curr_kernel_idx++;
        }
    } else {
        std::cout << "Baseline performance does not exist, skipping.." << std::endl;
    }
}

CudaLaunchConfig CudaLaunchConfig::tune(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    CudaLaunchCall launch_call(func, gridDim, blockDim);
    CudaLaunchConfig best_config;
    std::vector<CudaLaunchConfig> candidates;

    float best_time_ms = std::numeric_limits<float>::max();
    float time_ms;
    float base_time_ms;

    auto kernel_name = TallyDaemon::daemon->host_func_to_demangled_kernel_name_map[func];

    std::cout << "[Profile result]" <<std::endl;
    std::cout << "\tKernel: " << kernel_name << std::endl;
    std::cout << "\tblockDim: " << blockDim.x << " " << blockDim.y << " " << blockDim.z << std::endl;
    std::cout << "\tgridDim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;

    // default config - use_original=true
    CudaLaunchConfig base_config;
    CudaLaunchCallConfig base_call_config(launch_call, base_config);

    // warmup first
    base_config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, 1, 10000);

    base_time_ms = TallyDaemon::daemon->get_execution_time(base_call_config);
    if (base_time_ms <= 0) {

        auto res = base_config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, 1, 10000);
        base_time_ms = res.first / res.second;
        TallyDaemon::daemon->set_execution_time(base_call_config, base_time_ms);
    }

    std::cout << "\tBaseline: Time: " << base_time_ms << std::endl;

    if (USE_TRANSFORM) {

        uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
        uint32_t total_threads = gridDim.x * gridDim.y * gridDim.z * threads_per_block;

        // some sliced configs
        for (uint32_t _threads_per_block : { 129560, 161280, 174080, 184320, 196608 }) {
            // for (bool _use_cuda_graph : { true, false }) {
            for (bool _use_cuda_graph : { false }) {
                CudaLaunchConfig config(false, true, false, _use_cuda_graph, _threads_per_block);
                candidates.push_back(config);
            }
        }

        // some PTB configs
        uint32_t _num_blocks_per_sm = 1;
        while(_num_blocks_per_sm * threads_per_block <= 1024) {
            CudaLaunchConfig config(false, false, true, false, 0, _num_blocks_per_sm);
            candidates.push_back(config);
            _num_blocks_per_sm++;
        }
        
        for (auto &config : candidates) {

            CudaLaunchCallConfig call_config(launch_call, config);
            time_ms = TallyDaemon::daemon->get_execution_time(call_config);

            if (time_ms <= 0) {
                auto res = config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, 1, 10000);
                time_ms = res.first / res.second;

                std::cout << "\t" << config << " Time: " << time_ms << std::endl;
                TallyDaemon::daemon->set_execution_time(call_config, time_ms);
            }

            if (time_ms < best_time_ms) {
                best_config = config;
                best_time_ms = time_ms;
            }
        }

        if (best_time_ms >= USE_TRANSFORM_THRESHOLD * base_time_ms) {
            best_config = base_config;
        }
    } else {
        best_config = base_config;
    }

    std::cout << "Choosen: " << best_config << std::endl;
    TallyDaemon::daemon->set_launch_config(launch_call, best_config);
    TallyDaemon::daemon->save_performance_cache();

    return best_config;
}

cudaError_t CudaLaunchConfig::launch(
    const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream,
    bool run_profile, float *elapsed_time_ms)
{
    cudaEvent_t _start, _stop;

    if (use_original) {

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaDeviceSynchronize();

            cudaEventRecord(_start);
        }

        auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        return err;
    } else if (use_sliced) {

        uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;

        auto cu_func = TallyDaemon::daemon->sliced_kernel_map[func].first;
        auto num_args = TallyDaemon::daemon->sliced_kernel_map[func].second;

        assert(cu_func);

        CudaGraphCall *cuda_graph_call = nullptr;

        cudaStream_t _stream;

        if (use_cuda_graph) {

            // Try to use cuda graph first 
            for (auto *call : TallyDaemon::daemon->cuda_graph_vec) {
                if (call->equals(func, args, num_args - 1, gridDim, blockDim)) {
                    cuda_graph_call = call;
                    break;
                }
            }

            if (!cuda_graph_call) {
                // Otherwise, construct one
                cuda_graph_call = new CudaGraphCall(func, args, num_args - 1, gridDim, blockDim);
                TallyDaemon::daemon->cuda_graph_vec.push_back(cuda_graph_call);
            }

            _stream = TallyDaemon::daemon->stream;
        } else {
            _stream = stream;
        }

        dim3 new_grid_dim;
        dim3 blockOffset(0, 0, 0);

        uint32_t num_blocks = (threads_per_slice + threads_per_block - 1) / threads_per_block;
        if (num_blocks <= gridDim.x) {
            new_grid_dim = dim3(num_blocks, 1, 1);
        } else {
            uint32_t num_blocks_y = (num_blocks + gridDim.x - 1) / gridDim.x;
            if (num_blocks_y <= gridDim.y) {
                new_grid_dim = dim3(gridDim.x, num_blocks_y, 1);
            } else {
                uint32_t num_blocks_z = (num_blocks_y + gridDim.y - 1) / gridDim.y;
                new_grid_dim = dim3(gridDim.x, gridDim.y, std::min(num_blocks_z, gridDim.z));
            }
        }

        if (use_cuda_graph) {
            cudaStreamBeginCapture(TallyDaemon::daemon->stream, cudaStreamCaptureModeGlobal);
        } else {

            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaDeviceSynchronize();

                cudaEventRecord(_start);
            }
        }

        CUresult res;
        while (blockOffset.x < gridDim.x && blockOffset.y < gridDim.y && blockOffset.z < gridDim.z) {

            void *KernelParams[num_args];
            for (size_t i = 0; i < num_args - 1; i++) {
                KernelParams[i] = args[i];
            }
            KernelParams[num_args - 1] = &blockOffset;

            // This ensure that you won't go over the original grid size
            dim3 curr_grid_dim (
                std::min(gridDim.x - blockOffset.x, new_grid_dim.x),
                std::min(gridDim.y - blockOffset.y, new_grid_dim.y),
                std::min(gridDim.z - blockOffset.z, new_grid_dim.z)
            );

            res = lcuLaunchKernel(cu_func, curr_grid_dim.x, curr_grid_dim.y, curr_grid_dim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, _stream, KernelParams, NULL);

            if (res != CUDA_SUCCESS) {
                std::cerr << "Encountering res != CUDA_SUCCESS" << std::endl;
                return cudaErrorInvalidValue;
            }

            blockOffset.x += new_grid_dim.x;

            if (blockOffset.x >= gridDim.x) {
                blockOffset.x = 0;
                blockOffset.y += new_grid_dim.y;

                if (blockOffset.y >= gridDim.y) {
                    blockOffset.y = 0;
                    blockOffset.z += new_grid_dim.z;
                }
            }
        }

        if (use_cuda_graph) {
            cudaStreamEndCapture(TallyDaemon::daemon->stream, &(cuda_graph_call->graph));

            if (!cuda_graph_call->instantiated) {
                cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                cuda_graph_call->instantiated = true;
            } else {
                // graph already exists; try to apply changes
                cudaGraphExecUpdateResult update;

                std::cout << "try to update" << std::endl;
                
                try {

                    if (cudaGraphExecUpdate(cuda_graph_call->instance, cuda_graph_call->graph, NULL, &update) != cudaSuccess) 
                    {
                        cudaGraphExecDestroy(cuda_graph_call->instance);
                        cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                        std::cout << "update fail" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << "catched exception" << std::endl;
                }
            }

            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaDeviceSynchronize();

                cudaEventRecord(_start);
            }

            auto res = cudaGraphLaunch(cuda_graph_call->instance, stream);

            if (run_profile) {
                cudaEventRecord(_stop);
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
                cudaEventDestroy(_start);
                cudaEventDestroy(_stop);
            }

            return res;
        } else {

            if (run_profile) {
                cudaEventRecord(_stop);
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
                cudaEventDestroy(_start);
                cudaEventDestroy(_stop);
            }

            return cudaSuccess;
        }
    } else if (use_ptb) {

        auto cu_func = TallyDaemon::daemon->ptb_kernel_map[func].first;
        auto num_args = TallyDaemon::daemon->ptb_kernel_map[func].second;

        assert(cu_func);

        // Depend on number of PTBs/SM
        dim3 PTB_grid_dim(82 * num_blocks_per_sm);

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 1; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 1] = &gridDim;

        if (run_profile) {
            CHECK_CUDA_ERROR(cudaEventCreate(&_start));
            CHECK_CUDA_ERROR(cudaEventCreate(&_stop));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            CHECK_CUDA_ERROR(cudaEventRecord(_start));
        }

        auto res = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (run_profile) {
            CHECK_CUDA_ERROR(cudaEventRecord(_stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(_stop));
            CHECK_CUDA_ERROR(cudaEventElapsedTime(elapsed_time_ms, _start, _stop));
            CHECK_CUDA_ERROR(cudaEventDestroy(_start));
            CHECK_CUDA_ERROR(cudaEventDestroy(_stop));
        }

        if (res != CUDA_SUCCESS) {
            std::cerr << "Encountering res != CUDA_SUCCESS" << std::endl;
            return cudaErrorInvalidValue;
        }

        return cudaSuccess;
        
    } else {
        throw std::runtime_error("Invalid launch config.");
    }
}