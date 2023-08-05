#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <cassert>
#include <type_traits>

#include <cuda.h>

#include <tally/env.h>
#include <tally/cuda_launch.h>
#include <tally/transform.h>
#include <tally/generated/server.h>

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

std::vector<CudaLaunchConfig> CudaLaunchConfig::get_configs(uint32_t threads_per_block, uint32_t num_blocks)
{
    std::vector<CudaLaunchConfig> configs;

    configs.push_back(CudaLaunchConfig::default_config);

    // if (threads_per_block * num_blocks > 129560) {

        // some sliced configs
        // for (uint32_t _threads_per_block : { 129560, 161280, 174080, 184320, 196608 }) {
        //     // for (bool _use_cuda_graph : { true, false }) {
        //     for (bool _use_cuda_graph : { false }) {
        //         CudaLaunchConfig config(false, true, false, _use_cuda_graph, _threads_per_block);
        //         configs.push_back(config);
        //     }
        // }

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(_num_blocks_per_sm * threads_per_block <= 1024 && _num_blocks_per_sm * 82 <= num_blocks) {
        CudaLaunchConfig config(false, false, true, false, 0, _num_blocks_per_sm);
        configs.push_back(config);
        _num_blocks_per_sm++;
    }

    // }
    
    return configs;
}


// Instantiate template
template
CUresult CudaLaunchConfig::repeat_launch<const void *>(const void *, dim3, dim3, void **, size_t, cudaStream_t, float, float *, float *, uint32_t);

template
CUresult CudaLaunchConfig::repeat_launch<CUfunction>(CUfunction, dim3, dim3, void **, size_t, cudaStream_t, float, float *, float *, uint32_t);

// return (time, iterations)
template <typename T>
CUresult CudaLaunchConfig::repeat_launch(
    T func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream,
    float dur_seconds, float *time_ms, float *iters, uint32_t max_count)
{
    float _time_ms;
    CUresult err;

    // get a rough estimate of the kernel duration
    err = launch(func, gridDim, blockDim, args, sharedMem, stream, true, &_time_ms);

    uint64_t sync_interval = std::max((uint64_t)((dur_seconds * 1000.) / _time_ms) / 2, 1ul);

    auto startTime = std::chrono::steady_clock::now();
    uint64_t ckpt_count = 0;
    uint64_t count = 0;
    uint64_t elapsed_ns = 0;

    while (true) {

        // Perform your steps here
        err = launch(func, gridDim, blockDim, args, sharedMem, stream);
        count++;
        ckpt_count++;

        // Avoid launching too many kernels
        if (ckpt_count == sync_interval) {
            cudaStreamSynchronize(stream);
            ckpt_count = 0;
        }

        auto currentTime = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
        if ((max_count > 0 && count >= max_count) || ((double) elapsed_ns) / 1e9 >= dur_seconds) {
            cudaStreamSynchronize(stream);
            auto currentTime = std::chrono::steady_clock::now();
            elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
            break;
        }

    }

    if (time_ms) *time_ms = (double)elapsed_ns / 1e6;
    if (iters) *iters = count;

    return err;
}

CudaLaunchConfig CudaLaunchConfig::tune(const void * func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t  stream)
{
    CudaLaunchCall launch_call(func, gridDim, blockDim);
    CudaLaunchConfig best_config;
    std::vector<CudaLaunchConfig> candidates;

    float best_time_ms = std::numeric_limits<float>::max();
    float time_ms;
    float base_time_ms;

    auto kernel_name = TallyServer::server->host_func_to_demangled_kernel_name_map[func];

    std::cout << "[Profile result]" <<std::endl;
    std::cout << "\tKernel: " << kernel_name << std::endl;
    std::cout << "\tblockDim: " << blockDim.x << " " << blockDim.y << " " << blockDim.z << std::endl;
    std::cout << "\tgridDim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << std::endl;

    // default config - use_original=true
    CudaLaunchConfig base_config;
    CudaLaunchCallConfig base_call_config(launch_call, base_config);

    // warmup first
    base_config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, 1, nullptr, nullptr, 10000);

    if (base_time_ms <= 0) {

        float _time_ms;
        float iters;

        base_config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, 1, &_time_ms, &iters, 10000);
        base_time_ms = _time_ms / iters;
    }

    std::cout << "\tBaseline: Time: " << base_time_ms << std::endl;

    if (USE_TRANSFORM) {

        uint32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
        uint32_t num_blocks = gridDim.x * gridDim.y * gridDim.z;
        candidates = get_configs(threads_per_block, num_blocks);
        
        for (auto &config : candidates) {

            CudaLaunchCallConfig call_config(launch_call, config);

            if (time_ms <= 0) {
                float _time_ms;
                float iters;
                config.repeat_launch(func, gridDim, blockDim, args, sharedMem, stream, 1, &_time_ms, &iters, 10000);
                time_ms = _time_ms / iters;

                std::cout << "\t" << config << " Time: " << time_ms << std::endl;
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
    // TallyServer::server->save_performance_cache();

    return best_config;
}

// Instantiate template
template
CUresult CudaLaunchConfig::launch<const void *>(const void *, dim3, dim3, void **, size_t, cudaStream_t, bool, float *);

template
CUresult CudaLaunchConfig::launch<CUfunction>(CUfunction, dim3, dim3, void **, size_t, cudaStream_t, bool, float *);

void checkCudaErrors(CUresult err) {
  assert(err == CUDA_SUCCESS);
}

template <typename T>
CUresult CudaLaunchConfig::launch(
    T func, dim3  gridDim, dim3  blockDim, void ** args, size_t  sharedMem, cudaStream_t stream,
    bool run_profile, float *elapsed_time_ms)
{
    cudaEvent_t _start, _stop;

    if (use_original) {

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaStreamSynchronize(stream);

            cudaEventRecord(_start);
        }

        CUfunction cu_func;

        if constexpr (std::is_same<T, const void *>::value) {
            assert(TallyServer::server->original_kernel_map.find(func) != TallyServer::server->original_kernel_map.end());
            cu_func = TallyServer::server->original_kernel_map[func].first;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = func;
            // std::cout << "cu_func: " << cu_func << std::endl;
        } else {
            throw std::runtime_error("Unsupported typename");
        }

        assert(cu_func);

        size_t num_args;
        
        if (std::is_same<T, const void *>::value) {
            num_args = TallyServer::server->sliced_kernel_map[func].second;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            num_args = TallyServer::server->_jit_kernel_addr_to_args[func].size();
        } else {
            throw std::runtime_error("Unsupported typename");
        }

        auto err = lcuLaunchKernel(cu_func, gridDim.x, gridDim.y, gridDim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, NULL);

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

        CUfunction cu_func;
        size_t num_args;
        
        if (std::is_same<T, const void *>::value) {
            cu_func = TallyServer::server->sliced_kernel_map[func].first;
            num_args = TallyServer::server->sliced_kernel_map[func].second;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = TallyServer::server->jit_sliced_kernel_map[func];
            num_args = TallyServer::server->_jit_kernel_addr_to_args[func].size();
        } else {
            throw std::runtime_error("Unsupported typename");
        }

        assert(cu_func);

        CudaGraphCall *cuda_graph_call = nullptr;

        cudaStream_t _stream;

        if (std::is_same<T, const void *>::value) {
            if (use_cuda_graph) {

                // Try to use cuda graph first 
                for (auto *call : TallyServer::server->cuda_graph_vec) {
                    if (call->equals(func, args, num_args - 1, gridDim, blockDim)) {
                        cuda_graph_call = call;
                        break;
                    }
                }

                if (!cuda_graph_call) {
                    // Otherwise, construct one
                    cuda_graph_call = new CudaGraphCall(func, args, num_args - 1, gridDim, blockDim);
                    TallyServer::server->cuda_graph_vec.push_back(cuda_graph_call);
                }

                _stream = TallyServer::server->stream;
            } else {
                _stream = stream;
            }
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
            cudaStreamBeginCapture(TallyServer::server->stream, cudaStreamCaptureModeGlobal);
        } else {

            if (run_profile) {
                cudaEventCreate(&_start);
                cudaEventCreate(&_stop);
                cudaStreamSynchronize(stream);

                cudaEventRecord(_start);
            }
        }

        CUresult err;
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

            err = lcuLaunchKernel(cu_func, curr_grid_dim.x, curr_grid_dim.y, curr_grid_dim.z,
                                blockDim.x, blockDim.y, blockDim.z, sharedMem, _stream, KernelParams, NULL);

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
            cudaStreamEndCapture(TallyServer::server->stream, &(cuda_graph_call->graph));

            if (!cuda_graph_call->instantiated) {
                cudaGraphInstantiate(&(cuda_graph_call->instance), cuda_graph_call->graph, NULL, NULL, 0);
                cuda_graph_call->instantiated = true;
            } else {
                // graph already exists; try to apply changes
                cudaGraphExecUpdateResultInfo update;

                std::cout << "try to update" << std::endl;
                
                try {

                    if (cudaGraphExecUpdate(cuda_graph_call->instance, cuda_graph_call->graph, &update) != cudaSuccess) 
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
                cudaStreamSynchronize(stream);

                cudaEventRecord(_start);
            }

            auto graph_launch_err = cudaGraphLaunch(cuda_graph_call->instance, stream);

            if (run_profile) {
                cudaEventRecord(_stop);
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
                cudaEventDestroy(_start);
                cudaEventDestroy(_stop);
            }

            if (graph_launch_err) {
                throw std::runtime_error("cudaGraphLaunch returns error code!");
            }

            return CUDA_SUCCESS;
        } else {

            if (run_profile) {
                cudaEventRecord(_stop);
                cudaEventSynchronize(_stop);
                cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
                cudaEventDestroy(_start);
                cudaEventDestroy(_stop);
            }

            return err;
        }
    } else if (use_ptb) {

        CUfunction cu_func;
        size_t num_args;

        if (std::is_same<T, const void *>::value) {
            cu_func = TallyServer::server->ptb_kernel_map[func].first;
            num_args = TallyServer::server->ptb_kernel_map[func].second;
        } else if constexpr (std::is_same<T, CUfunction>::value) {
            cu_func = TallyServer::server->jit_ptb_kernel_map[func];
            num_args = TallyServer::server->_jit_kernel_addr_to_args[func].size();
        } else {
            throw std::runtime_error("Unsupported typename");
        }

        assert(cu_func);

        // Depend on number of PTBs/SM
        dim3 PTB_grid_dim(82 * num_blocks_per_sm);

        void *KernelParams[num_args];
        for (size_t i = 0; i < num_args - 1; i++) {
            KernelParams[i] = args[i];
        }
        KernelParams[num_args - 1] = &gridDim;

        if (run_profile) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaStreamSynchronize(stream);

            cudaEventRecord(_start);
        }

        auto err = lcuLaunchKernel(cu_func, PTB_grid_dim.x, PTB_grid_dim.y, PTB_grid_dim.z,
                              blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, KernelParams, NULL);

        if (run_profile) {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(elapsed_time_ms, _start, _stop);
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        return err;
        
    } else {
        throw std::runtime_error("Invalid launch config.");
    }
}