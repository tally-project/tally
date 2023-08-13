#ifndef TALLY_CUDA_LAUNCH_H
#define TALLY_CUDA_LAUNCH_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <tally/env.h>

// Used at runtime as key to launch configuration
struct CudaLaunchCall {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;

    CudaLaunchCall(){}

    CudaLaunchCall(const void *func, dim3 gridDim, dim3 blockDim) :
        func(func),
        gridDim(gridDim),
        blockDim(blockDim)
    {}

    bool operator==(const CudaLaunchCall &other) const
    {

        bool res = (func == other.func
                    && gridDim.x == other.gridDim.x
                    && gridDim.y == other.gridDim.y
                    && gridDim.z == other.gridDim.z
                    && blockDim.x == other.blockDim.x
                    && blockDim.y == other.blockDim.y
                    && blockDim.z == other.blockDim.z);
        return res;
    }
};

// TODO: rewrite this hash function
template <>
struct std::hash<CudaLaunchCall>
{
    std::size_t operator()(const CudaLaunchCall& k) const
    {
        auto _hash = std::hash<const void *>()(k.func);
        return _hash;
    }
};

struct CudaLaunchCallMeta {
    // Set at compile time
    int max_threads_per_block;

    // Can be set at runtime
    int static_shmem_size_bytes;

    // Set at compile time
    int num_regs;

    // Can be set at runtime
    int max_dynamic_shmem_size_bytes;

    // Not sure what this is but it can be set at runtime
    // Include here just to keep in mind
    int preferred_shmem_carveout;
};

struct CudaLaunchCallPair {
    CudaLaunchCall launch_call_1;
    CudaLaunchCall launch_call_2;

    bool operator==(const CudaLaunchCallPair &other) const
    {
        return (launch_call_1 == other.launch_call_1 && launch_call_2 == other.launch_call_2) ||
               (launch_call_1 == other.launch_call_2 && launch_call_2 == other.launch_call_1);
    }
};

template <>
struct std::hash<CudaLaunchCallPair>
{
    std::size_t operator()(const CudaLaunchCallPair& k) const
    {
        auto _hash = std::hash<CudaLaunchCall>()(k.launch_call_1) |
                     std::hash<CudaLaunchCall>()(k.launch_call_2);
        return _hash;
    }
};

class CudaGraphCall {

public:
    const void *_host_func;
    std::vector<void *> _args;
    dim3 _gridDim;
    dim3 _blockDim;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool instantiated = false;

    CudaGraphCall(const void * host_func, void **args, size_t nargs, dim3 gridDim, dim3 blockDim)
    {
        _host_func = host_func;
        for (size_t i = 0; i < nargs; i++) {
            _args.push_back(args[i]);
        }
        _gridDim = gridDim;
        _blockDim = blockDim;
    }

    bool equals(const void * host_func, void **args, size_t nargs, dim3 gridDim, dim3 blockDim) {
        if (host_func != _host_func || _args.size() != nargs ||
            gridDim.x != _gridDim.x || gridDim.y != _gridDim.y ||  gridDim.z != _gridDim.z || 
            blockDim.x != _blockDim.x || blockDim.y != _blockDim.y || blockDim.z != _blockDim.z)
        {
            return false;
        }

        for (size_t i = 0; i < nargs; i++) {
            if (args[i] != _args[i]) {
                return false;
            }
        }

        return true;
    }
};

class CudaLaunchConfig {
public:

    static const CudaLaunchConfig default_config;

    // Choose which kernel version to launch
    bool use_original = true;
    bool use_sliced = false;
    bool use_ptb = false;
    
    // Specific to use_sliced
    bool use_cuda_graph = false;
    uint32_t threads_per_slice = 0;

    // Specific to use_ptb
    uint32_t num_blocks_per_sm = 0;

    // Static function - return the best config for a cuda launch
    static CudaLaunchConfig tune(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static std::vector<CudaLaunchConfig> get_configs(uint32_t threads_per_block, uint32_t num_blocks);

    CudaLaunchConfig(bool use_original=true, bool use_sliced=false, bool use_ptb=false, bool use_cuda_graph=false,
                 uint32_t threads_per_slice=0, uint32_t num_blocks_per_sm=0) :
        use_original(use_original), use_sliced(use_sliced), use_ptb(use_ptb), use_cuda_graph(use_cuda_graph),
        threads_per_slice(threads_per_slice), num_blocks_per_sm(num_blocks_per_sm)
    {}

    bool operator==(const CudaLaunchConfig &other) const
    {
        return (
            use_original == other.use_original &&
            use_sliced == other.use_sliced &&
            use_ptb == other.use_ptb &&
            use_cuda_graph == other.use_cuda_graph &&
            threads_per_slice == other.threads_per_slice &&
            num_blocks_per_sm == other.num_blocks_per_sm
        );
    }

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"use_original", use_original},
            {"use_ptb", use_ptb},
            {"num_blocks_per_sm", num_blocks_per_sm},
        });
    }

    std::string str() const
    {
        if (use_original) {
            return "original";
        } else if (use_ptb) {
            return "PTB_num_blocks_per_sm_" + std::to_string(num_blocks_per_sm);
        } else {
            return "";
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config);
    
    template <typename T>
    CUresult launch(T, dim3, dim3, void **, size_t, cudaStream_t, bool run_profile=false, float *elapsed_time_ms=nullptr);
    
    template <typename T>
    CUresult repeat_launch(T, dim3, dim3, void **, size_t, cudaStream_t, float dur_seconds, float *time_ms=nullptr, float *iters=nullptr, int32_t max_count=-1);
};

struct CudaLaunchCallConfig {
    CudaLaunchCall call;
    CudaLaunchConfig config;

    bool operator==(const CudaLaunchCallConfig &other) const
    {
        return (call == other.call
                && config == other.config);
    }
};

// TODO: rewrite this hash function
template <>
struct std::hash<CudaLaunchCallConfig>
{
    std::size_t operator()(const CudaLaunchCallConfig& k) const
    {
        auto _hash = std::hash<CudaLaunchCall>()(k.call);
        return _hash;
    }
};

struct CudaLaunchCallConfigPair {
    CudaLaunchCallConfig call_config_1;
    CudaLaunchCallConfig call_config_2;

    bool operator==(const CudaLaunchCallConfigPair &other) const
    {
        return (call_config_1 == other.call_config_1 && call_config_2 == other.call_config_2) ||
               (call_config_1 == other.call_config_2 && call_config_2 == other.call_config_1);
    }
};

// TODO: rewrite this hash function
template <>
struct std::hash<CudaLaunchCallConfigPair>
{
    std::size_t operator()(const CudaLaunchCallConfigPair& k) const
    {
        auto _hash = std::hash<CudaLaunchCallConfig>()(k.call_config_1) |
                     std::hash<CudaLaunchCallConfig>()(k.call_config_2);
        return _hash;
    }
};

struct KernelProfileMetrics {
    float latency_ms;
    float norm_speed;
    uint32_t iters;

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"latency_ms", latency_ms},
            {"norm_speed", norm_speed},
            {"iters", iters},
        });
    }
};

struct WorkloadPerformance {
    float latency_ms = 0.;
    float speedup = 0.;

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"latency_ms", latency_ms},
            {"speedup", speedup}
        });
    }
};

struct CudaLaunchCallConfigResult {
    CudaLaunchCall key;
    CudaLaunchConfig config;
    KernelProfileMetrics metrics;
};

struct CudaLaunchCallConfigPairResult {
    std::pair<CudaLaunchCallConfig, KernelProfileMetrics> call_config_norm_speed_1;
    std::pair<CudaLaunchCallConfig, KernelProfileMetrics> call_config_norm_speed_2;

    // For a fixed workload across all configs, what's the speedup against MPS?
    WorkloadPerformance fixed_workload_perf;

    // For a workload that is skewed to this specific config, what's the speedup against MPS?
    WorkloadPerformance unfair_workload_perf;

    float get_sum_norm_speed() {
        return call_config_norm_speed_1.second.norm_speed + call_config_norm_speed_2.second.norm_speed;
    }
};


#endif // TALLY_CUDA_LAUNCH_H