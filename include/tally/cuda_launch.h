#ifndef TALLY_CUDA_LAUNCH_H
#define TALLY_CUDA_LAUNCH_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <tally/env.h>

struct PTBArgs {
    uint32_t global_idx;
	bool retreat;
};

inline std::string get_dim3_str(dim3 dim)
{
    std::string dim_str = "(" + std::to_string(dim.x) + ", " + 
                                std::to_string(dim.y) + ", " +
                                std::to_string(dim.z) + ")";
    return dim_str;
}

struct CudaLaunchMetadata {
    int max_threads_per_block = 0;
    int static_shmem_size_bytes = 0;
    int num_regs = 0;
    int max_dynamic_shmem_size_bytes = 0;

    // runtime passed in through cudaLaunchKernel
    int dynamic_shmem_size_bytes = 0;

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"max_threads_per_block", max_threads_per_block},
            {"static_shmem_size_bytes", static_shmem_size_bytes},
            {"num_regs", num_regs},
            {"max_dynamic_shmem_size_bytes", max_dynamic_shmem_size_bytes},
            {"dynamic_shmem_size_bytes", dynamic_shmem_size_bytes},
        });
    }
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchMetadata& meta)
{
    os << "CudaLaunchMetadata: \n";
    os << "\tmax_threads_per_block: " << meta.max_threads_per_block << "\n";
    os << "\tstatic_shmem_size_bytes: " << meta.static_shmem_size_bytes << "\n";
    os << "\tnum_regs: " << meta.num_regs << "\n";
    os << "\tmax_dynamic_shmem_size_bytes: " << meta.max_dynamic_shmem_size_bytes << "\n";
    os << "\tdynamic_shmem_size_bytes: " << meta.dynamic_shmem_size_bytes;

    return os;
}

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

    std::string dim_str() {
        return get_dim3_str(gridDim) + "_" + get_dim3_str(blockDim);
    }

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
    bool use_ptb = false;
    bool use_dynamic_ptb = false;
    bool use_preemptive_ptb = false;

    // Specific to use_ptb
    uint32_t num_blocks_per_sm = 0;

    // Static function - return the best config for a cuda launch
    static std::vector<CudaLaunchConfig> get_profile_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks);
    static std::vector<CudaLaunchConfig> get_workload_agnostic_sharing_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks);
    static std::vector<CudaLaunchConfig> get_preemptive_configs(CudaLaunchCall &launch_call, uint32_t threads_per_block, uint32_t num_blocks);

    CudaLaunchConfig(
        bool use_original=true, bool use_ptb=false,
        bool use_dynamic_ptb=false, bool use_preemptive_ptb=false, uint32_t num_blocks_per_sm=0) :
        use_original(use_original),
        use_ptb(use_ptb),
        use_dynamic_ptb(use_dynamic_ptb),
        use_preemptive_ptb(use_preemptive_ptb),
        num_blocks_per_sm(num_blocks_per_sm)
    {}

    bool operator==(const CudaLaunchConfig &other) const
    {
        return (
            use_original == other.use_original &&
            use_ptb == other.use_ptb &&
            use_dynamic_ptb == other.use_dynamic_ptb &&
            use_preemptive_ptb == other.use_preemptive_ptb &&
            num_blocks_per_sm == other.num_blocks_per_sm
        );
    }

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"use_original", use_original},
            {"use_ptb", use_ptb},
            {"use_dynamic_ptb", use_dynamic_ptb},
            {"use_preemptive_ptb", use_preemptive_ptb},
            {"num_blocks_per_sm", num_blocks_per_sm},
        });
    }

    std::string str() const
    {
        if (use_original) {
            return "original";
        } else if (use_ptb) {
            return "PTB num_blocks_per_sm: " + std::to_string(num_blocks_per_sm);
        } else if (use_dynamic_ptb) {
            return "Dynamic PTB num_blocks_per_sm: " + std::to_string(num_blocks_per_sm);
        } else if (use_preemptive_ptb) {
            return "Preemptive PTB num_blocks_per_sm: " + std::to_string(num_blocks_per_sm);
        } else {
            return "";
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config);
    
    CUresult launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, PTBArgs *ptb_args, uint32_t *curr_idx_arr=nullptr);
    CUresult repeat_launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, float dur_seconds, PTBArgs *ptb_args, uint32_t *curr_idx_arr=nullptr, float *time_ms=nullptr, float *iters=nullptr, int32_t max_count=-1);
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

struct TempKernelProfileMetrics {
    float avg_latency_ms = 0.;
    uint32_t count = 0;
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
    CudaLaunchMetadata meta_data;
    KernelProfileMetrics metrics;
};

struct CudaLaunchCallConfigPairResult {
    std::pair<CudaLaunchCallConfig, KernelProfileMetrics> call_config_norm_speed_1;
    std::pair<CudaLaunchCallConfig, KernelProfileMetrics> call_config_norm_speed_2;

    std::pair<CudaLaunchCallConfig, CudaLaunchMetadata> call_config_meta_data_1;
    std::pair<CudaLaunchCallConfig, CudaLaunchMetadata> call_config_meta_data_2;

    // For a fixed workload across all configs, what's the speedup against MPS?
    WorkloadPerformance fixed_workload_perf;

    // For a workload that is skewed to this specific config, what's the speedup against MPS?
    WorkloadPerformance unfair_workload_perf;

    float get_sum_norm_speed() {
        return call_config_norm_speed_1.second.norm_speed + call_config_norm_speed_2.second.norm_speed;
    }

    // Get corresponding config for each launch call
    // The bool indicated whether the two kernels should be run time-shared
    std::tuple<CudaLaunchConfig, CudaLaunchConfig, bool> get_configs(CudaLaunchCall &launch_call_1, CudaLaunchCall &launch_call_2)
    {
        CudaLaunchConfig config_1;
        CudaLaunchConfig config_2;
        bool time_share = false;

        if (launch_call_1 == call_config_norm_speed_1.first.call) {
            config_1 = call_config_norm_speed_1.first.config;
            config_2 = call_config_norm_speed_2.first.config;
        } else {
            config_1 = call_config_norm_speed_2.first.config;
            config_2 = call_config_norm_speed_1.first.config;
        }

        if (get_sum_norm_speed() < TIME_SHARE_THRESHOLD) {
            time_share = true;
        }

        return std::make_tuple<CudaLaunchConfig, CudaLaunchConfig, bool>(std::move(config_1), std::move(config_2), std::move(time_share));
    }
};

struct WrappedCUfunction {
    CUfunction func;
    uint32_t num_args;

    CudaLaunchMetadata meta_data;
};

using partial_t = std::function<CUresult(CudaLaunchConfig, PTBArgs*, uint32_t*, bool, float, float*, float*, int32_t, bool)>;

struct KernelLaunchWrapper {

public:
	// Callable to launch kernel
	partial_t kernel_to_dispatch;

	void *args;

	// whether it is blackbox kernel from nvidia libraries
	bool is_library_call;

	// unique identification of the kernel
	CudaLaunchCall launch_call;

	// Stream to launch to
	cudaStream_t launch_stream;

	// Useful info
	int dynamic_shmem_size_bytes = 0;

	// For query the status of the kernel
	cudaEvent_t event = nullptr;

	void free_args()
	{
		free(args);
	}
};

#endif // TALLY_CUDA_LAUNCH_H