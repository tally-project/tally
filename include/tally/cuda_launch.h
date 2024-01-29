#ifndef TALLY_CUDA_LAUNCH_H
#define TALLY_CUDA_LAUNCH_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <tally/env.h>

struct PTBKernelArgs {
    uint32_t global_idx;
	bool retreat;
};

struct SlicedKernelArgs {
    dim3 sliced_gridDim;
    std::vector<dim3> block_offsets;

    // if < 0, launch all slices together
    // if >= 0, launch the i-th slice
    int launch_idx = -1;
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

    uint32_t num_blocks;
    uint32_t threads_per_block;

    CudaLaunchCall(){}

    CudaLaunchCall(const void *func, dim3 gridDim, dim3 blockDim) :
        func(func),
        gridDim(gridDim),
        blockDim(blockDim),
        num_blocks(gridDim.x * gridDim.y * gridDim.z),
        threads_per_block(blockDim.x * blockDim.y * blockDim.z)
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
    bool use_original = false;
    bool use_ptb = false;
    bool use_dynamic_ptb = false;
    bool use_preemptive_ptb = false;
    bool use_sliced = false;

    // Specific to ptb
    uint32_t blocks_per_sm = 0;
    uint32_t max_worker_blocks = 10000; // essentially INT_MAX

    // Specific to sliced
    uint32_t num_slices = 0;

    CudaLaunchConfig(){}

    static std::vector<CudaLaunchConfig> get_profile_configs(CudaLaunchCall &launch_call);
    static std::vector<CudaLaunchConfig> get_workload_agnostic_sharing_configs(CudaLaunchCall &launch_call);
    static std::vector<CudaLaunchConfig> get_preemptive_configs(CudaLaunchCall &launch_call);
    static std::vector<CudaLaunchConfig> get_sliced_configs(CudaLaunchCall &launch_call);

    static CudaLaunchConfig get_original_config() {
        CudaLaunchConfig config;
        config.use_original = true;
        return config;
    }

    static CudaLaunchConfig get_ptb_config(uint32_t blocks_per_sm) {
        CudaLaunchConfig config;
        config.use_dynamic_ptb = true;
        config.blocks_per_sm = blocks_per_sm;
        return config;
    }

    static CudaLaunchConfig get_dynamic_ptb_config(uint32_t blocks_per_sm) {
        CudaLaunchConfig config;
        config.use_ptb = true;
        config.blocks_per_sm = blocks_per_sm;
        return config;
    }

    static CudaLaunchConfig get_preemptive_ptb_config(uint32_t blocks_per_sm) {
        CudaLaunchConfig config;
        config.use_preemptive_ptb = true;
        config.blocks_per_sm = blocks_per_sm;
        return config;
    }

    static CudaLaunchConfig get_sliced_config(uint32_t num_slices) {
        CudaLaunchConfig config;
        config.use_sliced = true;
        config.num_slices = num_slices;
        return config;
    }

    bool operator==(const CudaLaunchConfig &other) const
    {
        return (
            use_original == other.use_original &&
            use_sliced == other.use_sliced &&
            use_ptb == other.use_ptb &&
            use_dynamic_ptb == other.use_dynamic_ptb &&
            use_preemptive_ptb == other.use_preemptive_ptb &&
            blocks_per_sm == other.blocks_per_sm &&
            num_slices == other.num_slices &&
            max_worker_blocks == other.max_worker_blocks
        );
    }

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"use_original", use_original},
            {"use_ptb", use_ptb},
            {"use_dynamic_ptb", use_dynamic_ptb},
            {"use_preemptive_ptb", use_preemptive_ptb},
            {"use_sliced", use_sliced},
            {"blocks_per_sm", blocks_per_sm},
            {"max_worker_blocks", max_worker_blocks},
            {"num_slices", num_slices},
        });
    }

    std::string str() const
    {
        if (use_original) {
            return "original";
        }

        else if (use_ptb || use_dynamic_ptb || use_preemptive_ptb) {
            auto str_builder = std::string("");
            if (use_ptb) {
                str_builder += "PTB: ";
            } else if (use_dynamic_ptb) {
                str_builder += "Dynamic PTB: ";
            } else if (use_preemptive_ptb) {
                str_builder += "Preemptive PTB: ";
            }

            if (max_worker_blocks < CUDA_NUM_SM) {
                str_builder += "Max worker blocks: " + std::to_string(max_worker_blocks);
            } else {
                str_builder += "Blocks per SM: " + std::to_string(blocks_per_sm);
            }

            return str_builder;
        }
        
        else if (use_sliced) {
            return "Sliced: Num slices: " + std::to_string(num_slices);
        }
        
        return "";
    }

    friend std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config);
    
    CUresult launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, PTBKernelArgs *ptb_args=nullptr,
                    uint32_t *curr_idx_arr=nullptr, SlicedKernelArgs *slice_args=nullptr);

    CUresult repeat_launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, float dur_seconds, PTBKernelArgs *ptb_args=nullptr,
                           uint32_t *curr_idx_arr=nullptr, SlicedKernelArgs *slice_args=nullptr, float *time_ms=nullptr, float *iters=nullptr,
                           int32_t max_count=-1); 
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

    void add_measurement(float latency_ms) {
        avg_latency_ms = (avg_latency_ms * count + latency_ms) / (count + 1);
        count += 1;
    }
};

struct KernelProfileMetrics {
    float latency_ms;
    float norm_speed;
    uint32_t iters;
    float preemption_latency_ms;

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"latency_ms", latency_ms},
            {"norm_speed", norm_speed},
            {"iters", iters},
            {"preemption_latency_ms", preemption_latency_ms}
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

using partial_t = std::function<CUresult(CudaLaunchConfig, PTBKernelArgs*, uint32_t*, SlicedKernelArgs*, bool, float, float*, float*, int32_t, bool)>;

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

std::vector<uint32_t> get_candidate_blocks_per_sm(uint32_t threads_per_block, uint32_t num_blocks, uint32_t max_threads_per_sm);
std::vector<uint32_t> get_candidate_num_slices(uint32_t threads_per_block, uint32_t num_blocks, uint32_t max_threads_per_sm);

SlicedKernelArgs get_sliced_kernel_args(dim3 gridDim, uint32_t num_slices);

#endif // TALLY_CUDA_LAUNCH_H