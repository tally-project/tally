#ifndef TALLY_CUDA_LAUNCH_H
#define TALLY_CUDA_LAUNCH_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tally/env.h>

// Used at runtime as key to launch configuration
struct CudaLaunchCall {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;

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
    static void profile_kernel(const void *, dim3, dim3, void **, size_t, cudaStream_t);

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

    friend std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config);
    cudaError_t launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, bool run_profile=false, float *elapsed_time_ms=nullptr);
    std::pair<float, float> repeat_launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, float dur_seconds, uint32_t max_count=-1);
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


#endif // TALLY_CUDA_LAUNCH_H