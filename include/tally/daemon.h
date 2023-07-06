#ifndef TALLY_DAEMON_H
#define TALLY_DAEMON_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tally/cuda_util.h>
#include <tally/cuda_launch.h>
#include <tally/transform.h>
#include <tally/cache.h>
#include <tally/env.h>

class TallyDaemon {
public:
    static std::shared_ptr<TallyDaemon> daemon;

    uint32_t curr_kernel_idx = 0;
    std::vector<CudaGraphCall*> cuda_graph_vec;
    cudaStream_t stream;
    bool kernels_registered = false;

    // Performance cache to use at runtime
    std::unordered_map<CudaLaunchCallConfig, float> config_latency_map;
    std::unordered_map<CudaLaunchCall, CudaLaunchConfig> kernel_config_map;

    // Conversion between kernel addr and kernel name
    std::map<std::string, const void *> mangled_kernel_name_to_host_func_map;
    std::map<std::string, const void *> demangled_kernel_name_to_host_func_map;
    std::map<const void *, std::string> host_func_to_demangled_kernel_name_map;

    // Will be loaded by TallyCache
    std::vector<std::pair<std::string, std::string>> sliced_ptx_fatbin_strs;
    std::vector<std::pair<std::string, std::string>> ptb_ptx_fatbin_strs;

    // Register transformed kernels here
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> sliced_kernel_map;
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> ptb_kernel_map;

    void register_kernels();
    void register_measurements();
    void register_fat_binary(const char* cubin_data, size_t cubin_size);

    // Set and Get kernel execution time
    float get_execution_time(CudaLaunchCallConfig&);
    void set_execution_time(CudaLaunchCallConfig&, float time_ms);

    void save_performance_cache();

    // Set and get kernel best launch config
    bool has_launch_config(CudaLaunchCall&);
    CudaLaunchConfig get_launch_config(CudaLaunchCall&);
    void set_launch_config(CudaLaunchCall&, CudaLaunchConfig &);

    TallyDaemon() {}
    ~TallyDaemon() {}
};

#endif // TALLY_DAEMON_H