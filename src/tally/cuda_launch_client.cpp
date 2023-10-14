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
#include <tally/client_offline.h>

const CudaLaunchConfig CudaLaunchConfig::default_config = CudaLaunchConfig();

std::ostream& operator<<(std::ostream& os, const CudaLaunchConfig& config)
{
    os << "CudaLaunchConfig: ";
    if (config.use_original) {
        os << "original";
    } else if (config.use_ptb) {
        os << "PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    } else if (config.use_dynamic_ptb) {
        os << "Dynamic PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    } else if (config.use_preemptive_ptb) {
        os << "Preemptive PTB: num_blocks_per_sm: " << config.num_blocks_per_sm;
    }
    return os;
}

// =================== Used by Workload Agnostic Sharing Scheduler ===========================
std::vector<CudaLaunchConfig> CudaLaunchConfig::get_workload_agnostic_sharing_configs(uint32_t threads_per_block, uint32_t num_blocks)
{
    std::vector<CudaLaunchConfig> configs;

    // some PTB configs
    uint32_t _num_blocks_per_sm = 1;
    while(true) {

        // One kernel should not take all the thread slots
        if (_num_blocks_per_sm * threads_per_block > PTB_MAX_NUM_THREADS_PER_SM) {
            break;
        }
        
        // There is no point going over the total num of blocks
        // But we will keep the (_num_blocks_per_sm == 1) case
        if (_num_blocks_per_sm > 1 && (_num_blocks_per_sm - 1) * CUDA_NUM_SM > num_blocks) {
            break;
        }

        // regular PTB
        CudaLaunchConfig ptb_config(false, true, false, false, _num_blocks_per_sm);

        // dynamic PTB
        CudaLaunchConfig dynamic_ptb_config(false, false, true, false, _num_blocks_per_sm);

        // preemptive PTB
        CudaLaunchConfig preemptive_ptb_config(false, false, false, true, _num_blocks_per_sm);

        configs.push_back(ptb_config);
        configs.push_back(dynamic_ptb_config);
        // configs.push_back(preemptive_ptb_config);

        _num_blocks_per_sm++;
    }
    
    return configs;
}