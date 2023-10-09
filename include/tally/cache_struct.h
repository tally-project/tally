#ifndef TALLY_CACHE_STRUCT_H
#define TALLY_CACHE_STRUCT_H

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <shared_mutex>

#include <nlohmann/json.hpp>

#include <tally/cuda_launch.h>
#include <tally/cuda_util.h>

static std::shared_mutex mutex_;

struct CudaLaunchKey {
    std::string kernel_name;
    dim3 gridDim;
    dim3 blockDim;
    size_t cubin_uid;

    bool operator==(const CudaLaunchKey &other) const
    {
        return (
            kernel_name == other.kernel_name
            && cubin_uid == other.cubin_uid
            && gridDim.x == other.gridDim.x
            && gridDim.y == other.gridDim.y
            && gridDim.z == other.gridDim.z
            && blockDim.x == other.blockDim.x
            && blockDim.y == other.blockDim.y
            && blockDim.z == other.blockDim.z
        );
    }

    uint64_t block_size() const
    {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    uint64_t total_threads() const
    {
        return gridDim.x * gridDim.y * gridDim.z * block_size();
    }

    uint64_t total_blocks() const
    {
        return gridDim.x * gridDim.y * gridDim.z;
    }

    nlohmann::json json() const
    {
        return nlohmann::json({
            {"kernel_name", kernel_name},
            {"cubin_uid", cubin_uid},
            {"gridDim", get_dim3_str(gridDim)},
            {"blockDim", get_dim3_str(blockDim)},
        });
    }

    std::string str() const
    {
        return kernel_name + "_" + get_dim3_str(gridDim) + "_" + get_dim3_str(blockDim) + "_" + std::to_string(cubin_uid);
    }
    
};

template <>
struct std::hash<CudaLaunchKey>
{
    std::size_t operator()(const CudaLaunchKey& k) const
    {
        auto _hash = std::hash<std::string>()(k.kernel_name);
        return _hash;
    }
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchKey& launch_key)
{
    os << "LaunchKey: \n";
    os << "\tName: " << launch_key.kernel_name << "\n";
    os << "\tblockDim: " << "(" << launch_key.blockDim.x << ", " << launch_key.blockDim.y << ", " << launch_key.blockDim.z << ")" << "\n";
    os << "\tgridDim: " << "(" << launch_key.gridDim.x << ", " << launch_key.gridDim.y << ", " << launch_key.gridDim.z << ")";
    
    return os;
}

struct CudaLaunchKeyPair {
    CudaLaunchKey launch_key_1;
    CudaLaunchKey launch_key_2;

    bool operator==(const CudaLaunchKeyPair &other) const
    {
        return (launch_key_1 == other.launch_key_1 && launch_key_2 == other.launch_key_2) ||
               (launch_key_1 == other.launch_key_2 && launch_key_2 == other.launch_key_1);
    }
};

template <>
struct std::hash<CudaLaunchKeyPair>
{
    std::size_t operator()(const CudaLaunchKeyPair& k) const
    {
        auto _hash = std::hash<CudaLaunchKey>()(k.launch_key_1) |
                     std::hash<CudaLaunchKey>()(k.launch_key_2);
        return _hash;
    }
};

struct CudaLaunchKeyConfig {
    CudaLaunchKey key;
    CudaLaunchConfig config;

    bool operator==(const CudaLaunchKeyConfig &other) const
    {
        return (key == other.key && config == other.config);
    }
};

template <>
struct std::hash<CudaLaunchKeyConfig>
{
    std::size_t operator()(const CudaLaunchKeyConfig& k) const
    {
        auto _hash = std::hash<CudaLaunchKey>()(k.key);
        return _hash;
    }
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchKeyConfig& key_config)
{
    os << "CudaLaunchKeyConfig: \n";
    os << key_config.key << "\n";
    os << key_config.config;

    return os;
}

struct CudaLaunchKeyConfigPair {
    CudaLaunchKeyConfig key_config_1;
    CudaLaunchKeyConfig key_config_2;

    bool operator==(const CudaLaunchKeyConfigPair &other) const
    {
        return (key_config_1 == other.key_config_1 && key_config_2 == other.key_config_2) ||
               (key_config_1 == other.key_config_2 && key_config_2 == other.key_config_1);
    }
};

template <>
struct std::hash<CudaLaunchKeyConfigPair>
{
    std::size_t operator()(const CudaLaunchKeyConfigPair& k) const
    {
        auto _hash = std::hash<CudaLaunchKeyConfig>()(k.key_config_1) |
                     std::hash<CudaLaunchKeyConfig>()(k.key_config_2);
        return _hash;
    }
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchKeyConfigPair& key_config_pair)
{
    os << "CudaLaunchKeyConfigPair: \n";
    os << "First: \n";
    os << key_config_pair.key_config_1 << "\n";
    os << "Second: \n";
    os << key_config_pair.key_config_2;

    return os;
}

struct CudaLaunchKeyConfigResult {
    CudaLaunchKey key;
    CudaLaunchConfig config;
    CudaLaunchMetadata meta_data;
    KernelProfileMetrics metrics;
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchKeyConfigResult& res)
{
    os << "CudaLaunchKeyConfigResult: \n";
    os << "\tNorm Speed: " << res.metrics.norm_speed << "\n";
    os << "\tLatency: " << res.metrics.latency_ms << " ms";

    return os;
}

struct CudaLaunchKeyConfigPairResult {
    std::pair<CudaLaunchKeyConfig, KernelProfileMetrics> config_key_norm_speed_1;
    std::pair<CudaLaunchKeyConfig, KernelProfileMetrics> config_key_norm_speed_2;

    std::pair<CudaLaunchKeyConfig, CudaLaunchMetadata> config_key_meta_data_1;
    std::pair<CudaLaunchKeyConfig, CudaLaunchMetadata> config_key_meta_data_2;

    // For a fixed workload across all configs, what's the speedup against MPS?
    WorkloadPerformance fixed_workload_perf;

    // For a workload that is skewed to this specific config, what's the speedup against MPS?
    WorkloadPerformance unfair_workload_perf;

    float get_sum_norm_speed() const {
        return config_key_norm_speed_1.second.norm_speed + config_key_norm_speed_2.second.norm_speed;
    }

    std::pair<bool, std::string> exceeds_hardware_limit() const
    {
        auto key_1 = config_key_norm_speed_1.first.key;
        auto key_2 = config_key_norm_speed_2.first.key;

        auto config_1 = config_key_norm_speed_1.first.config;
        auto config_2 = config_key_norm_speed_2.first.config;

        auto meta_data_1 = config_key_meta_data_1.second;
        auto meta_data_2 = config_key_meta_data_2.second;

        uint64_t num_threads_1;
        uint64_t num_threads_2;

        uint64_t num_blocks_1;
        uint64_t num_blocks_2;

        if (config_1.use_original) {
            num_threads_1 = key_1.total_threads() / CUDA_NUM_SM;
            num_blocks_1 = key_1.total_blocks() / CUDA_NUM_SM;
        } else {
            num_threads_1 = key_1.block_size() * config_1.num_blocks_per_sm;
            num_blocks_1 = config_1.num_blocks_per_sm;
        }

        if (config_2.use_original) {
            num_threads_2 = key_2.total_threads() / CUDA_NUM_SM;
            num_blocks_2 = key_2.total_blocks() / CUDA_NUM_SM;
        } else {
            num_threads_2 = key_2.block_size() * config_2.num_blocks_per_sm;
            num_blocks_2 = config_2.num_blocks_per_sm;
        }

        uint64_t sum_threads = num_threads_1 + num_threads_2;

        if (sum_threads > CUDA_MAX_NUM_THREADS_PER_SM) {
            return std::pair<bool, std::string>(true, 
                std::string("Exceeding threads per SM limit.") + 
                std::string(" Limit: ") + std::to_string(CUDA_MAX_NUM_THREADS_PER_SM) +
                std::string(" Total threads: ") + std::to_string(sum_threads));
        }

        uint64_t sum_regs = num_threads_1 * meta_data_1.num_regs + num_threads_2 * meta_data_2.num_regs;

        if (sum_regs > CUDA_MAX_NUM_REGISTERS_PER_SM) {
            return std::pair<bool, std::string>(true, 
                std::string("Exceeding registers per SM limit.") + 
                std::string(" Limit: ") + std::to_string(CUDA_MAX_NUM_REGISTERS_PER_SM) +
                std::string(" Total regitsers: ") + std::to_string(sum_regs));
        }


        uint32_t total_shm_size = num_blocks_1 * (meta_data_1.static_shmem_size_bytes + meta_data_1.dynamic_shmem_size_bytes) + 
                                  num_blocks_2 * (meta_data_2.static_shmem_size_bytes + meta_data_2.dynamic_shmem_size_bytes);

        if (total_shm_size > CUDA_MAX_SHM_BYTES_PER_SM) {
            return std::pair<bool, std::string>(true, 
                std::string("Exceeding shared memory per SM limit.") + 
                std::string(" Limit: ") + std::to_string(CUDA_MAX_SHM_BYTES_PER_SM) +
                std::string(" Total shm size: ") + std::to_string(total_shm_size));
        }

        return std::pair<bool, std::string>(false, "");
    }
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchKeyConfigPairResult& res)
{
    float norm_speed_1 = res.config_key_norm_speed_1.second.norm_speed;
    float norm_speed_2 = res.config_key_norm_speed_2.second.norm_speed;

    auto check_hardware_limit = res.exceeds_hardware_limit();
    auto exceed_limit = check_hardware_limit.first;
    auto reason = check_hardware_limit.second;

    os << "CudaLaunchKeyConfigPairResult: \n";
    os << "\tK1 Norm Speed: " << norm_speed_1 << " K2 Norm Speed: " << norm_speed_2 << " Sum: " << res.get_sum_norm_speed() << "\n";
    os << "Fixed Workload Performance: speedup: " << res.fixed_workload_perf.speedup << "\n";
    os << "Unfair Workload Performance: speedup: " << res.unfair_workload_perf.speedup << "\n";
    os << "ExceedsHardwareLimit: " << std::to_string(exceed_limit);

    if (exceed_limit) {
        os << "\n";
        os << "ExceedsLimitReason: " << reason;
    }

    return os;
}

// For each cubin file
// Don't forget to modify serialization.h when modifying this struct
class CubinData
{
public:

    // Unique id of this cubin data
    uint32_t cubin_uid;

    // a cubin file
    std::string cubin_data;

    // Key: kernel name, value: vector of the sizes of arguments in ordinal order
    std::map<std::string, std::vector<uint32_t>> kernel_args;

    // All the original PTX files and fatbin
    std::vector<std::pair<std::string, std::string>> original_data;

    // All the PTB PTX files and fatbin
    std::vector<std::pair<std::string, std::string>> ptb_data;

    // All the dynamic PTB PTX files and fatbin
    std::vector<std::pair<std::string, std::string>> dynamic_ptb_data;

    // All the preemptive  PTB PTX files and fatbin
    std::vector<std::pair<std::string, std::string>> preemptive_ptb_data;
};

class CubinCache
{
public:
    uint32_t uid_counter = 0;

    // Cubin size : Cubin data
    std::map<size_t, std::vector<CubinData>> cubin_map;

    bool contains(const char* cubin_data, size_t cubin_size)
    {
        return find_transform_data(cubin_data, cubin_size) != nullptr;
    }

    CubinData* find_transform_data(const char* cubin_data, size_t cubin_size)
    {
        std::shared_lock lock(mutex_);
        if (cubin_map.find(cubin_size) != cubin_map.end()) {
            for (auto &data : cubin_map[cubin_size]) {
                if (memcmp(data.cubin_data.c_str(), cubin_data, cubin_size) == 0) {
                    return &data;
                }
            }
        }

        return nullptr;
    }

    std::vector<std::pair<std::string, std::string>>
    get_ptb_data(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->ptb_data;
    }

    std::vector<std::pair<std::string, std::string>>
    get_dynamic_ptb_data(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->dynamic_ptb_data;
    }

    std::vector<std::pair<std::string, std::string>>
    get_preemptive_ptb_data(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->preemptive_ptb_data;
    }

    std::vector<std::pair<std::string, std::string>>
    get_original_data(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->original_data;
    }

    std::map<std::string, std::vector<uint32_t>>
    get_kernel_args(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);

        if (!transform_data) {
            std::cout << "!transform_data" << std::endl;
        }
        return transform_data->kernel_args;
    }

    const char *get_cubin_data_ptr(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->cubin_data.c_str();
    }

    std::string get_cubin_data_str_from_cubin_uid(uint32_t cubin_uid)
    {
        std::shared_lock lock(mutex_);

        for (auto &pair : cubin_map) {
            auto &cubin_data_vec = pair.second;

            for (auto &data : cubin_data_vec) {
                if (data.cubin_uid == cubin_uid) {
                    return data.cubin_data;
                }
            }
        }

        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Fail to find cubin str from uid.");
        return "";
    }

    const char *get_cubin_data_str_ptr_from_cubin_uid(uint32_t cubin_uid)
    {
        std::shared_lock lock(mutex_);

        for (auto &pair : cubin_map) {
            auto &cubin_data_vec = pair.second;

            for (auto &data : cubin_data_vec) {
                if (data.cubin_uid == cubin_uid) {
                    return data.cubin_data.c_str();
                }
            }
        }

        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Fail to find cubin str ptr from uid.");
        return nullptr;
    }

    size_t get_cubin_size_from_cubin_uid(uint32_t cubin_uid)
    {
        std::shared_lock lock(mutex_);

        for (auto &pair : cubin_map) {
            auto &cubin_data_vec = pair.second;

            for (auto &data : cubin_data_vec) {
                if (data.cubin_uid == cubin_uid) {
                    return pair.first;
                }
            }
        }

        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Fail to find cubin size from uid.");
        return -1;
    }

    uint32_t get_cubin_data_uid(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->cubin_uid;
    }

    void add_data(
        size_t cubin_size,
        std::string &cubin_str,
        std::map<std::string, std::vector<uint32_t>> &kernel_args,
        std::vector<std::pair<std::string, std::string>> &original_data,
        std::vector<std::pair<std::string, std::string>> &ptb_data,
        std::vector<std::pair<std::string, std::string>> &dynamic_ptb_data,
        std::vector<std::pair<std::string, std::string>> &preemptive_ptb_data
    )
    {
        std::unique_lock lock(mutex_);
        cubin_map[cubin_size].push_back( CubinData { uid_counter, cubin_str, kernel_args, original_data, ptb_data, dynamic_ptb_data, preemptive_ptb_data } );
        uid_counter++;
    }
};

class PerformanceCache
{
public:
    // Single-kernel performance
    std::unordered_map<CudaLaunchKeyConfig, CudaLaunchKeyConfigResult> single_kernel_perf_map;
    std::unordered_map<CudaLaunchKey, CudaLaunchKeyConfigResult> single_kernel_best_config_map;

    // Kernel pair and the normalized speed
    std::unordered_map<CudaLaunchKeyPair, std::unordered_map<CudaLaunchKeyConfigPair, CudaLaunchKeyConfigPairResult>> kernel_pair_perf_map;
    std::unordered_map<CudaLaunchKeyPair, CudaLaunchKeyConfigPairResult> kernel_pair_best_config_map;

    void set_single_kernel_perf(CudaLaunchKey &launch_key, CudaLaunchConfig &launch_config, CudaLaunchKeyConfigResult &res)
    {
        CudaLaunchKeyConfig key_config(launch_key,launch_config);
        single_kernel_perf_map[key_config] = res;
        // write_single_kernel_perf_to_file();

        // std::cout << "=============== Single Kernel Profile Result ===============" << std::endl;

        // std::cout << launch_key << std::endl;
        // std::cout << launch_config << std::endl;
        // std::cout << res.meta_data << std::endl;
        // std::cout << res << std::endl;
        // std::cout << "\n" << std::endl;
    }

    void set_single_kernel_best_config(CudaLaunchKey &launch_key, CudaLaunchKeyConfigResult &res)
    {
        single_kernel_best_config_map[launch_key] = res;
        // write_single_kernel_best_config_to_file();

        // std::cout << "=============== Single Kernel Best Result ===============" << std::endl;

        // std::cout << launch_key << std::endl;
        // std::cout << res.config << std::endl;
        // std::cout << res.meta_data << std::endl;
        // std::cout << res << std::endl;
        // std::cout << "\n" << std::endl;
    }

    void set_kernel_pair_perf(CudaLaunchKey &launch_key_1, CudaLaunchKey &launch_key_2, 
                              CudaLaunchConfig &launch_config_1, CudaLaunchConfig &launch_config_2,
                              CudaLaunchKeyConfigPairResult &res)
    {
        CudaLaunchKeyPair key_pair(launch_key_1, launch_key_2);

        CudaLaunchKeyConfig key_config_1(launch_key_1, launch_config_1);
        CudaLaunchKeyConfig key_config_2(launch_key_2, launch_config_2);

        CudaLaunchKeyConfigPair key_config_pair(key_config_1, key_config_2);

        kernel_pair_perf_map[key_pair][key_config_pair] = res;

        // write_kernel_pair_perf_to_file();

        // std::cout << "=============== Kernel Pair Profile Result ===============" << std::endl;

        // std::cout << launch_key_1 << std::endl;
        // std::cout << launch_config_1 << std::endl;
        // std::cout << res.config_key_meta_data_1.second << std::endl;
        // std::cout << launch_key_2 << std::endl;
        // std::cout << launch_config_2 << std::endl;
        // std::cout << res.config_key_meta_data_2.second << std::endl;
        // std::cout << res << std::endl;
        // std::cout << "\n" << std::endl;
    }

    void set_kernel_pair_best_config(CudaLaunchKey &launch_key_1, CudaLaunchKey &launch_key_2, CudaLaunchKeyConfigPairResult &res)
    {
        CudaLaunchKeyPair key_pair(launch_key_1, launch_key_2);
        kernel_pair_best_config_map[key_pair] = res;

        // write_kernel_pair_best_config_to_file();

        // std::cout << "=============== Kernel Pair Best Result ===============" << std::endl;

        // auto launch_config_1 = res.config_key_norm_speed_1.first.config;
        // auto launch_config_2 = res.config_key_norm_speed_2.first.config;

        // std::cout << launch_key_1 << std::endl;
        // std::cout << launch_config_1 << std::endl;
        // std::cout << res.config_key_meta_data_1.second << std::endl;
        // std::cout << launch_key_2 << std::endl;
        // std::cout << launch_config_2 << std::endl;
        // std::cout << res.config_key_meta_data_2.second << std::endl;
        // std::cout << res << std::endl;
        // std::cout << "\n" << std::endl;
    }

    void write_single_kernel_perf_to_file() const
    {
        nlohmann::json json;
   
        for (auto &pair : single_kernel_perf_map) {

            auto key_config = pair.first;
            auto res = pair.second;

            std::string key_str = key_config.key.str();

            if (!json.contains(key_str)) {
                json[key_str]["LaunchKey"] = key_config.key.json();
                json[key_str]["Results"] = nlohmann::json::array();
            }

            nlohmann::json entry = nlohmann::json({
                {"LaunchConfig", key_config.config.json()},
                {"LaunchMetadata", res.meta_data.json()},
                {"ResultMetrics", res.metrics.json()}
            });

            json[key_str]["Results"].push_back(entry);
        }

        write_json_to_file(json, "single_kernel_perf.json");
    }

    void write_single_kernel_best_config_to_file() const
    {
        nlohmann::json json;

        for (auto &pair : single_kernel_best_config_map) {
            auto key = pair.first;
            auto best_res = pair.second;

            std::string key_str = key.str();

            if (!json.contains(key_str)) {
                json[key_str]["LaunchKey"] = key.json();
                json[key_str]["Results"] = nlohmann::json::array();
            }

            nlohmann::json entry = nlohmann::json({
                {"LaunchConfig", best_res.config.json()},
                {"LaunchMetadata", best_res.meta_data.json()},
                {"Metrics", best_res.metrics.json()},
            });

            json[key_str]["Results"].push_back(entry);
        }

        write_json_to_file(json, "single_kernel_best_config.json");
    }

    void write_kernel_pair_perf_to_file() const
    {
        nlohmann::json json;

        for (auto &pair : kernel_pair_perf_map) {
            auto &config_res_map = pair.second;

            for (auto &config_res_pair : config_res_map) {

                auto &key_config_pair = config_res_pair.first;
                auto &res = config_res_pair.second;

                auto &key_1 = key_config_pair.key_config_1.key;
                auto &key_2 = key_config_pair.key_config_2.key;

                auto &config_1 = key_config_pair.key_config_1.config;
                auto &config_2 = key_config_pair.key_config_2.config;

                std::string group_name = key_1.str() + "_" + key_2.str();

                if (!json.contains(group_name)) {
                    json[group_name]["LaunchKey_1"] = key_1.json();
                    json[group_name]["LaunchKey_2"] = key_2.json();
                    json[group_name]["Results"] = nlohmann::json::array();
                }

                nlohmann::json entry = nlohmann::json({
                    {"LaunchConfig_1", config_1.json()},
                    {"LaunchMetadata_1", res.config_key_meta_data_1.second.json()},
                    {"ResultMetrics_1", res.config_key_norm_speed_1.second.json()},
                    {"LaunchConfig_2", config_2.json()},
                    {"LaunchMetadata_2", res.config_key_meta_data_2.second.json()},
                    {"ResultMetrics_2", res.config_key_norm_speed_2.second.json()},
                    {"FixedWorkloadPerf", res.fixed_workload_perf.json()},
                    {"UnfairWorkloadPerf", res.unfair_workload_perf.json()},
                    {"SumNormSpeed", res.get_sum_norm_speed()},
                    {"ExceedsHardwareLimit", res.exceeds_hardware_limit()}
                });

                json[group_name]["Results"].push_back(entry);
            }
        }

        write_json_to_file(json, "kernel_pair_perf.json");
    }

    void write_kernel_pair_best_config_to_file() const
    {
        nlohmann::json json;

        for (auto &pair : kernel_pair_best_config_map) {
            auto &launch_key_pair = pair.first;
            auto &res = pair.second;

            auto &key_1 = launch_key_pair.launch_key_1;
            auto &key_2 = launch_key_pair.launch_key_2;

            std::string group_name = key_1.str() + "_" + key_2.str();

            if (!json.contains(group_name)) {
                json[group_name]["LaunchKey_1"] = key_1.json();
                json[group_name]["LaunchKey_2"] = key_2.json();
                json[group_name]["Results"] = nlohmann::json::array();
            }

            nlohmann::json entry = nlohmann::json({
                {"LaunchConfig_1", res.config_key_norm_speed_1.first.config.json()},
                {"LaunchMetadata_1", res.config_key_meta_data_1.second.json()},
                {"ResultMetrics_1", res.config_key_norm_speed_1.second.json()},
                {"LaunchConfig_2", res.config_key_norm_speed_2.first.config.json()},
                {"LaunchMetadata_2", res.config_key_meta_data_2.second.json()},
                {"ResultMetrics_2", res.config_key_norm_speed_2.second.json()},
                {"FixedWorkloadPerf", res.fixed_workload_perf.json()},
                {"UnfairWorkloadPerf", res.unfair_workload_perf.json()},
                {"SumNormSpeed", res.get_sum_norm_speed()},
                {"ExceedsHardwareLimit", res.exceeds_hardware_limit()}
            });

            json[group_name]["Results"].push_back(entry);
        }

        write_json_to_file(json, "kernel_pair_best_config.json");
    }

    void write_json_to_file(nlohmann::json &json, std::string file_name) const
    {
        auto perf_dir = std::filesystem::path("perf_results");
        if (!std::filesystem::is_directory(perf_dir)) {
            std::filesystem::create_directory(perf_dir);
        }

        auto policy = SCHEDULER_POLICY;
        std::string policy_str;

        if (policy == TALLY_SCHEDULER_POLICY::PROFILE) {
            policy_str = "profile";
        } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AGNOSTIC_SHARING) {
            policy_str = "workload_agnostic";
        } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AWARE_SHARING) {
            policy_str = "workload_aware";
        } else {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": Unknown policy enum.");
        }

        auto policy_perf_dir = perf_dir / policy_str;
        if (!std::filesystem::is_directory(policy_perf_dir)) {
            std::filesystem::create_directory(policy_perf_dir);
        }

        auto result_path = policy_perf_dir / file_name;

        std::ofstream file(result_path);
        file << std::setw(4) << json << std::endl;
    }
};

#endif // TALLY_CACHE_STRUCT_H