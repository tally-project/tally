#ifndef TALLY_CACHE_STRUCT_H
#define TALLY_CACHE_STRUCT_H

#include <map>
#include <unordered_map>
#include <vector>
#include <string>

#include <tally/cuda_launch.h>

struct CudaLaunchKey {
    std::string kernel_name;
    dim3 gridDim;
    dim3 blockDim;

    bool operator==(const CudaLaunchKey &other) const
    {
        return (
            kernel_name == other.kernel_name
            && gridDim.x == other.gridDim.x
            && gridDim.y == other.gridDim.y
            && gridDim.z == other.gridDim.z
            && blockDim.x == other.blockDim.x
            && blockDim.y == other.blockDim.y
            && blockDim.z == other.blockDim.z
        );
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

struct CudaLaunchKeyConfigPairResult {
    std::pair<CudaLaunchKeyConfig, float> config_key_norm_speed_1;
    std::pair<CudaLaunchKeyConfig, float> config_key_norm_speed_2;

    float get_sum_norm_speed() const {
        return config_key_norm_speed_1.second + config_key_norm_speed_2.second;
    }
};

static std::ostream& operator<<(std::ostream& os, const CudaLaunchKeyConfigPairResult& res)
{
    float norm_speed_1 = res.config_key_norm_speed_1.second;
    float norm_speed_2 = res.config_key_norm_speed_2.second;

    os << "CudaLaunchKeyConfigPairResult: \n";
    os << "\tK1 Norm Speed: " << norm_speed_1 << " K2 Norm Speed: " << norm_speed_2 << " Sum: " << res.get_sum_norm_speed();

    return os;
}

// For each cubin file
// Don't forget to modify serialization.h when modifying this struct
class CubinData
{
public:
    // magic
    int magic;

    // version
    int version;

    // a cubin file
    std::string cubin_data;

    // Key: kernel name, value: vector of the sizes of arguments in ordinal order
    std::map<std::string, std::vector<uint32_t>> kernel_args;

    // All the sliced PTX files and fatbin
    std::vector<std::pair<std::string, std::string>> sliced_data;

    // All the PTB PTX files and fatbin
    std::vector<std::pair<std::string, std::string>> ptb_data;
};

class CubinCache
{
public:
    // Cubin size : Cubin data
    std::map<size_t, std::vector<CubinData>> cubin_map;

    bool contains(const char* cubin_data, size_t cubin_size)
    {
        return find_transform_data(cubin_data, cubin_size) != nullptr;
    }

    CubinData* find_transform_data(const char* cubin_data, size_t cubin_size)
    {
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
    get_sliced_data(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->sliced_data;
    }

    std::map<std::string, std::vector<uint32_t>>
    get_kernel_args(const char* cubin_data, size_t cubin_size)
    {
        auto transform_data = find_transform_data(cubin_data, cubin_size);
        assert(transform_data);
        return transform_data->kernel_args;
    }

    void add_data(
        size_t cubin_size,
        int magic,
        int version,
        std::string &cubin_str,
        std::map<std::string, std::vector<uint32_t>> &kernel_args,
        std::vector<std::pair<std::string, std::string>> &sliced_data,
        std::vector<std::pair<std::string, std::string>> &ptb_data
    )
    {
        cubin_map[cubin_size].push_back( CubinData { magic, version, cubin_str, kernel_args, sliced_data, ptb_data } );
    }
};

class PerformanceCache
{
public:
    // Kernel pair and the normalized speed
    std::unordered_map<CudaLaunchKeyPair, std::unordered_map<CudaLaunchKeyConfigPair, CudaLaunchKeyConfigPairResult>> kernel_pair_perf_map;
    std::unordered_map<CudaLaunchKeyPair, CudaLaunchKeyConfigPairResult> kernel_pair_best_config_map;

    void set_kernel_pair_perf(CudaLaunchKey &launch_key_1, CudaLaunchKey &launch_key_2, CudaLaunchConfig launch_config_1, CudaLaunchConfig launch_config_2, CudaLaunchKeyConfigPairResult &res)
    {
        CudaLaunchKeyPair key_pair(launch_key_1, launch_key_2);

        CudaLaunchKeyConfig key_config_1(launch_key_1, launch_config_1);
        CudaLaunchKeyConfig key_config_2(launch_key_2, launch_config_2);

        CudaLaunchKeyConfigPair key_config_pair(key_config_1, key_config_2);

        kernel_pair_perf_map[key_pair][key_config_pair] = res;

        std::cout << "=============== Kernel Pair Profile Result ===============" << std::endl;

        std::cout << launch_key_1 << std::endl;
        std::cout << launch_config_1 << std::endl;
        std::cout << launch_key_2 << std::endl;
        std::cout << launch_config_2 << std::endl;
        std::cout << res << std::endl;
        std::cout << std::endl;
    }

    void set_kernel_pair_best_config(CudaLaunchKey &launch_key_1, CudaLaunchKey &launch_key_2, CudaLaunchKeyConfigPairResult &res)
    {
        CudaLaunchKeyPair key_pair(launch_key_1, launch_key_2);
        kernel_pair_best_config_map[key_pair] = res;

        std::cout << "=============== Kernel Pair Best Result ===============" << std::endl;

        auto launch_config_1 = res.config_key_norm_speed_1.first.config;
        auto launch_config_2 = res.config_key_norm_speed_2.first.config;

        std::cout << launch_key_1 << std::endl;
        std::cout << launch_config_1 << std::endl;
        std::cout << launch_key_2 << std::endl;
        std::cout << launch_config_2 << std::endl;
        std::cout << res << std::endl;
        std::cout << std::endl;
    }

};

#endif // TALLY_CACHE_STRUCT_H