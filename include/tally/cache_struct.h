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
    // Best launch config for each CudaLaunchKey
    std::unordered_map<CudaLaunchKey, CudaLaunchConfig> kernel_config_map;

    // Latency (ms) for kernel and config pair
    std::unordered_map<CudaLaunchKeyConfig, float> config_latency_map;

    void set_execution_time(CudaLaunchKeyConfig& config, float time_ms)
    {
        config_latency_map[config] = time_ms;
    }

    void set_launch_config(CudaLaunchKey& key, CudaLaunchConfig& config)
    {
        kernel_config_map[key] = config;
    }
};

#endif // TALLY_CACHE_STRUCT_H