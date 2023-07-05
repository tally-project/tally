#ifndef TALLY_TRANSFORM_H
#define TALLY_TRANSFORM_H

#include <string>
#include <cstdio>
#include <limits>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <fatbinary_section.h>

#include <tally/util.h>
#include <tally/generated/cuda_api.h>

std::string get_fatbin_str_from_ptx_str(std::string ptx_str);
void write_binary_to_file(std::string path, const char* data, uint32_t size);
void write_str_to_file(std::string path, std::string str);
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);
std::string gen_sliced_ptx(std::string ptx_path);
std::string gen_ptb_ptx(std::string ptx_path);
std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> register_kernels_from_ptx_fatbin(std::vector<std::pair<std::string, std::string>> &ptx_fatbin_strs, std::map<std::string, const void *> &kernel_name_map);
std::vector<std::pair<std::string, uint32_t>> get_kernel_names_and_nparams_from_ptx(std::string &ptx_str);
std::vector<std::pair<std::string, std::vector<uint32_t>>> get_kernel_names_and_param_sizes_from_elf(std::string elf_file_name);

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

class LaunchConfig;

class CubinCache
{
public:

    static std::unique_ptr<CubinCache> cache;

    // Cubin size : vec<(cubin data, (vec<(sliced_ptx, sliced_fatbin)>, vec<(ptb_ptx, ptb_fatbin)>))>
    std::map<size_t,std::vector<std::pair<std::string, std::pair<
                        std::vector<std::pair<std::string, std::string>>,
                        std::vector<std::pair<std::string, std::string>>>>>>
        transform_cache;
    std::string cache_file;

    CubinCache() :
        cache_file(".tally_cache")
    {
        std::ifstream file(cache_file);
        if (file.is_open()) {
            boost::archive::text_iarchive archive(file);
            archive >> transform_cache;
            file.close();
        } else {
            std::cout << "cache not exists" << std::endl;
        }
    }

    ~CubinCache()
    {
        save_cache();
    }

    void save_cache()
    {
        std::ofstream file(cache_file);
        boost::archive::text_oarchive archive(file);
        archive << transform_cache;
    }

    std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>>
    get_transform_ptx_fatbin_strs(const char* cubin_data, size_t cubin_size)
    {

        std::pair<std::vector<std::pair<std::string, std::string>>, std::vector<std::pair<std::string, std::string>>> ptx_fatbin_strs;
        
        if (transform_cache.find(cubin_size) != transform_cache.end()) {
            for (auto &_pair : transform_cache[cubin_size]) {
                if (memcmp(_pair.first.c_str(), cubin_data, cubin_size) == 0) {
                    return _pair.second;
                }
            }
        } else {

            std::string cubin_tmp_path("/tmp/output.cubin");
            write_binary_to_file(cubin_tmp_path, cubin_data, cubin_size);
            auto ptx_file_names = gen_ptx_from_cubin(cubin_tmp_path);
            std::remove(cubin_tmp_path.c_str());

            std::string cubin_str(cubin_data, cubin_size);

            for (const auto& ptx_file_name : ptx_file_names) {
                auto sliced_ptx_str = gen_sliced_ptx(ptx_file_name);
                auto sliced_fatbin_str = get_fatbin_str_from_ptx_str(sliced_ptx_str);
                ptx_fatbin_strs.first.push_back(std::make_pair(sliced_ptx_str, sliced_fatbin_str));

                auto ptb_ptx_str = gen_ptb_ptx(ptx_file_name);
                auto ptb_fatbin_str = get_fatbin_str_from_ptx_str(ptb_ptx_str);
                ptx_fatbin_strs.second.push_back(std::make_pair(ptb_ptx_str, ptb_fatbin_str));

                std::remove(ptx_file_name.c_str());
            }

            transform_cache[cubin_size].push_back(std::make_pair(cubin_str, ptx_fatbin_strs));

            // Avoid too much I/O overhead on very small update.
            if (cubin_size > 4194304) {
                save_cache();
            }
        }

        return ptx_fatbin_strs;
    }
};

struct CudaLaunchCall {
    const void *func;
    dim3 gridDim;
    dim3 blockDim;

    bool operator==(const CudaLaunchCall &other) const
    { return (func == other.func
                && gridDim.x == other.gridDim.x
                && gridDim.y == other.gridDim.y
                && gridDim.z == other.gridDim.z
                && blockDim.x == other.blockDim.x
                && blockDim.y == other.blockDim.y
                && blockDim.z == other.blockDim.z);
    }
};

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

class LaunchConfig {
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

    friend std::ostream& operator<<(std::ostream& os, const LaunchConfig& config) {
        os << "LaunchConfig: ";
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

    LaunchConfig(bool use_original=true, bool use_sliced=false, bool use_ptb=false, bool use_cuda_graph=false, uint32_t threads_per_slice=0, uint32_t num_blocks_per_sm=0) :
        use_original(use_original), use_sliced(use_sliced), use_ptb(use_ptb), use_cuda_graph(use_cuda_graph),
        threads_per_slice(threads_per_slice), num_blocks_per_sm(num_blocks_per_sm)
    {}

    cudaError_t launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, bool run_profile=false, float *elapsed_time_ms=nullptr);
    static LaunchConfig tune(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    std::pair<float, float> repeat_launch(const void *, dim3, dim3, void **, size_t, cudaStream_t, float dur_seconds, uint32_t max_count=-1);
};

class Transform {

public:

    static std::unique_ptr<Transform> tracer;

    uint32_t curr_kernel_idx = 0;
    std::unordered_map<CudaLaunchCall, float> kernel_baseline_performance;
    std::unordered_map<CudaLaunchCall, LaunchConfig> kernel_profile_map;
    std::map<std::string, const void *> kernel_name_to_host_func_map;
    std::map<const void *, std::string> host_func_to_kernel_name_map;
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> sliced_kernel_map;
    std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> ptb_kernel_map;
    std::vector<CudaGraphCall*> cuda_graph_vec;
    cudaStream_t stream;

    std::vector<std::pair<std::string, std::string>> sliced_ptx_fatbin_strs;
    std::vector<std::pair<std::string, std::string>> ptb_ptx_fatbin_strs;
    bool kernels_registered = false;

    void register_kernels()
    {
        lcudaStreamCreate(&stream);

        ptb_kernel_map = register_kernels_from_ptx_fatbin(ptb_ptx_fatbin_strs, kernel_name_to_host_func_map);
        sliced_kernel_map = register_kernels_from_ptx_fatbin(sliced_ptx_fatbin_strs, kernel_name_to_host_func_map);
        kernels_registered = true;
    }

    Transform(){}
    ~Transform(){}
};

#endif // TALLY_TRANSFORM_H