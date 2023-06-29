#ifndef TALLY_KERNEL_SLICE_H
#define TALLY_KERNEL_SLICE_H

#include <string>
#include <cstdio>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <cuda.h>

#include <tally/util.h>

void write_binary_to_file(std::string path, const char* data, uint32_t size);
void write_str_to_file(std::string path, std::string str);
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);
std::string gen_sliced_ptx(std::string ptx_path);
std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> register_kernels_from_ptx_fatbin(std::vector<std::pair<std::string, std::string>> &sliced_ptx_fatbin_strs, std::map<std::string, const void *> &kernel_name_map);
std::vector<std::pair<std::string, uint32_t>> get_kernel_names_and_nparams_from_ptx(std::string &ptx_str);
std::vector<std::pair<std::string, std::vector<uint32_t>>> get_kernel_names_and_param_sizes_from_elf(std::string elf_file_name);

class CubinCache
{
public:

    static CubinCache *cache;

    // Cubin size : vec<(cubin data, vec<(sliced_ptx, sliced_fatbin)>)>
    std::map<size_t, std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>>> sliced_ptx_cache;
    std::string cache_file;

    CubinCache() :
        cache_file("CubinCache.txt")
    {
        std::ifstream file(cache_file);
        if (file.is_open()) {
            boost::archive::text_iarchive archive(file);
            archive >> sliced_ptx_cache;
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
        archive << sliced_ptx_cache;
    }

    std::vector<std::pair<std::string, std::string>> get_sliced_ptx_fatbin_strs(const char* cubin_data, size_t cubin_size)
    {
        std::vector<std::pair<std::string, std::string>> strs;

        if (sliced_ptx_cache.find(cubin_size) != sliced_ptx_cache.end()) {
            for (auto &_pair : sliced_ptx_cache[cubin_size]) {
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
                write_str_to_file("/tmp/output.ptx", sliced_ptx_str);
                auto res = exec("nvcc /tmp/output.ptx --fatbin -arch sm_86 -o /tmp/output.fatbin");

                if (res.second != 0) {
                    throw std::runtime_error("Fail to compile PTX.");
                }

                std::ifstream ifs("/tmp/output.fatbin", std::ios::binary);
                auto sliced_fatbin_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

                strs.push_back(std::make_pair(sliced_ptx_str, sliced_fatbin_str));
                std::remove(ptx_file_name.c_str());
                std::remove("/tmp/output.ptx");
                std::remove("/tmp/output.fatbin");
            }

            sliced_ptx_cache[cubin_size].push_back(std::make_pair(cubin_str, strs));

            // Avoid too much I/O overhead on very small update.
            if (cubin_size > 4194304) {
                save_cache();
            }
        }

        return strs;
    }
};

#endif // TALLY_KERNEL_SLICE_H