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

void write_cubin_to_file(std::string path, const char* data, uint32_t size);
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);
std::string gen_sliced_ptx(std::string ptx_path);
std::map<std::string, std::pair<CUfunction, uint32_t>> register_ptx(std::string &ptx_str);
std::vector<std::string> get_kernel_names_from_ptx(std::string &ptx_str);

class CubinCache
{
public:
    // Cubin size : vec<(cubin data, vec<sliced_ptx>)>
    std::map<size_t, std::vector<std::pair<std::string, std::vector<std::string>>>> sliced_ptx_cache;
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

    ~CubinCache() {
        save_cache();
    }

    void save_cache()
    {
        std::cout << "saving cache" << std::endl;
        std::ofstream file(cache_file);
        boost::archive::text_oarchive archive(file);
        archive << sliced_ptx_cache;
    }

    std::vector<std::string> get_sliced_ptx_strs(const char* cubin_data, size_t cubin_size)
    {
        std::vector<std::string> strs;

        if (sliced_ptx_cache.find(cubin_size) != sliced_ptx_cache.end()) {
            for (auto &_pair : sliced_ptx_cache[cubin_size]) {
                if (memcmp(_pair.first.c_str(), cubin_data, cubin_size) == 0) {
                    std::cout << "Cache hit" << std::endl;
                    return _pair.second;
                }
            }
        } else {
            std::string cubin_tmp_path("/tmp/output.cubin");
            write_cubin_to_file(cubin_tmp_path, cubin_data, cubin_size);
            auto ptx_file_names = gen_ptx_from_cubin(cubin_tmp_path);
            std::remove(cubin_tmp_path.c_str());

            std::string cubin_str(cubin_data, cubin_size);

            for (const auto& ptx_file_name : ptx_file_names) {
                auto sliced_ptx_str = gen_sliced_ptx(ptx_file_name);
                strs.push_back(sliced_ptx_str);
                std::remove(ptx_file_name.c_str());
            }

            sliced_ptx_cache[cubin_size].push_back(std::make_pair(cubin_str, strs));
            save_cache();
        }

        return strs;
    }
};

extern CubinCache cubin_cache;

#endif // TALLY_KERNEL_SLICE_H