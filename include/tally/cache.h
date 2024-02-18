#ifndef TALLY_CACHE_H
#define TALLY_CACHE_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <tally/log.h>
#include <tally/env.h>
#include <tally/util.h>
#include <tally/cache_struct.h>
#include <tally/serialization.h>

template <typename T>
void load_cache_from_file(std::string file_name, T &_cache) {
    std::ifstream cache_file(file_name);
    if (cache_file.is_open()) {
        boost::archive::text_iarchive archive(cache_file);
        archive >> _cache;
        cache_file.close();
    }
}

// Only server will save cache
template <typename T>
void save_cache_to_file(std::string file_name, T &_cache) {
    std::ofstream cache_file(file_name);
    boost::archive::text_oarchive archive(cache_file);
    archive << _cache;
    cache_file.close();
}

class CubinCache
{
public:
    CubinCache();

    void save();
    void add_data(size_t cubin_size, std::string &cubin_str, std::map<std::string, std::vector<uint32_t>> &kernel_args);
    
    CubinData* find_transform_data(const char* cubin_data, size_t cubin_size);

    bool contains(const char* cubin_data, size_t cubin_size);
    std::map<std::string, std::vector<uint32_t>>& get_kernel_args(const char* cubin_data, size_t cubin_size);
    std::string &get_transform_ptx_str(const char* cubin_data, size_t cubin_size);
    std::string &get_transform_fatbin_str(const char* cubin_data, size_t cubin_size);
    const char *get_cubin_data_ptr(const char* cubin_data, size_t cubin_size);
    std::string get_cubin_data_str_from_cubin_uid(uint32_t cubin_uid);
    const char *get_cubin_data_str_ptr_from_cubin_uid(uint32_t cubin_uid);
    size_t get_cubin_size_from_cubin_uid(uint32_t cubin_uid);
    uint32_t get_cubin_data_uid(const char* cubin_data, size_t cubin_size);

    void load_cubin_cache(size_t cubin_size);
    void save_cubin_cache(size_t cubin_size);

private:
    std::string cache_dir;
    std::string uid_cache_file_name;

    // id counter
    uint32_t uid_counter = 0;

    // Cubin size : Cubin data
    std::map<size_t, std::vector<CubinData>> cubin_map;

    std::vector<size_t> changed_cubin_sizes;
};

class TallyCache
{
public:
    static std::shared_ptr<TallyCache> cache;

    TallyCache();

    CubinCache &get_cubin_cache();
    PerformanceCache &get_performance_cache();

    void save_transform_cache();
    void save_performance_cache();

private:
    CubinCache cubin_cache;
    PerformanceCache performance_cache;

    std::string performance_cache_file;
    bool performance_cache_loaded = false;
};

#endif // TALLY_CACHE_H