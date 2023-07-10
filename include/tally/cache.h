#ifndef TALLY_CACHE_H
#define TALLY_CACHE_H

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <tally/util.h>
#include <tally/cache_struct.h>
#include <tally/serialization.h>

class TallyCache
{
public:
    static std::shared_ptr<TallyCache> cache;

    CubinCache cubin_cache;
    PerformanceCache performance_cache;

    std::string cubin_cache_file;
    std::string cubin_cache_file_client;
    std::string performance_cache_file;

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

    void save_transform_cache() {
        save_cache_to_file<CubinCache>(cubin_cache_file, cubin_cache);
    }

    void save_performance_cache() {
        save_cache_to_file<PerformanceCache>(performance_cache_file, performance_cache);
    }

    void load_cache() {
// If client, make a copy of the cache
// boost throws exception when client server both load the cache
#ifdef IS_CLIENT
        load_cache_from_file<CubinCache>(cubin_cache_file_client, cubin_cache);
#else
        load_cache_from_file<CubinCache>(cubin_cache_file, cubin_cache);
        load_cache_from_file<PerformanceCache>(performance_cache_file, performance_cache);
#endif
    }

    TallyCache() :
        cubin_cache_file(".tally_cache"),
        cubin_cache_file_client(".tally_cache_client"),
        performance_cache_file(".tally_perf_cache")
    {
        load_cache();
    }

    ~TallyCache() {}
};

#endif // TALLY_CACHE_H