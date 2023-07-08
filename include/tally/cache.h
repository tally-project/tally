#ifndef TALLY_CACHE_H
#define TALLY_CACHE_H

#include <iostream>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <tally/cache_struct.h>
#include <tally/serialization.h>

class TallyCache
{
public:
    static std::shared_ptr<TallyCache> cache;

    CubinCache cubin_cache;
    PerformanceCache performance_cache;

    std::string transform_cache_file;
    std::string performance_cache_file;

    template <typename T>
    void load_cache_from_file(std::string file_name, T &cache) {
        std::ifstream file(file_name);
        if (file.is_open()) {
            boost::archive::text_iarchive archive(file);
            archive >> cache;
            file.close();
        }
    }

    template <typename T>
    void save_cache_to_file(std::string file_name, T &cache) {
        std::ofstream file(file_name);
        boost::archive::text_oarchive archive(file);
        archive << cache;
    }

    void save_transform_cache() {
        save_cache_to_file<CubinCache>(transform_cache_file, cubin_cache);
    }

    void save_performance_cache() {
        save_cache_to_file<PerformanceCache>(performance_cache_file, performance_cache);
    }

    void load_cache() {
        load_cache_from_file<CubinCache>(transform_cache_file, cubin_cache);
        load_cache_from_file<PerformanceCache>(performance_cache_file, performance_cache);
    }

    TallyCache() :
        transform_cache_file(".tally_cache"),
        performance_cache_file(".tally_perf_cache")
    {
        load_cache();
    }

    ~TallyCache() {}
};

#endif // TALLY_CACHE_H