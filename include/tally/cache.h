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

static std::shared_mutex cache_mutex;

class TallyCache
{
public:
    static std::shared_ptr<TallyCache> cache;

    CubinCache cubin_cache;
    PerformanceCache performance_cache;

    bool transform_cache_changed = false;
    bool perf_cache_changed = false;

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
        if (transform_cache_changed) {
            std::unique_lock lock(mutex_);
            spdlog::info("Saving transform cache ...");
            
            save_cache_to_file<CubinCache>(cubin_cache_file, cubin_cache);
            transform_cache_changed = false;
        }
    }

    void save_performance_cache() {
        if (perf_cache_changed) {
            std::unique_lock lock(mutex_);
            spdlog::info("Saving performance cache ...");

            save_cache_to_file<PerformanceCache>(performance_cache_file, performance_cache);
            performance_cache.write_single_kernel_perf_to_file();
            performance_cache.write_single_kernel_best_config_to_file();
            performance_cache.write_kernel_pair_perf_to_file();
            performance_cache.write_kernel_pair_best_config_to_file();

            perf_cache_changed = false;
        }
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

    TallyCache()
    {
        register_env_vars();

        auto policy = SCHEDULER_POLICY;

        auto home_path = std::filesystem::path(getenv("HOME"));
        auto cache_path = home_path / ".cache/tally";
        if (!std::filesystem::is_directory(cache_path)) {
            std::filesystem::create_directory(cache_path);
        }

        cubin_cache_file = (cache_path / ".tally_cache").string();
        cubin_cache_file_client = (cache_path / ".tally_cache_client").string();

        if (policy == TALLY_SCHEDULER_POLICY::PROFILE) {
            performance_cache_file = (cache_path / ".tally_perf_cache_profile").string();
        } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AGNOSTIC_SHARING) {
            performance_cache_file = (cache_path / ".tally_perf_cache_workload_agnostic").string();
        } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AWARE_SHARING) {
            performance_cache_file = (cache_path / ".tally_perf_cache_workload_aware").string();
        }

        load_cache();
    }

    ~TallyCache() {}
};

#endif // TALLY_CACHE_H