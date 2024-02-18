#include <vector>

#include <tally/cache.h>
#include <tally/util.h>
#include <tally/consts.h>

std::shared_ptr<TallyCache> TallyCache::cache = std::make_shared<TallyCache>();

CubinCache::CubinCache()
{
    auto home_path = std::filesystem::path(getenv("HOME"));
    cache_dir = (home_path / ".cache/tally").string();

    // Try to load uid counter
    uid_cache_file_name = cache_dir + "/uid_count.txt";
    std::ifstream uid_cache_file(uid_cache_file_name);

    if (uid_cache_file.is_open()) {
        uid_cache_file >> uid_counter;
        uid_cache_file.close();
    }
}

void CubinCache::load_cubin_cache(size_t cubin_size) {

    auto has_cubin_size = cubin_map.find(cubin_size) != cubin_map.end();

    // if not exists, look up file system
    if (!has_cubin_size) {
        auto cache_file_name = std::to_string(cubin_size) + ".cache";
        auto cache_path = cache_dir + "/" + cache_file_name;

        std::vector<CubinData> cubin_data_vec;
        load_cache_from_file<std::vector<CubinData>>(cache_path, cubin_data_vec);
        cubin_map[cubin_size] = cubin_data_vec;

    }
}

void CubinCache::save_cubin_cache(size_t cubin_size) {
    auto cache_file_name = std::to_string(cubin_size) + ".cache";
    auto cache_path = cache_dir + "/" + cache_file_name;

    save_cache_to_file<std::vector<CubinData>>(cache_path, cubin_map[cubin_size]);
}

void CubinCache::save()
{
    if (changed_cubin_sizes.size() > 0) {

        TALLY_SPD_LOG_ALWAYS("Saving transform cache ...");

        // save uid_counter
        std::ofstream uid_cache_file(uid_cache_file_name);
        uid_cache_file << uid_counter;
        uid_cache_file.close();

        for (auto cubin_size : changed_cubin_sizes) {
            save_cubin_cache(cubin_size);
        }

        changed_cubin_sizes.clear();
    }
}

void CubinCache::add_data(
    size_t cubin_size,
    std::string &cubin_str,
    std::map<std::string, std::vector<uint32_t>> &kernel_args
)
{
    std::unique_lock lock(mutex_);
    cubin_map[cubin_size].push_back( CubinData { uid_counter, cubin_str, cubin_size, kernel_args } );
    uid_counter++;
    changed_cubin_sizes.push_back(cubin_size);
}

CubinData* CubinCache::find_transform_data(const char* cubin_data, size_t cubin_size)
{
    std::shared_lock lock(mutex_);

    auto has_cubin_size = cubin_map.find(cubin_size) != cubin_map.end();

    // if not exists, look up file system
    if (!has_cubin_size) {
        load_cubin_cache(cubin_size);
    }

    for (auto &data : cubin_map[cubin_size]) {
        if (memcmp(data.cubin_data.c_str(), cubin_data, cubin_size) == 0) {
            return &data;
        }
    }

    return nullptr;
}

bool CubinCache::contains(const char* cubin_data, size_t cubin_size)
{
    return find_transform_data(cubin_data, cubin_size) != nullptr;
}

std::map<std::string, std::vector<uint32_t>>&
CubinCache::get_kernel_args(const char* cubin_data, size_t cubin_size)
{
    auto transform_data = find_transform_data(cubin_data, cubin_size);
    assert(transform_data);

    if (!transform_data) {
        throw std::runtime_error("cannot find transform_data");
    }

    return transform_data->kernel_args;
}

std::string &CubinCache::get_transform_ptx_str(const char* cubin_data, size_t cubin_size)
{
    auto transform_data = find_transform_data(cubin_data, cubin_size);
    assert(transform_data);

    if (!transform_data) {
        throw std::runtime_error("cannot find transform_data");
    }

    if (!transform_data->compiled) {
        transform_data->compile();
        changed_cubin_sizes.push_back(cubin_size);
    }

    return transform_data->ptx_str;
}

std::string &CubinCache::get_transform_fatbin_str(const char* cubin_data, size_t cubin_size)
{
    auto transform_data = find_transform_data(cubin_data, cubin_size);
    assert(transform_data);

    if (!transform_data) {
        throw std::runtime_error("cannot find transform_data");
    }

    if (!transform_data->compiled) {
        transform_data->compile();
        changed_cubin_sizes.push_back(cubin_size);
    }

    return transform_data->fatbin_str;
}

const char *CubinCache::get_cubin_data_ptr(const char* cubin_data, size_t cubin_size)
{
    auto transform_data = find_transform_data(cubin_data, cubin_size);
    assert(transform_data);
    return transform_data->cubin_data.c_str();
}

std::string CubinCache::get_cubin_data_str_from_cubin_uid(uint32_t cubin_uid)
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

const char *CubinCache::get_cubin_data_str_ptr_from_cubin_uid(uint32_t cubin_uid)
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

size_t CubinCache::get_cubin_size_from_cubin_uid(uint32_t cubin_uid)
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

uint32_t CubinCache::get_cubin_data_uid(const char* cubin_data, size_t cubin_size)
{
    auto transform_data = find_transform_data(cubin_data, cubin_size);
    assert(transform_data);
    return transform_data->cubin_uid;
}

CubinCache &TallyCache::get_cubin_cache() {
    return cubin_cache;
}

PerformanceCache &TallyCache::get_performance_cache() {
    if (!performance_cache_loaded) {
        load_cache_from_file<PerformanceCache>(performance_cache_file, performance_cache);
        performance_cache_loaded = true;
    }
    return performance_cache;
}

void TallyCache::save_transform_cache() {
    cubin_cache.save();
}

void TallyCache::save_performance_cache() {
    if (performance_cache.changed) {
        std::unique_lock lock(mutex_);
        TALLY_SPD_LOG_ALWAYS("Saving performance cache ...");

        save_cache_to_file<PerformanceCache>(performance_cache_file, performance_cache);
        performance_cache.write_single_kernel_perf_to_file();
        // performance_cache.write_single_kernel_best_config_to_file();
        performance_cache.write_kernel_pair_perf_to_file();
        performance_cache.write_kernel_pair_best_config_to_file();

        performance_cache.changed = false;
    }
}

TallyCache::TallyCache()
{
    register_env_vars();

    auto home_path = std::filesystem::path(getenv("HOME"));
    auto cache_path = home_path / ".cache/tally";
    if (!std::filesystem::is_directory(cache_path)) {
        std::filesystem::create_directory(cache_path);
    }

    auto policy = SCHEDULER_POLICY;
    if (policy == TALLY_SCHEDULER_POLICY::PROFILE) {
        performance_cache_file = (cache_path / ".tally_perf_cache_profile").string();
    } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AGNOSTIC_SHARING) {
        performance_cache_file = (cache_path / ".tally_perf_cache_workload_agnostic").string();
    } else if (policy == TALLY_SCHEDULER_POLICY::WORKLOAD_AWARE_SHARING) {
        performance_cache_file = (cache_path / ".tally_perf_cache_workload_aware").string();
    } else if (policy == TALLY_SCHEDULER_POLICY::PRIORITY) {
        performance_cache_file = (cache_path / ".tally_perf_cache_priority").string();
    }
}