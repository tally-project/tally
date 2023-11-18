#include <vector>

#include <tally/cache.h>
#include <tally/util.h>
#include <tally/consts.h>

std::shared_ptr<TallyCache> TallyCache::cache;

#if not defined(RUN_LOCALLY)
__attribute__((__constructor__)) void init_cache()
{
    NO_INIT_PROCESS_KEYWORDS_VEC;

    auto process_name = get_process_name(getpid());

    for (auto &keyword : no_init_process_keywords) {

        if (containsSubstring(process_name, keyword)) {
            return;
        }
    }

    std::ofstream cache_log_file("tally-cache.log", std::ios::app);
    cache_log_file << process_name << std::endl;
    cache_log_file.close();

    TallyCache::cache = std::make_shared<TallyCache>();
}
#endif