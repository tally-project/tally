#include <tally/cache.h>
#include <tally/cuda_util.h>
#include <tally/env.h>

std::shared_ptr<TallyCache> TallyCache::cache = std::make_shared<TallyCache>();