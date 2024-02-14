#ifndef TALLY_CACHE_UTIL_H
#define TALLY_CACHE_UTIL_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tally/log.h>
#include <tally/transform.h>
#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cache.h>

static void cache_cubin_data(const char* cubin_data, size_t cubin_size, int elf_filename=-1)
{
    // Write cubin data to file
    std::string cubin_tmp_path = get_tmp_file_path(".cubin");
    write_binary_to_file(cubin_tmp_path, cubin_data, cubin_size);
    
    // Extract elf code from cubin file
    std::string tmp_elf_file_name = get_tmp_file_path(".elf", elf_filename);
    exec(
        "cuobjdump " + cubin_tmp_path + " -elf" + 
            " -arch sm_" + std::string(CUDA_COMPUTE_VERSION) +
            " > " + tmp_elf_file_name
    );

    // If already exists, return early
    if (TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size)) {
        
        // Delete cubin file
        std::remove(cubin_tmp_path.c_str());

        return;
    }

    TALLY_SPD_LOG_ALWAYS("Caching fatbin data of size " + std::to_string(cubin_size) + " ...");

    // Delete cubin file
    std::remove(cubin_tmp_path.c_str());

    // These four objects will be cached
    std::string cubin_str(cubin_data, cubin_size);
    std::map<std::string, std::vector<uint32_t>> kernel_args;

    // Parse arguments info from elf code
    kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file_name);

    TallyCache::cache->cubin_cache.add_data(cubin_size, cubin_str, kernel_args);
    TallyCache::cache->transform_cache_changed = true;
}

#endif // TALLY_CACHE_UTIL_H