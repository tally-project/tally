#ifndef TALLY_CACHE_UTIL_H
#define TALLY_CACHE_UTIL_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cache.h>

void cache_cubin_data(const char* cubin_data, size_t cubin_size, int magic, int version)
{
    // Not exist in cache
    if (TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size)) {
        return;
    }

    // Write cubin data to file
    std::string cubin_tmp_path = get_tmp_file_path(".cubin");
    write_binary_to_file(cubin_tmp_path, cubin_data, cubin_size);
    
    // Extract PTX code from cubin file
    auto ptx_file_names = gen_ptx_from_cubin(cubin_tmp_path);

    // Extract elf code from cubin file
    std::string tmp_elf_file_name = get_tmp_file_path(".elf");
    exec("cuobjdump " + cubin_tmp_path + " -elf > " + tmp_elf_file_name);

    // Delete cubin file
    std::remove(cubin_tmp_path.c_str());

    // These four objects will be cached
    std::string cubin_str(cubin_data, cubin_size);
    std::map<std::string, std::vector<uint32_t>> kernel_args;
    std::vector<std::pair<std::string, std::string>> sliced_data;
    std::vector<std::pair<std::string, std::string>> ptb_data;

    // Parse arguments info from elf code
    kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file_name);
    // Delete elf
    std::remove(tmp_elf_file_name.c_str());

    // Generate transformed version from the PTX code
    for (const auto& ptx_file_name : ptx_file_names) {
        // auto sliced_ptx_str = gen_sliced_ptx(ptx_file_name);
        // auto sliced_fatbin_str = get_fatbin_str_from_ptx_str(sliced_ptx_str);
        // sliced_data.push_back(std::make_pair(sliced_ptx_str, sliced_fatbin_str));

        // auto ptb_ptx_str = gen_ptb_ptx(ptx_file_name);
        // auto ptb_fatbin_str = get_fatbin_str_from_ptx_str(ptb_ptx_str);
        // ptb_data.push_back(std::make_pair(ptb_ptx_str, ptb_fatbin_str));

        // Delete ptx
        std::remove(ptx_file_name.c_str());
    }

    TallyCache::cache->cubin_cache.add_data(cubin_size, magic, version, cubin_str, kernel_args, sliced_data, ptb_data);
    TallyCache::cache->save_transform_cache();
}

#endif // TALLY_CACHE_UTIL_H