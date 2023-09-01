#ifndef TALLY_CACHE_UTIL_H
#define TALLY_CACHE_UTIL_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <tally/transform.h>
#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cache.h>

static void cache_cubin_data(const char* cubin_data, size_t cubin_size, int elf_filename=-1)
{
    // If already exists, do nothing
    if (TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size)) {
        return;
    }

    // Write cubin data to file
    std::string cubin_tmp_path = get_tmp_file_path(".cubin");
    write_binary_to_file(cubin_tmp_path, cubin_data, cubin_size);
    
    // Extract elf code from cubin file
    std::string tmp_elf_file_name = get_tmp_file_path(".elf", elf_filename);
    exec("cuobjdump " + cubin_tmp_path + " -elf > " + tmp_elf_file_name);

    // Extract PTX code from cubin file
    auto ptx_file_names = gen_ptx_from_cubin(cubin_tmp_path);

    // Delete cubin file
    std::remove(cubin_tmp_path.c_str());

    // These four objects will be cached
    std::string cubin_str(cubin_data, cubin_size);
    std::map<std::string, std::vector<uint32_t>> kernel_args;
    std::vector<std::pair<std::string, std::string>> original_data;
    std::vector<std::pair<std::string, std::string>> ptb_data;
    std::vector<std::pair<std::string, std::string>> dynamic_ptb_data;
    std::vector<std::pair<std::string, std::string>> preemptive_ptb_data;

    // Parse arguments info from elf code
    kernel_args = get_kernel_names_and_param_sizes_from_elf(tmp_elf_file_name);

    // elf will be deleted by the client side
    // std::remove(tmp_elf_file_name.c_str());

    // Generate transformed version from the PTX code
    for (const auto& ptx_file_name : ptx_file_names) {

        auto original_ptx_str = gen_original_ptx(ptx_file_name);
        auto original_fatbin_str = get_fatbin_str_from_ptx_str(original_ptx_str);
        original_data.push_back(std::make_pair(original_ptx_str, original_fatbin_str));

        auto ptb_ptx_str = gen_ptb_ptx(ptx_file_name);
        auto ptb_fatbin_str = get_fatbin_str_from_ptx_str(ptb_ptx_str);
        ptb_data.push_back(std::make_pair(ptb_ptx_str, ptb_fatbin_str));

        auto dynamic_ptb_ptx_str = gen_dynamic_ptb_ptx(ptx_file_name);
        auto dynamic_ptb_fatbin_str = get_fatbin_str_from_ptx_str(dynamic_ptb_ptx_str);
        dynamic_ptb_data.push_back(std::make_pair(dynamic_ptb_ptx_str, dynamic_ptb_fatbin_str));

        auto preemptive_ptb_ptx_str = gen_preemptive_ptb_ptx(ptx_file_name);
        auto preemptive_ptb_fatbin_str = get_fatbin_str_from_ptx_str(preemptive_ptb_ptx_str);
        preemptive_ptb_data.push_back(std::make_pair(preemptive_ptb_ptx_str, preemptive_ptb_fatbin_str));

        // Delete ptx
        std::remove(ptx_file_name.c_str());
    }

    TallyCache::cache->cubin_cache.add_data(cubin_size, cubin_str, kernel_args, original_data, ptb_data, dynamic_ptb_data, preemptive_ptb_data);
    TallyCache::cache->save_transform_cache();
}

#endif // TALLY_CACHE_UTIL_H