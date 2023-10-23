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
    exec("cuobjdump " + cubin_tmp_path + " -elf > " + tmp_elf_file_name);

    // If already exists, return early
    if (TallyCache::cache->cubin_cache.contains(cubin_data, cubin_size)) {
        
        // Delete cubin file
        std::remove(cubin_tmp_path.c_str());

        return;
    }

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

    if (ptx_file_names.size() > 1) {
        throw std::runtime_error("Assuming one fatbin only contains one ptx file");
    }

    if (ptx_file_names.size() > 0) {

        auto ptx_file_name = ptx_file_names[0];

        // Generate transformed version from the PTX code
        spdlog::info("Generating transformed code for " + ptx_file_name + " ...");

        auto transform_ptx_str = gen_transform_ptx(ptx_file_name);
        auto transform_fatbin_str = get_fatbin_str_from_ptx_str(transform_ptx_str);

        // Delete ptx
        std::remove(ptx_file_name.c_str());

        TallyCache::cache->cubin_cache.add_data(cubin_size, cubin_str, kernel_args, transform_ptx_str, transform_fatbin_str);
        TallyCache::cache->transform_cache_changed = true;
    } else {

        spdlog::warn("No PTX file found in fatbin data.");

        std::string transform_ptx_str = "";
        std::string transform_fatbin_str = "";

        TallyCache::cache->cubin_cache.add_data(cubin_size, cubin_str, kernel_args, transform_ptx_str, transform_fatbin_str);
        TallyCache::cache->transform_cache_changed = true;

    }

    TallyCache::cache->save_transform_cache();
}

#endif // TALLY_CACHE_UTIL_H