#ifndef TALLY_CUDA_UTIL_H
#define TALLY_CUDA_UTIL_H

#include <string>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

void write_cubin_to_file(const char *cubin_data, uint32_t cubin_size);

std::string get_fatbin_str_from_ptx_str(std::string ptx_str);

std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);

std::unordered_map<const void *, std::pair<CUfunction, uint32_t>> register_kernels_from_ptx_fatbin(
    std::vector<std::pair<std::string, std::string>> &ptx_fatbin_strs,
    std::map<std::string, const void *> &kernel_name_map
);

std::vector<std::pair<std::string, uint32_t>> get_kernel_names_and_nparams_from_ptx(std::string &ptx_str);

std::map<std::string, std::vector<uint32_t>> get_kernel_names_and_param_sizes_from_elf(std::string elf_path);

#endif // TALLY_CUDA_UTIL_H