#ifndef TALLY_KERNEL_SLICE_H
#define TALLY_KERNEL_SLICE_H

#include <string>

#include <cuda.h>

std::string write_cubin_to_file(const char* data, uint32_t size);
std::vector<std::string> gen_ptx_from_cubin(std::string cubin_path);
void gen_sliced_ptx(std::string ptx_path, std::string output_path);
std::map<std::string, std::pair<CUfunction, uint32_t>> register_ptx(std::string ptx_path);

#endif // TALLY_KERNEL_SLICE_H