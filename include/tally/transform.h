#ifndef TALLY_TRANSFORM_H
#define TALLY_TRANSFORM_H

#include <string>
#include <cstdio>
#include <limits>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <fatbinary_section.h>

#include <tally/util.h>
#include <tally/cuda_util.h>
#include <tally/cuda_launch.h>
#include <tally/generated/cuda_api.h>

// Transform the kernel to make sure threads return at the same time
// This is to fix deadlock issue when some threads exit early while others block at __syncthreads()
std::string gen_sync_aware_kernel(std::string &ptx_str);

// Generate PTB version of a kernel
std::string gen_ptb_kernel(std::string &ptx_str);

// Generate dynamic PTB version of a kernel
std::string gen_dynamic_ptb_kernel(std::string &ptx_str);

// Generate preemptive PTB version of a kernel
std::string gen_preemptive_ptb_kernel(std::string &ptx_str);

// Generate sliced version of a kernel
std::string gen_sliced_kernel(std::string &ptx_str);

// Generate combined version of a PTX file
std::string gen_transform_ptx(std::string &ptx_path);

#endif // TALLY_TRANSFORM_H