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

// Generating PTB version of a PTX file
std::string gen_ptb_ptx(std::string &ptx_str);

// Generating dynamic PTB version of a PTX file
std::string gen_dynamic_ptb_ptx(std::string &ptx_str);

// Generating preemptive PTB version of a PTX file
std::string gen_preemptive_ptb_ptx(std::string &ptx_str);

// Generating combined version of a PTX file
std::string gen_transform_ptx(std::string &ptx_path);

#endif // TALLY_TRANSFORM_H