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

// Generating original version of a PTX file
std::string gen_original_ptx(std::string ptx_path);

// Generating sliced version of a PTX file
std::string gen_sliced_ptx(std::string ptx_path);

// Generating PTB version of a PTX file
std::string gen_ptb_ptx(std::string ptx_path);

#endif // TALLY_TRANSFORM_H