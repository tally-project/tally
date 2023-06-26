#include <string>

#include <tally/const.h>

#define REGISTER_UINT32_ENV_VAR(NAME, DEFAULT_VAL) \
    uint32_t NAME = std::getenv(#NAME) ? std::stoi(std::getenv(#NAME)) : DEFAULT_VAL;

#define REGISTER_BOOL_ENV_VAR(NAME, DEFAULT_VAL) \
    bool NAME = std::getenv(#NAME) ? (std::string(std::getenv(#NAME)) == "TRUE") : DEFAULT_VAL;

REGISTER_UINT32_ENV_VAR(THREADS_PER_SLICE, 196608);
REGISTER_BOOL_ENV_VAR(USE_CUDA_GRAPH, false);
REGISTER_UINT32_ENV_VAR(KERNEL_SLICE_THRESHOLD, 3);
